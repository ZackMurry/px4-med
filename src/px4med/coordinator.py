"""Coordinator — owns both Drone instances, drives the per-step inference loop.

Each tick:
  1. Gather telemetry from both drones.
  2. Build joint state vector (280 floats).
  3. Run policy inference → action for each agent.
  4. Dispatch waypoint offsets (or land command).
  5. Check termination conditions.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from .actions import CRUISE_DOWN_M, action_to_offset, is_land_action
from .drone import Telemetry
from .metrics import StepRecord
from .state import build_state

if TYPE_CHECKING:
    from .drone import Drone
    from .environment import WorldEnvironment
    from .metrics import MetricsCollector
    from .policy import PolicyNet

logger = logging.getLogger(__name__)

# Agent start positions from AneeshMARL5.py fixed_layout: grid (x, y)
# Agent 0 at (1,1), Agent 1 at (1,13)
_DEFAULT_START_GRIDS: list[tuple[int, int]] = [(1, 1), (1, 13)]


class Coordinator:
    """Drives both drones through the RL policy for one episode."""

    def __init__(
        self,
        drones: list[Drone],
        policy: PolicyNet,
        world: WorldEnvironment,
        metrics: MetricsCollector,
        step_hz: float = 2.0,
    ) -> None:
        self.drones = drones
        self.policy = policy
        self.world = world
        self.metrics = metrics
        self.step_interval = 1.0 / step_hz

    async def run_episode(self, episode: int = 0, max_steps: int = 800) -> dict:
        """Arm, take off, run RL loop, land all. Return summary dict."""
        self.world.reset()

        # Arm and take off both drones concurrently
        await asyncio.gather(*(d.arm() for d in self.drones))
        await asyncio.gather(*(d.takeoff() for d in self.drones))

        # Reposition to training-env start positions before the RL loop.
        # The training env initialises agents at grid (1,1) and (1,13);
        # SITL spawns both at NED (0,0), which the policy has never seen.
        mpc = float(self.world.config.get("grid", {}).get("meters_per_cell", 2.0))
        start_grids = self.world.config.get("agent_start_positions", _DEFAULT_START_GRIDS)
        logger.info("Repositioning drones to training start positions: %s", start_grids)
        await asyncio.gather(*(
            self.drones[i].send_waypoint(
                -start_grids[i][1] * mpc,   # north = -grid_y * mpc
                start_grids[i][0] * mpc,    # east  =  grid_x * mpc
                CRUISE_DOWN_M,
            )
            for i in range(len(self.drones))
        ))

        landed = [False, False]
        step = 0
        loop = asyncio.get_running_loop()

        while step < max_steps and not all(landed):
            step_start = loop.time()

            # 1. Gather telemetry
            telems: list[Telemetry] = list(
                await asyncio.gather(*(d.get_telemetry() for d in self.drones))
            )

            # 2. Check deliveries against world
            step_deliveries: list[int] = []
            for i, telem in enumerate(telems):
                if landed[i]:
                    continue
                pid = self.world.check_delivery(i, telem.north_m, telem.east_m)
                if pid is not None:
                    step_deliveries.append(pid)
                    logger.info(
                        "Episode %d step %d: drone %d delivered patient %d",
                        episode, step, i, pid,
                    )

            # 3. Build per-agent state vectors
            states = [
                build_state(i, telems[i], telems[1 - i], self.world)
                for i in range(2)
            ]

            # 4. Policy inference (joint state: self ‖ other)
            actions = [
                self.policy.select_action(states[i], states[1 - i])
                for i in range(2)
            ]
            logger.debug("Episode %d step %d: actions=%s", episode, step, actions)

            # 5. Log step
            self.metrics.log_step(StepRecord(
                episode=episode,
                step=step,
                timestamp=time.time(),
                drone0_north=telems[0].north_m,
                drone0_east=telems[0].east_m,
                drone0_battery=telems[0].battery_pct,
                drone1_north=telems[1].north_m,
                drone1_east=telems[1].east_m,
                drone1_battery=telems[1].battery_pct,
                actions=actions,
                deliveries=step_deliveries,
            ))

            # 6. Dispatch actions concurrently
            await asyncio.gather(*(
                self._dispatch(i, actions[i], telems[i], landed)
                for i in range(2)
            ))
            # Mark drones that chose to land as landed
            for i in range(2):
                if is_land_action(actions[i]):
                    landed[i] = True

            # 7. Advance world state (timers, hazard zones, new patients)
            self.world.step()

            step += 1

            # 8. Pace at step_hz (sleep any remaining budget in this interval)
            elapsed = loop.time() - step_start
            remaining = self.step_interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)

        # Land any drones still airborne
        if not all(landed):
            await asyncio.gather(*(
                self.drones[i].land()
                for i in range(2)
                if not landed[i]
            ))

        # Final telemetry for summary
        final_telems: list[Telemetry] = list(
            await asyncio.gather(*(d.get_telemetry() for d in self.drones))
        )

        # Tally outcomes from world state
        episode_deliveries = sum(1 for p in self.world.patients if p.actually_delivered)
        episode_deaths = sum(
            1 for p in self.world.patients if p.delivered and not p.actually_delivered
        )
        both_landed = all(t.is_landed for t in final_telems)
        batteries = [t.battery_pct for t in final_telems]
        triage_eff = (
            episode_deliveries / (episode_deliveries + episode_deaths)
            if (episode_deliveries + episode_deaths) > 0
            else 0.0
        )

        summary = {
            "episode": episode,
            "steps": step,
            "patients_delivered": episode_deliveries,
            "patients_died": episode_deaths,
            "both_landed": both_landed,
            "battery_remaining": batteries,
            "total_reward": float(episode_deliveries),
            "triage_efficiency": triage_eff,
        }
        logger.info("Episode %d complete: %s", episode, summary)
        return summary

    async def _dispatch(
        self,
        drone_idx: int,
        action: int,
        telem: Telemetry,
        landed: list[bool],
    ) -> None:
        """Send one action to one drone; skip if already landed."""
        if landed[drone_idx]:
            return
        drone = self.drones[drone_idx]
        if is_land_action(action):
            await drone.land()
        else:
            offset = action_to_offset(action)
            await drone.send_waypoint(
                telem.north_m + offset.d_north,
                telem.east_m + offset.d_east,
                CRUISE_DOWN_M,
            )
