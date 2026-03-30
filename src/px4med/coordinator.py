"""Coordinator — drives the policy using training-env state over PX4 SITL."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from .actions import CRUISE_DOWN_M
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
        total_reward = 0.0
        episode_wind_entries = [0, 0]
        episode_low_signal_entries = [0, 0]
        episode_obstacle_collisions = 0
        episode_agent_collisions = 0

        while step < max_steps and not all(landed):
            step_start = loop.time()

            # 1. Gather telemetry
            telems: list[Telemetry] = list(
                await asyncio.gather(*(d.get_telemetry() for d in self.drones))
            )

            # 2. Sync quantised grid positions from telemetry before building state.
            self.world.agent_grids = [
                self.world.get_grid_pos(telem.north_m, telem.east_m)
                for telem in telems
            ]

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

            # 5. Advance world state using the training-env transition logic.
            step_data = self.world.step(actions)
            total_reward += sum(step_data["rewards"])
            for i in range(2):
                episode_wind_entries[i] += step_data["wind_entries"][i]
                episode_low_signal_entries[i] += step_data["low_signal_entries"][i]
            episode_obstacle_collisions += step_data["obstacle_collisions"]
            episode_agent_collisions += step_data["agent_collisions"]

            step_deliveries = list(step_data["deliveries"])
            logger.debug("Episode %d step %d: rewards=%s", episode, step, step_data["rewards"])

            # 6. Log step
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
                rewards=step_data["rewards"],
                simulated_positions=[list(pos) for pos in step_data["sim_positions"]],
                wind_entries=step_data["wind_entries"],
                low_signal_entries=step_data["low_signal_entries"],
                obstacle_collisions=step_data["obstacle_collisions"],
                agent_collisions=step_data["agent_collisions"],
                landing_attempts=step_data["landing_attempts"],
                landed_this_step=step_data["landed_this_step"],
            ))

            # 7. Dispatch the world-approved transition targets.
            await asyncio.gather(*(
                self._dispatch(i, landed)
                for i in range(2)
            ))
            landed = list(self.world.landed)

            step += 1

            if step_data["done"]:
                break

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
        episode_spawned = sum(1 for p in self.world.patients if p.active)
        both_landed = all(t.is_landed for t in final_telems)
        batteries = [t.battery_pct for t in final_telems]
        triage = self.world.triage_summary()

        summary = {
            "episode": episode,
            "steps": step,
            "patients_delivered": episode_deliveries,
            "patients_died": episode_deaths,
            "patients_spawned": episode_spawned,
            "both_landed": both_landed,
            "battery_remaining": batteries,
            "simulated_battery_remaining": list(self.world.batteries),
            "total_reward": total_reward,
            "triage_efficiency": float(triage["triage_efficiency"]),
            "wind_entries": episode_wind_entries,
            "low_signal_entries": episode_low_signal_entries,
            "obstacle_collisions": episode_obstacle_collisions,
            "agent_collisions": episode_agent_collisions,
        }
        logger.info("Episode %d complete: %s", episode, summary)
        return summary

    async def _dispatch(
        self,
        drone_idx: int,
        landed: list[bool],
    ) -> None:
        """Send one world-approved target to one drone; skip if already landed."""
        if landed[drone_idx]:
            return
        drone = self.drones[drone_idx]
        if self.world.landed[drone_idx]:
            await drone.land()
        else:
            grid_x, grid_y = self.world.agent_grids[drone_idx]
            north_m, east_m = self.world.grid_to_ned(grid_x, grid_y)
            await drone.send_waypoint(
                north_m,
                east_m,
                CRUISE_DOWN_M,
            )
