"""Coordinator — drives the policy using training-env state over PX4 SITL."""
from __future__ import annotations

import asyncio
from collections import deque
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
_MOVE_ACTIONS = [0, 1, 2, 3]
_ACTION_DELTAS = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
}
_START_REPOSITION_TIMEOUT_S = 12.0
_START_SETTLE_TIMEOUT_S = 10.0
_START_SETTLE_RADIUS_CELLS = 1
_MAX_TELEMETRY_STEP_JUMP_CELLS = 8
_MAX_TRACKING_ERROR_M = 20.0
_MAX_ALTITUDE_M = 50.0


class Coordinator:
    """Drives both drones through the RL policy for one episode."""

    def __init__(
        self,
        drones: list[Drone],
        policy: PolicyNet,
        world: WorldEnvironment,
        metrics: MetricsCollector,
        step_hz: float = 2.0,
        action_delay_steps: int = 0,
        enable_cycle_breaking: bool = False,
    ) -> None:
        self.drones = drones
        self.policy = policy
        self.world = world
        self.metrics = metrics
        self.step_interval = 1.0 / step_hz
        self.action_delay_steps = max(0, action_delay_steps)
        self.enable_cycle_breaking = enable_cycle_breaking
        self._position_history = [deque(maxlen=8), deque(maxlen=8)]

    def _write_phase_status(self, note: str) -> None:
        write_status = getattr(self.metrics, "write_status", None)
        if callable(write_status):
            write_status(status="running", note=note)

    async def run_episode(self, episode: int = 0, max_steps: int = 800) -> dict:
        """Arm, take off, run RL loop, land all. Return summary dict."""
        self.world.reset()
        self._position_history = [deque(maxlen=8), deque(maxlen=8)]

        # Arm and take off both drones concurrently
        self._write_phase_status("coordinator: arming drones")
        await asyncio.gather(*(d.arm() for d in self.drones))
        self._write_phase_status("coordinator: takeoff")
        await asyncio.gather(*(d.takeoff() for d in self.drones))

        # Reposition to training-env start positions before the RL loop.
        # The training env initialises agents at grid (1,1) and (1,13);
        # SITL spawns both at NED (0,0), which the policy has never seen.
        mpc = float(self.world.config.get("grid", {}).get("meters_per_cell", 2.0))
        start_grids = self.world.config.get("agent_start_positions", _DEFAULT_START_GRIDS)
        logger.info("Repositioning drones to training start positions: %s", start_grids)
        self._write_phase_status("coordinator: repositioning to start grids")
        await asyncio.gather(*(
            self.drones[i].send_waypoint(
                -start_grids[i][1] * mpc,   # north = -grid_y * mpc
                start_grids[i][0] * mpc,    # east  =  grid_x * mpc
                CRUISE_DOWN_M,
                timeout_s=_START_REPOSITION_TIMEOUT_S,
            )
            for i in range(len(self.drones))
        ))
        self._write_phase_status("coordinator: waiting for start-grid settle")
        await self._wait_for_start_positions(start_grids)

        landed = [False, False]
        step = 0
        loop = asyncio.get_running_loop()
        total_reward = 0.0
        episode_wind_entries = [0, 0]
        episode_low_signal_entries = [0, 0]
        episode_obstacle_collisions = 0
        episode_agent_collisions = 0
        action_queues = [
            deque([-1] * self.action_delay_steps)
            for _ in range(2)
        ]

        while step < max_steps and not all(landed):
            step_start = loop.time()

            # 1. Gather telemetry
            if step == 0:
                self._write_phase_status("coordinator: first telemetry sample")
            telems: list[Telemetry] = list(
                await asyncio.gather(*(d.get_telemetry() for d in self.drones))
            )
            self._validate_altitudes(telems)

            # 2. Sync quantised grid positions from telemetry before building state.
            actual_grids = [
                self.world.get_grid_pos(telem.north_m, telem.east_m)
                for telem in telems
            ]
            self._validate_telemetry_positions(actual_grids)
            self.world.agent_grids = actual_grids
            self._record_positions()

            # 3. Build per-agent state vectors
            states = [
                build_state(i, telems[i], telems[1 - i], self.world)
                for i in range(2)
            ]

            # 4. Policy inference (joint state: self ‖ other)
            if hasattr(self.policy, "select_actions"):
                raw_actions = list(self.policy.select_actions(self.world))
            else:
                raw_actions = [
                    self.policy.select_action(states[i], states[1 - i])
                    for i in range(2)
                ]

            if self.enable_cycle_breaking:
                raw_actions = self._break_loops(raw_actions, step)

            actions: list[int] = []
            for i, action in enumerate(raw_actions):
                if self.action_delay_steps > 0:
                    action_queues[i].append(action)
                    actions.append(action_queues[i].popleft())
                else:
                    actions.append(action)

            logger.debug(
                "Episode %d step %d: raw_actions=%s executed_actions=%s",
                episode,
                step,
                raw_actions,
                actions,
            )

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
            logger.info(
                "Episode %d step %d/%d | actions=%s | sim_pos=%s | remaining=%d | "
                "target_dist=%s | deliveries=%s | landed=%s | reward=%.2f",
                episode,
                step + 1,
                max_steps,
                actions,
                step_data["sim_positions"],
                self._remaining_patients(),
                self._target_distances(),
                step_deliveries,
                self.world.landed,
                float(sum(step_data["rewards"])),
            )

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
                remaining_patients=self._remaining_patients(),
                target_distances=self._target_distances(),
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

    def _validate_telemetry_positions(
        self,
        actual_grids: list[tuple[int, int]],
    ) -> None:
        """Fail fast on telemetry jumps large enough to invalidate world-state sync."""
        for i, actual in enumerate(actual_grids):
            expected = tuple(self.world.agent_grids[i])
            jump_cells = abs(actual[0] - expected[0]) + abs(actual[1] - expected[1])
            tracking_error_m = (
                jump_cells
                * float(self.world.config.get("grid", {}).get("meters_per_cell", 2.0))
            )
            if jump_cells > _MAX_TELEMETRY_STEP_JUMP_CELLS or tracking_error_m > _MAX_TRACKING_ERROR_M:
                raise RuntimeError(
                    "Telemetry jump too large for safe SITL/world sync: "
                    f"drone={i} expected_grid={expected} actual_grid={actual} "
                    f"jump_cells={jump_cells} tracking_error_m={tracking_error_m:.1f}"
                )

    def _validate_altitudes(self, telems: list[Telemetry]) -> None:
        """Fail fast if a drone climbs above the allowed AGL ceiling."""
        for i, telem in enumerate(telems):
            altitude_m = max(0.0, -float(telem.down_m))
            if altitude_m > _MAX_ALTITUDE_M:
                raise RuntimeError(
                    f"Drone exceeded altitude ceiling: drone={i} "
                    f"altitude_m={altitude_m:.1f} limit_m={_MAX_ALTITUDE_M:.1f}"
                )

    async def _wait_for_start_positions(
        self,
        start_grids: list[tuple[int, int]],
    ) -> None:
        """Hold episode start until both drones are near the expected training grids."""
        deadline = asyncio.get_running_loop().time() + _START_SETTLE_TIMEOUT_S
        while True:
            telems: list[Telemetry] = list(
                await asyncio.gather(*(d.get_telemetry() for d in self.drones))
            )
            actual_grids = [
                self.world.get_grid_pos(telem.north_m, telem.east_m)
                for telem in telems
            ]
            if all(
                abs(actual[0] - target[0]) + abs(actual[1] - target[1]) <= _START_SETTLE_RADIUS_CELLS
                for actual, target in zip(actual_grids, start_grids)
            ):
                self.world.agent_grids = actual_grids
                logger.info("Drones settled near start grids: %s", actual_grids)
                return
            if asyncio.get_running_loop().time() > deadline:
                logger.warning(
                    "Timed out waiting for training start positions. expected=%s actual=%s",
                    start_grids,
                    actual_grids,
                )
                self.world.agent_grids = actual_grids
                return
            await asyncio.sleep(0.25)

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

    def _record_positions(self) -> None:
        for i in range(2):
            self._position_history[i].append(tuple(self.world.agent_grids[i]))

    def _break_loops(self, actions: list[int], step: int) -> list[int]:
        adjusted = list(actions)
        for agent_idx, action in enumerate(actions):
            if self.world.landed[agent_idx]:
                continue
            if not self._is_square_loop(agent_idx):
                continue
            if action == 4 and self.world.agent_grids[agent_idx] == self.world.landing_grid(agent_idx):
                continue
            override = self._choose_escape_action(agent_idx, action)
            if override != action:
                logger.warning(
                    "Episode step %d: drone %d loop detected at %s, overriding action %s -> %s",
                    step,
                    agent_idx,
                    list(self._position_history[agent_idx]),
                    action,
                    override,
                )
                adjusted[agent_idx] = override
        return adjusted

    def _is_square_loop(self, agent_idx: int) -> bool:
        history = list(self._position_history[agent_idx])
        if len(history) < 5:
            return self._is_two_point_loop(agent_idx)
        recent = history[-5:]
        return (
            recent[0] == recent[-1] and len(set(recent[:-1])) == 4
        ) or self._is_two_point_loop(agent_idx)

    def _is_two_point_loop(self, agent_idx: int) -> bool:
        history = list(self._position_history[agent_idx])
        if len(history) < 4:
            return False
        recent = history[-4:]
        return recent[0] == recent[2] and recent[1] == recent[3] and recent[0] != recent[1]

    def _choose_escape_action(self, agent_idx: int, current_action: int) -> int:
        pos = tuple(self.world.agent_grids[agent_idx])
        recent_cells = set(self._position_history[agent_idx])
        target = self._target_grid(agent_idx)
        grid_size = int(self.world.config.get("grid", {}).get("size", 50))
        best_action = current_action
        best_score = float("-inf")

        for action in _MOVE_ACTIONS:
            if action == current_action:
                continue
            dx, dy = _ACTION_DELTAS[action]
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt[0] < 0 or nxt[0] >= grid_size or nxt[1] < 0 or nxt[1] >= grid_size:
                continue
            if nxt in self.world.obstacles:
                continue

            score = 0.0
            if nxt not in recent_cells:
                score += 5.0
            score -= float(self.world.manhattan_distance(nxt, target))
            score += float(self.world.manhattan_distance(pos, target)) * 0.5
            if nxt == pos:
                score -= 10.0
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _target_grid(self, agent_idx: int) -> tuple[int, int]:
        nearest = self.world.nearest_undelivered_patient(tuple(self.world.agent_grids[agent_idx]))
        if nearest is not None:
            return self.world.patient_grid(nearest)
        return self.world.landing_grid(agent_idx)

    def _remaining_patients(self) -> int:
        return sum(1 for patient in self.world.patients if patient.active and not patient.delivered)

    def _target_distances(self) -> list[int]:
        distances: list[int] = []
        for agent_idx in range(2):
            if self.world.landed[agent_idx]:
                distances.append(0)
                continue
            target = self._target_grid(agent_idx)
            distances.append(
                self.world.manhattan_distance(tuple(self.world.agent_grids[agent_idx]), target)
            )
        return distances
