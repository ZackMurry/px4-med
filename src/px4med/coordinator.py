"""Coordinator — drives the CEDA-FGCS policy using training-env state over PX4 SITL."""
from __future__ import annotations

import asyncio
from collections import deque
import logging
import time
from typing import TYPE_CHECKING

from .actions import ACTION_HOVER, CRUISE_DOWN_M
from .drone import Telemetry
from .fgcs_state import build_observation
from .metrics import StepRecord
from .true_world import accumulate_hazard, new_hazard_totals

if TYPE_CHECKING:
    from .drone import Drone
    from .environment import WorldEnvironment
    from .fgcs_policy import FGCSPolicy
    from .metrics import MetricsCollector

logger = logging.getLogger(__name__)

_START_REPOSITION_TIMEOUT_S = 45.0
_START_SETTLE_TIMEOUT_S = 30.0
_START_SETTLE_RADIUS_CELLS = 1
# A drone this far from its assigned start after the settle timeout is not
# braking late — it never flew. Healthy drones settle on the exact cell, and
# the observed fault sat 109 cells away, so anything in between is comfortably
# separated. See HANDOFF.md §5c.
_INERT_START_MAX_CELLS = 12
_MAX_TELEMETRY_STEP_JUMP_CELLS = 8
_MAX_TRACKING_ERROR_M = 20.0
_MAX_ALTITUDE_M = 50.0


class InertDroneError(RuntimeError):
    """A drone reported airborne but never actually moved (HANDOFF.md §5c)."""


def find_stranded_drones(
    actual_grids: list[tuple[int, int]],
    start_grids: list[tuple[int, int]],
    max_cells: int = _INERT_START_MAX_CELLS,
) -> list[tuple[int, int]]:
    """(index, manhattan_distance) for drones implausibly far from their start."""
    stranded = []
    for i, (actual, target) in enumerate(zip(actual_grids, start_grids)):
        distance = abs(actual[0] - target[0]) + abs(actual[1] - target[1])
        if distance > max_cells:
            stranded.append((i, distance))
    return stranded


class Coordinator:
    """Drives the drone fleet through the RL policy for one episode."""

    def __init__(
        self,
        drones: list[Drone],
        policy: "FGCSPolicy",
        world: WorldEnvironment,
        metrics: MetricsCollector,
        step_hz: float = 2.0,
        action_delay_steps: int = 0,
    ) -> None:
        self.drones = drones
        self.policy = policy
        self.world = world
        self.metrics = metrics
        self.step_interval = 1.0 / step_hz
        self.action_delay_steps = max(0, action_delay_steps)
        self.n = len(drones)

    def _write_phase_status(self, note: str) -> None:
        write_status = getattr(self.metrics, "write_status", None)
        if callable(write_status):
            write_status(status="running", note=note)

    async def run_episode(self, episode: int = 0, max_steps: int = 800) -> dict:
        """Arm, take off, run RL loop, land all. Return summary dict."""
        self.world.max_steps = max_steps
        self.world.reset()
        n = self.n

        self._write_phase_status("coordinator: arming drones")
        await asyncio.gather(*(d.arm() for d in self.drones))
        self._write_phase_status("coordinator: takeoff")
        await asyncio.gather(*(d.takeoff() for d in self.drones))

        # Reposition to training-env start positions before the RL loop.
        mpc = float(self.world.config.get("grid", {}).get("meters_per_cell", 2.0))
        start_grids = self.world.start_grids
        logger.info("Repositioning drones to training start positions: %s", start_grids)
        self._write_phase_status("coordinator: repositioning to start grids")
        await asyncio.gather(*(
            self.drones[i].send_waypoint(
                -start_grids[i][1] * mpc,   # north = -grid_y * mpc
                start_grids[i][0] * mpc,    # east  =  grid_x * mpc
                CRUISE_DOWN_M,
                timeout_s=_START_REPOSITION_TIMEOUT_S,
            )
            for i in range(n)
        ))
        self._write_phase_status("coordinator: waiting for start-grid settle")
        await self._wait_for_start_positions(start_grids)

        step = 0
        loop = asyncio.get_running_loop()
        total_reward = 0.0
        episode_wind_entries = [0] * n
        episode_low_signal_entries = [0] * n
        episode_obstacle_collisions = 0
        episode_agent_collisions = 0
        episode_wrong_land_attempts = 0
        hazard_totals = new_hazard_totals()
        action_queues = [
            deque([ACTION_HOVER] * self.action_delay_steps)
            for _ in range(n)
        ]

        def fleet_settled(i: int) -> bool:
            return self.world.landed[i] or self.world.depleted[i]

        while step < max_steps and not all(fleet_settled(i) for i in range(n)):
            step_start = loop.time()

            # 1. Gather telemetry
            if step == 0:
                self._write_phase_status("coordinator: first telemetry sample")
            telems: list[Telemetry] = list(
                await asyncio.gather(*(d.get_telemetry() for d in self.drones))
            )
            self._validate_altitudes(telems)

            # 2. Sync quantised grid positions from telemetry before building state.
            #    Landed/depleted drones keep their world position (telemetry from a
            #    landed drone stays at the pad; a depleted drone is frozen in world).
            actual_grids = list(self.world.agent_grids)
            g_max = self.world.grid_size - 1
            for i in range(n):
                if not fleet_settled(i):
                    gx, gy = self.world.get_grid_pos(
                        telems[i].north_m, telems[i].east_m
                    )
                    # Clamp to the world: braking overshoot at map edges can
                    # round telemetry to out-of-bounds cells; dispatching
                    # those back as waypoints creates a runaway feedback
                    # loop (drone chases its own drift off the map).
                    actual_grids[i] = (
                        min(max(gx, 0), g_max),
                        min(max(gy, 0), g_max),
                    )
            jump = self._telemetry_jump(actual_grids)
            if jump is not None:
                # Re-sample once: transient EKF glitches recover; real
                # divergence will trip again and abort the episode.
                logger.warning("Telemetry jump detected (%s); re-sampling once", jump)
                await asyncio.sleep(1.0)
                telems = list(
                    await asyncio.gather(*(d.get_telemetry() for d in self.drones))
                )
                for i in range(n):
                    if not fleet_settled(i):
                        actual_grids[i] = self.world.get_grid_pos(
                            telems[i].north_m, telems[i].east_m
                        )
            self._validate_telemetry_positions(actual_grids)
            self.world.agent_grids = actual_grids

            # 3. Build fleet observation and run policy inference.
            # TrueWorld supplies the training env's own get_state(); the
            # legacy world falls back to the reconstructed builder.
            if hasattr(self.world, "observation"):
                observation = self.world.observation()
            else:
                observation = build_observation(self.world)
            raw_actions = self.policy.select_actions(observation)

            actions: list[int] = []
            for i, action in enumerate(raw_actions):
                if self.action_delay_steps > 0:
                    action_queues[i].append(action)
                    actions.append(action_queues[i].popleft())
                else:
                    actions.append(action)

            # 4. Advance world state using the training-env transition logic.
            step_data = self.world.step(actions)
            total_reward += sum(step_data["rewards"])
            for i in range(n):
                episode_wind_entries[i] += step_data["wind_entries"][i]
                episode_low_signal_entries[i] += step_data["low_signal_entries"][i]
            episode_obstacle_collisions += step_data["obstacle_collisions"]
            episode_agent_collisions += step_data["agent_collisions"]
            accumulate_hazard(hazard_totals, step_data.get("raw_step_data"))
            if "wrong_land_count" in step_data:
                episode_wrong_land_attempts += step_data["wrong_land_count"]
            else:
                episode_wrong_land_attempts += sum(
                    1
                    for i in range(n)
                    if step_data["landing_attempts"][i]
                    and not step_data["landed_this_step"][i]
                )

            step_deliveries = list(step_data["deliveries"])
            logger.info(
                "Episode %d step %d/%d | actions=%s | sim_pos=%s | pending=%d | "
                "deliveries=%s | landed=%s | depleted=%s | phase=%s | reward=%.2f",
                episode,
                step + 1,
                max_steps,
                actions,
                step_data["sim_positions"],
                self.world.pending_count(),
                step_deliveries,
                self.world.landed,
                self.world.depleted,
                "landing" if self.world.landing_phase() else "rescue",
                float(sum(step_data["rewards"])),
            )

            # 5. Log step
            self.metrics.log_step(StepRecord(
                episode=episode,
                step=step,
                timestamp=time.time(),
                drone_north=[t.north_m for t in telems],
                drone_east=[t.east_m for t in telems],
                drone_battery=[t.battery_pct for t in telems],
                actions=actions,
                deliveries=step_deliveries,
                rewards=step_data["rewards"],
                remaining_patients=self.world.pending_count(),
                target_distances=self._target_distances(),
                simulated_positions=[list(pos) for pos in step_data["sim_positions"]],
                wind_entries=step_data["wind_entries"],
                low_signal_entries=step_data["low_signal_entries"],
                obstacle_collisions=step_data["obstacle_collisions"],
                agent_collisions=step_data["agent_collisions"],
                landing_attempts=step_data["landing_attempts"],
                landed_this_step=step_data["landed_this_step"],
            ))

            # 6. Dispatch the world-approved transition targets.
            await asyncio.gather(*(self._dispatch(i) for i in range(n)))

            step += 1

            if step_data["done"]:
                break

            # 7. Pace at step_hz (sleep any remaining budget in this interval)
            elapsed = loop.time() - step_start
            remaining = self.step_interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)

        # Land any drones still airborne (episode timeout / depleted cleanup)
        still_airborne = [i for i in range(n) if not self.world.landed[i]]
        if still_airborne:
            await asyncio.gather(*(
                self.drones[i].land() for i in still_airborne
            ), return_exceptions=True)

        # Final telemetry for summary
        final_telems: list[Telemetry] = list(
            await asyncio.gather(*(d.get_telemetry() for d in self.drones))
        )

        # Tally outcomes from world state
        episode_deliveries = sum(1 for p in self.world.patients if p.delivered)
        episode_deaths = sum(1 for p in self.world.patients if p.died)
        episode_spawned = sum(1 for p in self.world.patients if p.active)
        all_landed = all(self.world.landed)
        batteries = [t.battery_pct for t in final_telems]
        triage = self.world.triage_summary()

        summary = {
            "episode": episode,
            "steps": step,
            "patients_delivered": episode_deliveries,
            "patients_died": episode_deaths,
            "patients_spawned": episode_spawned,
            "all_landed": all_landed,
            "drones_landed": sum(self.world.landed),
            "drones_depleted": sum(self.world.depleted),
            "battery_remaining": batteries,
            "simulated_battery_remaining": list(self.world.batteries),
            "total_reward": total_reward,
            "triage_efficiency": float(triage["triage_efficiency"]),
            "wind_entries": episode_wind_entries,
            "low_signal_entries": episode_low_signal_entries,
            "obstacle_collisions": episode_obstacle_collisions,
            "agent_collisions": episode_agent_collisions,
            "wrong_land_attempts": episode_wrong_land_attempts,
            "hazard_totals": hazard_totals,
        }
        logger.info("Episode %d complete: %s", episode, summary)
        return summary

    def _telemetry_jump(
        self,
        actual_grids: list[tuple[int, int]],
    ) -> str | None:
        """Return a description of the first oversized telemetry jump, if any."""
        for i, actual in enumerate(actual_grids):
            expected = tuple(self.world.agent_grids[i])
            jump_cells = abs(actual[0] - expected[0]) + abs(actual[1] - expected[1])
            if jump_cells > _MAX_TELEMETRY_STEP_JUMP_CELLS:
                return f"drone={i} expected={expected} actual={actual} jump={jump_cells}"
        return None

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
            if self.world.landed[i] or self.world.depleted[i]:
                continue
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
        """Hold episode start until all drones are near their start grids."""
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
                stranded = find_stranded_drones(actual_grids, start_grids)
                if stranded:
                    # An intermittent SITL fault leaves one drone's telemetry
                    # pinned at the map origin while it reports armed and
                    # airborne (HANDOFF.md §5c). Proceeding produces a silently
                    # 4-drone episode whose mission_success and all_landed are
                    # forced to 0 — corrupt data that looks plausible. Fail the
                    # attempt instead and let the runner retry on a fresh
                    # container. Healthy drones land on their start cell
                    # exactly, so the generous threshold makes this specific to
                    # the pathological case, not to ordinary braking overshoot.
                    detail = "; ".join(
                        f"drone={i} expected={start_grids[i]} "
                        f"actual={actual_grids[i]} off_by={distance}_cells"
                        for i, distance in stranded
                    )
                    raise InertDroneError(
                        "Drone(s) never reached their start position — "
                        f"suspected inert-telemetry fault: {detail}"
                    )
                logger.warning(
                    "Timed out waiting for training start positions. expected=%s actual=%s",
                    start_grids,
                    actual_grids,
                )
                self.world.agent_grids = actual_grids
                return
            await asyncio.sleep(0.25)

    async def _dispatch(self, drone_idx: int) -> None:
        """Send one world-approved target to one drone.

        A depleted (virtual battery = 0) drone is physically landed once and
        then left alone; the world keeps it as a dead entity. Hover holds the
        current cell setpoint; land is only issued when the world accepted it.
        """
        drone = self.drones[drone_idx]
        if self.world.landed[drone_idx]:
            if drone._offboard_active:
                await drone.land()
            return
        if self.world.depleted[drone_idx]:
            if drone._offboard_active:
                await drone.land()
            return
        grid_x, grid_y = self.world.agent_grids[drone_idx]
        north_m, east_m = self.world.grid_to_ned(grid_x, grid_y)
        await drone.send_waypoint(north_m, east_m, CRUISE_DOWN_M)

    def _target_grid(self, agent_idx: int) -> tuple[int, int]:
        nearest = self.world.nearest_undelivered_patient(self.world.agent_grids[agent_idx])
        if nearest is not None:
            return self.world.patient_grid(nearest)
        return self.world.landing_grid(agent_idx)

    def _target_distances(self) -> list[int]:
        distances: list[int] = []
        for agent_idx in range(self.n):
            if self.world.landed[agent_idx] or self.world.depleted[agent_idx]:
                distances.append(0)
                continue
            target = self._target_grid(agent_idx)
            distances.append(
                self.world.manhattan_distance(self.world.agent_grids[agent_idx], target)
            )
        return distances
