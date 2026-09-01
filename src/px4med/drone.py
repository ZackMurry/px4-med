"""Drone class — wraps a single MAVSDK connection to one PX4 SITL instance."""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mavsdk import System

logger = logging.getLogger(__name__)

# Arrival tolerance for send_waypoint() busy-wait (metres)
WAYPOINT_ARRIVAL_RADIUS_M = 2.0
# How long to wait for drone to reach a waypoint before moving on (seconds)
WAYPOINT_TIMEOUT_S = 2.0
BASE_XY_CRUISE_M_S = 5.0
BASE_XY_VEL_MAX_M_S = 12.0
BASE_Z_VEL_UP_M_S = 3.0
BASE_Z_VEL_DOWN_M_S = 1.5
DEFAULT_SIM_BAT_DRAIN = 5000.0
PX4_SIM_BAT_MIN_PCT = 10.0
TELEMETRY_TIMEOUT_S = 30.0
IN_AIR_TIMEOUT_S = 90.0
LAND_TIMEOUT_S = 120.0


@dataclass
class Telemetry:
    """Snapshot of one drone's state, in NED metres relative to home position."""
    north_m: float
    east_m: float
    down_m: float       # negative = above ground
    battery_pct: float  # 0.0–100.0  (matches training env MAX_BATTERY=100)
    is_landed: bool


class Drone:
    """Manages MAVSDK connection and commands for one PX4 drone."""

    def __init__(
        self,
        drone_id: int,
        mavsdk_address: str,
        grpc_port: int | None = None,
    ) -> None:
        self.drone_id = drone_id
        self.mavsdk_address = mavsdk_address
        self.grpc_port = grpc_port if grpc_port is not None else 50051 + drone_id
        self._system: Optional[System] = None
        self._offboard_active = False
        self._last_battery_pct: float = 100.0
        self._last_is_landed: bool = True
        self._battery_skip_until: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, timeout: float = 180.0) -> None:
        """Connect to PX4 via MAVSDK and wait for a valid local position estimate."""
        from mavsdk import System
        # Use a dedicated gRPC port and MAVLink client sysid per drone so
        # multiple MAVSDK backends do not collide in one process.
        self._system = System(
            port=self.grpc_port,
            sysid=245 + self.drone_id,
        )
        await self._system.connect(system_address=self.mavsdk_address)

        # Wait for heartbeat
        try:
            async with asyncio.timeout(timeout):
                async for state in self._system.core.connection_state():
                    if state.is_connected:
                        logger.info(
                            "Drone %d: MAVSDK connected (%s, grpc=%d)",
                            self.drone_id,
                            self.mavsdk_address,
                            self.grpc_port,
                        )
                        break
        except TimeoutError:
            raise TimeoutError(
                f"Drone {self.drone_id}: MAVSDK connection timed out after "
                f"{timeout:.0f}s ({self.mavsdk_address})"
            )

        # Ask PX4 for steady telemetry stream rates up front, and re-request
        # periodically below: if the backend attaches while PX4 mavlink is
        # still initializing, the one-shot requests get lost.
        await self._request_stream_rates()

        # Gate on readiness. MAVSDK's health.is_local_position_ok can stay
        # frozen at False forever on multi-instance boots even though the
        # PX4-side EKF is fully converged (verified via px4-ekf2 status), so
        # health is only tried briefly; the authoritative fallback check is a
        # live, stable LOCAL_POSITION_NED stream — which is also exactly what
        # the control loop consumes.
        health_budget = min(90.0, timeout / 2)
        healthy = False
        last_log = 0.0
        try:
            async with asyncio.timeout(health_budget):
                async for health in self._system.telemetry.health():
                    if health.is_local_position_ok and health.is_home_position_ok:
                        healthy = True
                        logger.info(
                            "Drone %d: local position + home position ready",
                            self.drone_id,
                        )
                        break
                    now = asyncio.get_running_loop().time()
                    if now - last_log > 30.0:
                        last_log = now
                        await self._request_stream_rates()
                        logger.info(
                            "Drone %d: waiting on health gyro=%s accel=%s mag=%s "
                            "local=%s global=%s home=%s",
                            self.drone_id,
                            health.is_gyrometer_calibration_ok,
                            health.is_accelerometer_calibration_ok,
                            health.is_magnetometer_calibration_ok,
                            health.is_local_position_ok,
                            health.is_global_position_ok,
                            health.is_home_position_ok,
                        )
        except TimeoutError:
            pass

        if not healthy:
            logger.warning(
                "Drone %d: health flags frozen — falling back to direct "
                "position-stream verification",
                self.drone_id,
            )
            await self._verify_position_stream(timeout=timeout - health_budget)

    async def _request_stream_rates(self) -> None:
        """Request telemetry stream rates (best effort, idempotent)."""
        assert self._system is not None
        for setter, rate in (
            ("set_rate_battery", 2.0),
            ("set_rate_landed_state", 2.0),
            ("set_rate_position_velocity_ned", 10.0),
            ("set_rate_home", 1.0),
            ("set_rate_position", 2.0),
            ("set_rate_odometry", 5.0),
            ("set_rate_gps_info", 1.0),
        ):
            try:
                async with asyncio.timeout(5.0):
                    await getattr(self._system.telemetry, setter)(rate)
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(
                    "Drone %d: %s(%s) failed: %s", self.drone_id, setter, rate, exc
                )

    async def _verify_position_stream(self, timeout: float) -> None:
        """Confirm a live, stable NED position stream (EKF output flowing)."""
        assert self._system is not None
        samples: list[tuple[float, float, float]] = []
        try:
            async with asyncio.timeout(timeout):
                async for pv in self._system.telemetry.position_velocity_ned():
                    p = pv.position
                    if not all(
                        math.isfinite(v) for v in (p.north_m, p.east_m, p.down_m)
                    ):
                        samples.clear()
                        continue
                    samples.append((p.north_m, p.east_m, p.down_m))
                    if len(samples) >= 10:
                        spread = max(
                            abs(a - b)
                            for axis in range(3)
                            for a, b in [(samples[0][axis], samples[-1][axis])]
                        )
                        if spread < 2.0:
                            logger.info(
                                "Drone %d: position stream verified "
                                "(%d samples, spread %.2f m)",
                                self.drone_id,
                                len(samples),
                                spread,
                            )
                            return
                        samples.pop(0)
        except TimeoutError:
            raise TimeoutError(
                f"Drone {self.drone_id}: no stable position stream after "
                f"{timeout:.0f}s (health also not ok)"
            )

    async def _next_stream_value(
        self,
        stream,
        *,
        timeout: float,
        label: str,
    ):
        """Return the next item from a MAVSDK async stream with a timeout."""
        try:
            async with asyncio.timeout(timeout):
                async for item in stream:
                    return item
        except TimeoutError as exc:
            raise TimeoutError(
                f"Drone {self.drone_id}: timed out waiting for {label} after {timeout:.0f}s"
            ) from exc
        raise RuntimeError(f"Drone {self.drone_id}: {label} stream closed unexpectedly")

    async def arm(self, retry_window_s: float = 90.0) -> None:
        """Arm, retrying while PX4 finishes preflight checks (COMMAND_DENIED)."""
        assert self._system is not None, "call connect() first"
        from mavsdk.action import ActionError

        deadline = asyncio.get_running_loop().time() + retry_window_s
        attempt = 0
        while True:
            attempt += 1
            try:
                await self._system.action.arm()
                logger.info("Drone %d: armed (attempt %d)", self.drone_id, attempt)
                return
            except ActionError as exc:
                if asyncio.get_running_loop().time() > deadline:
                    raise
                logger.warning(
                    "Drone %d: arm attempt %d denied (%s), retrying ...",
                    self.drone_id,
                    attempt,
                    exc._result.result_str if hasattr(exc, "_result") else exc,
                )
                await asyncio.sleep(2.0)

    async def configure_speed_profile(self, speed_factor: float = 1.0) -> None:
        """Scale PX4 horizontal/vertical speed limits for faster SITL runs."""
        assert self._system is not None, "call connect() first"
        if speed_factor <= 0.0:
            raise ValueError("speed_factor must be positive")

        async def _set_float(name: str, value: float) -> None:
            await self._system.param.set_param_float(name, value)
            logger.info("Drone %d: param %s=%.2f", self.drone_id, name, value)

        xy_cruise = BASE_XY_CRUISE_M_S * speed_factor
        xy_max = BASE_XY_VEL_MAX_M_S * speed_factor
        z_up = BASE_Z_VEL_UP_M_S * speed_factor
        z_down = BASE_Z_VEL_DOWN_M_S * speed_factor

        await _set_float("MPC_XY_CRUISE", xy_cruise)
        await _set_float("MPC_XY_VEL_MAX", xy_max)
        await _set_float("MPC_Z_VEL_MAX_UP", z_up)
        await _set_float("MPC_Z_VEL_MAX_DN", z_down)
        logger.info(
            "Drone %d: speed profile factor=%.2f xy_cruise=%.2f xy_max=%.2f z_up=%.2f z_down=%.2f",
            self.drone_id,
            speed_factor,
            xy_cruise,
            xy_max,
            z_up,
            z_down,
        )

    async def configure_battery_profile(self, drain_rate: float = DEFAULT_SIM_BAT_DRAIN) -> None:
        """Configure PX4 SITL battery parameters for longer experimental runs."""
        assert self._system is not None, "call connect() first"
        if drain_rate < 0.0:
            raise ValueError("drain_rate must be non-negative")

        async def _set_int(name: str, value: int) -> None:
            await self._system.param.set_param_int(name, value)
            logger.info("Drone %d: param %s=%d", self.drone_id, name, value)

        async def _set_float(name: str, value: float) -> None:
            await self._system.param.set_param_float(name, value)
            logger.info("Drone %d: param %s=%.2f", self.drone_id, name, value)

        async def _set_float_verified(name: str, value: float, attempts: int = 3) -> None:
            for attempt in range(1, attempts + 1):
                await _set_float(name, value)
                actual = await self._system.param.get_param_float(name)
                if abs(actual - value) <= 1e-6:
                    logger.info(
                        "Drone %d: verified param %s=%.2f",
                        self.drone_id,
                        name,
                        actual,
                    )
                    return
                logger.warning(
                    "Drone %d: param verify mismatch for %s expected=%.2f actual=%.2f attempt=%d/%d",
                    self.drone_id,
                    name,
                    value,
                    actual,
                    attempt,
                    attempts,
                )
                await asyncio.sleep(0.2)
            raise RuntimeError(
                f"Drone {self.drone_id}: failed to verify PX4 param {name}={value}"
            )

        sim_bat_min_pct = 100.0 if drain_rate <= 0.0 else PX4_SIM_BAT_MIN_PCT
        # Disable onboard ULog flight logging — .ulg files (~15 MB per flight
        # per drone) were filling the mounted log volume across long runs.
        await _set_int("SDLOG_MODE", -1)
        await _set_int("COM_LOW_BAT_ACT", 0)
        await _set_float("COM_ARM_BAT_MIN", 0.0)
        await _set_int("COM_ARM_WO_GPS", 2)
        await _set_int("CBRK_SUPPLY_CHK", 894281)
        await _set_float("SIM_BAT_MIN_PCT", sim_bat_min_pct)
        await _set_float_verified("SIM_BAT_DRAIN", drain_rate)
        logger.info(
            "Drone %d: battery profile drain_rate=%.2f sim_bat_min_pct=%.2f",
            self.drone_id,
            drain_rate,
            sim_bat_min_pct,
        )

    async def takeoff(self, altitude_m: float = 20.0) -> None:
        """Command auto-takeoff and wait until the drone is airborne."""
        assert self._system is not None
        await self._system.action.set_takeoff_altitude(altitude_m)
        await self._system.action.takeoff()
        while True:
            in_air = await self._next_stream_value(
                self._system.telemetry.in_air(),
                timeout=IN_AIR_TIMEOUT_S,
                label="in-air state",
            )
            if in_air:
                logger.info("Drone %d: airborne at %.1f m AGL", self.drone_id, altitude_m)
                return

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    async def send_waypoint(
        self,
        north_m: float,
        east_m: float,
        down_m: float,
        *,
        arrival_radius_m: float = WAYPOINT_ARRIVAL_RADIUS_M,
        timeout_s: float = WAYPOINT_TIMEOUT_S,
    ) -> None:
        """Send an absolute NED position setpoint via offboard mode.

        On the first call, starts offboard mode from the drone's current position
        to avoid a sudden jump. Waits up to WAYPOINT_TIMEOUT_S for arrival within
        WAYPOINT_ARRIVAL_RADIUS_M, then returns regardless (matches 2 Hz step loop).
        """
        from mavsdk.offboard import OffboardError, PositionNedYaw
        assert self._system is not None
        target = PositionNedYaw(north_m, east_m, down_m, 0.0)

        if not self._offboard_active:
            # Seed offboard with current position before enabling the mode
            telem = await self.get_telemetry()
            hold = PositionNedYaw(telem.north_m, telem.east_m, telem.down_m, 0.0)
            await self._system.offboard.set_position_ned(hold)
            try:
                await self._system.offboard.start()
            except OffboardError as e:
                raise RuntimeError(
                    f"Drone {self.drone_id}: failed to start offboard mode: {e}"
                ) from e
            self._offboard_active = True
            logger.info("Drone %d: offboard mode started", self.drone_id)

        await self._system.offboard.set_position_ned(target)

        # Busy-wait for arrival or step timeout
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            pos_vel = await self._next_stream_value(
                self._system.telemetry.position_velocity_ned(),
                timeout=min(TELEMETRY_TIMEOUT_S, timeout_s),
                label="position telemetry",
            )
            p = pos_vel.position
            dist = math.sqrt(
                (p.north_m - north_m) ** 2
                + (p.east_m - east_m) ** 2
                + (p.down_m - down_m) ** 2
            )
            if dist < arrival_radius_m or loop.time() > deadline:
                break

    async def land(self) -> None:
        """Stop offboard mode and command landing; wait until on the ground."""
        assert self._system is not None
        if self._offboard_active:
            try:
                await self._system.offboard.stop()
            except Exception:
                pass
            self._offboard_active = False

        await self._system.action.land()
        logger.info("Drone %d: landing commanded", self.drone_id)

        from mavsdk.telemetry import LandedState
        deadline = asyncio.get_running_loop().time() + LAND_TIMEOUT_S
        while True:
            state = await self._next_stream_value(
                self._system.telemetry.landed_state(),
                timeout=TELEMETRY_TIMEOUT_S,
                label="landed state",
            )
            if state == LandedState.ON_GROUND:
                logger.info("Drone %d: on the ground", self.drone_id)
                return
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(
                    f"Drone {self.drone_id}: landing did not complete after {LAND_TIMEOUT_S:.0f}s"
                )

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    async def get_telemetry(self) -> Telemetry:
        """Return a fresh snapshot of position, battery, and landed state."""
        assert self._system is not None

        # Read next value from each telemetry stream independently
        pos_vel = await self._next_stream_value(
            self._system.telemetry.position_velocity_ned(),
            timeout=TELEMETRY_TIMEOUT_S,
            label="position telemetry",
        )
        pos = pos_vel.position

        # Battery is diagnostic (the mission-energy ledger lives in the world
        # model). A stalled stream falls back to the last known value, and a
        # stall triggers a 30s read backoff — paying a 5s timeout on every
        # step drags the control loop from ~2s to ~6s per step (this timed
        # out a full episode on 2026-08-29).
        now = asyncio.get_running_loop().time()
        if now >= self._battery_skip_until:
            try:
                bat = await self._next_stream_value(
                    self._system.telemetry.battery(),
                    timeout=2.0,
                    label="battery telemetry",
                )
                # MAVSDK v3.x returns remaining_percent as 0–100 (not 0–1)
                self._last_battery_pct = bat.remaining_percent
            except TimeoutError:
                self._battery_skip_until = now + 30.0
                logger.warning(
                    "Drone %d: battery telemetry stalled, backing off 30s "
                    "(last known %.1f%%)",
                    self.drone_id,
                    self._last_battery_pct,
                )
        battery_pct = self._last_battery_pct

        from mavsdk.telemetry import LandedState
        try:
            state = await self._next_stream_value(
                self._system.telemetry.landed_state(),
                timeout=2.0,
                label="landed state",
            )
            self._last_is_landed = state == LandedState.ON_GROUND
        except TimeoutError:
            self._last_is_landed = self._last_is_landed
        is_landed = self._last_is_landed

        return Telemetry(
            north_m=pos.north_m,
            east_m=pos.east_m,
            down_m=pos.down_m,
            battery_pct=battery_pct,
            is_landed=is_landed,
        )
