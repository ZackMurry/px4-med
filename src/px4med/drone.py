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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, timeout: float = 30.0) -> None:
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

        # Wait for local position (EKF converged) — required before offboard
        try:
            async with asyncio.timeout(timeout):
                async for health in self._system.telemetry.health():
                    if health.is_local_position_ok:
                        logger.info(
                            "Drone %d: local position estimate ready", self.drone_id
                        )
                        break
        except TimeoutError:
            raise TimeoutError(
                f"Drone {self.drone_id}: no local position estimate after {timeout:.0f}s"
            )

    async def arm(self) -> None:
        assert self._system is not None, "call connect() first"
        await self._system.action.arm()
        logger.info("Drone %d: armed", self.drone_id)

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
        async for in_air in self._system.telemetry.in_air():
            if in_air:
                logger.info("Drone %d: airborne at %.1f m AGL", self.drone_id, altitude_m)
                break

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
        async for pos_vel in self._system.telemetry.position_velocity_ned():
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
        async for state in self._system.telemetry.landed_state():
            if state == LandedState.ON_GROUND:
                logger.info("Drone %d: on the ground", self.drone_id)
                break

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    async def get_telemetry(self) -> Telemetry:
        """Return a fresh snapshot of position, battery, and landed state."""
        assert self._system is not None

        # Read next value from each telemetry stream independently
        async for pos_vel in self._system.telemetry.position_velocity_ned():
            pos = pos_vel.position
            break

        async for bat in self._system.telemetry.battery():
            # MAVSDK v3.x returns remaining_percent as 0–100 (not 0–1)
            battery_pct = bat.remaining_percent
            break

        from mavsdk.telemetry import LandedState
        async for state in self._system.telemetry.landed_state():
            is_landed = state == LandedState.ON_GROUND
            break

        return Telemetry(
            north_m=pos.north_m,
            east_m=pos.east_m,
            down_m=pos.down_m,
            battery_pct=battery_pct,
            is_landed=is_landed,
        )
