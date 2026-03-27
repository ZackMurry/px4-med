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
WAYPOINT_ARRIVAL_RADIUS_M = 0.5
# How long to wait for drone to reach a waypoint before moving on (seconds)
WAYPOINT_TIMEOUT_S = 30.0


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

    async def takeoff(self, altitude_m: float = 5.0) -> None:
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

    async def send_waypoint(self, north_m: float, east_m: float, down_m: float) -> None:
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
        deadline = loop.time() + WAYPOINT_TIMEOUT_S
        async for pos_vel in self._system.telemetry.position_velocity_ned():
            p = pos_vel.position
            dist = math.sqrt(
                (p.north_m - north_m) ** 2
                + (p.east_m - east_m) ** 2
                + (p.down_m - down_m) ** 2
            )
            if dist < WAYPOINT_ARRIVAL_RADIUS_M or loop.time() > deadline:
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
