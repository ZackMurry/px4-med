#!/usr/bin/env python3
"""Probe one drone's MAVSDK-visible arming preconditions.

Run as an ephemeral subprocess by `boot.settle_and_gate()` — one process per
drone, before the episode worker starts, so a bad instance can be restarted
while it is still cheap.

Requires local AND global AND home position, not just local. Why: there is an
intermittent fault (HANDOFF.md §5, §5b) where PX4's own EKF is completely
healthy — `px4-ekf2 status` reports `attitude: 1, local position: 1, global
position: 1` and `home_position` is published — while the MAVSDK view of that
one instance sits at `global=False home=False` indefinitely. PX4 then refuses
to arm that drone, so the episode dies ~18 minutes later on
`ActionError: COMMAND_DENIED` after burning the arm retry window.

Gating on local position alone let those instances through. A healthy drone
satisfies all three within about a second, and an affected one never does, so
requiring them is both cheap and sharply discriminating — and it routes the
failure into the gate's existing `restart_instance()` + re-probe recovery.
"""
from __future__ import annotations

import argparse
import asyncio

from mavsdk import System


def _flags(health) -> str:
    if health is None:
        return "no health messages"
    return (
        f"gyro={health.is_gyrometer_calibration_ok} "
        f"accel={health.is_accelerometer_calibration_ok} "
        f"mag={health.is_magnetometer_calibration_ok} "
        f"local={health.is_local_position_ok} "
        f"global={health.is_global_position_ok} "
        f"home={health.is_home_position_ok} "
        # Logged, deliberately not gated on: is_armable is the most direct
        # signal PX4 will accept an arm command, but we have no evidence yet
        # that it ever goes true in this SITL setup, and gating on something
        # that never fires would fail every boot. Collect data first.
        f"armable={health.is_armable}"
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=14540)
    parser.add_argument("--grpc-port", type=int, default=50061)
    parser.add_argument("--sysid", type=int, default=251)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--allow-local-only", action="store_true",
        help="Accept local position alone (pre-2026-08-30 behaviour). Escape "
             "hatch if global/home ever proves legitimately unavailable.",
    )
    args = parser.parse_args()

    s = System(port=args.grpc_port, sysid=args.sysid)
    await s.connect(system_address=f"udpin://0.0.0.0:{args.port}")

    # The connect phase MUST be bounded. If the mavsdk_server child dies (it
    # goes <defunct> and the gRPC stream simply never yields), this loop blocks
    # forever — which once hung the convergence gate for 6 h while Gazebo burned
    # 83% CPU, because boot._probe only did a bare `await proc.wait()` and the
    # runner's heartbeat watchdog covers worker processes, not the gate.
    connect_budget = max(30.0, args.timeout / 2)
    try:
        async with asyncio.timeout(connect_budget):
            async for state in s.core.connection_state():
                if state.is_connected:
                    print(f"connected on {args.port}", flush=True)
                    break
    except TimeoutError:
        print(
            f"TIMEOUT: no MAVSDK connection on {args.port} within "
            f"{connect_budget:.0f}s (mavsdk_server dead?)",
            flush=True,
        )
        return 1

    loop = asyncio.get_event_loop()
    deadline = loop.time() + args.timeout
    last = None
    async for h in s.telemetry.health():
        last = h
        ready = h.is_local_position_ok and (
            args.allow_local_only
            or (h.is_global_position_ok and h.is_home_position_ok)
        )
        if ready:
            print(f"POSITION HEALTH OK ({_flags(h)})", flush=True)
            return 0
        if loop.time() > deadline:
            break
    print(f"TIMEOUT: {_flags(last)}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
