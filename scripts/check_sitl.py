#!/usr/bin/env python3
"""Level-3 integration test: start the SITL container and verify both drones stream telemetry.

Usage:
    # Start container, wait for both drones, print telemetry, then stop:
    python scripts/check_sitl.py

    # Skip container start (container already running):
    python scripts/check_sitl.py --no-start

    # Only check one drone:
    python scripts/check_sitl.py --drone 0
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from px4med.docker_manager import DockerManager
from px4med.drone import Drone


async def main(start_container: bool, drone_arg: str, timeout: float) -> None:
    dm = DockerManager()

    if start_container:
        dm.start()
        print(f"Container started: {dm.container_id[:12]}")

    try:
        print(f"Waiting for both PX4 instances (timeout={timeout:.0f}s) ...")
        await dm.wait_healthy(timeout=timeout)
        print("  Both drones reachable via MAVSDK.\n")

        drone_ids = [0, 1] if drone_arg == "both" else [int(drone_arg)]
        for drone_id in drone_ids:
            address = dm.mavsdk_addresses[drone_id]
            print(f"Drone {drone_id}  ({address})")
            d = Drone(drone_id=drone_id, mavsdk_address=address)
            await d.connect(timeout=30.0)
            t = await d.get_telemetry()
            print(f"  north_m   : {t.north_m:.3f}")
            print(f"  east_m    : {t.east_m:.3f}")
            print(f"  down_m    : {t.down_m:.3f}")
            print(f"  battery   : {t.battery_pct:.1f}%")
            print(f"  is_landed : {t.is_landed}\n")

        print("All checks passed.")

    finally:
        if start_container:
            dm.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PX4 SITL connectivity and telemetry")
    parser.add_argument("--no-start", action="store_true",
                        help="Skip docker start (container already running)")
    parser.add_argument("--drone", default="both", choices=["0", "1", "both"],
                        help="Which drone(s) to check (default: both)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Seconds to wait for SITL ready (default: 120)")
    args = parser.parse_args()
    asyncio.run(main(not args.no_start, args.drone, args.timeout))
