"""PX4 SITL container lifecycle — single container, two drones.

Follows the same pattern as runner.py:
  - `docker run -d --rm --network host` with start_multi.sh volume-mounted
  - MAVSDK connectivity polled directly (not Docker healthchecks)
  - Fail-fast if the container exits before both drones are ready
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DOCKER_IMAGE = "zackmurry/dronevalkit-sim:latest"

_HOST_SCRIPT = Path(__file__).parents[2] / "docker" / "start_multi.sh"
_CONTAINER_SCRIPT = "/root/dronevalkit/start_multi.sh"
_CONTAINER_LOG_DIR = "/root/PX4-Autopilot/build/px4_sitl_default/rootfs/log"

# udpin: MAVSDK listens for PX4's outgoing MAVLink heartbeats
_MAVSDK_ADDRESSES = [
    f"udpin://0.0.0.0:{14540 + i}" for i in range(2)
]


class SimulationError(RuntimeError):
    pass


class DockerManager:
    """Manages one PX4 SITL container running two drone instances."""

    def __init__(
        self,
        image: str = DOCKER_IMAGE,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.image = image
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parents[3] / "logs"
        self.container_id: Optional[str] = None

    @property
    def mavsdk_addresses(self) -> list[str]:
        return _MAVSDK_ADDRESSES

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the two-drone SITL container."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if not _HOST_SCRIPT.is_file():
            raise FileNotFoundError(
                f"start_multi.sh not found at {_HOST_SCRIPT}"
            )

        cmd = [
            "docker", "run", "-d", "--rm",
            "--network", "host",
            "-v", f"{_HOST_SCRIPT}:{_CONTAINER_SCRIPT}:ro",
            "-v", f"{self.log_dir}:{_CONTAINER_LOG_DIR}",
            "-e", "NUM_DRONES=2",
            "-e", "PX4_BASE_INSTANCE=0",
            "-e", "DRONE_MODEL=gz_x500",
            self.image,
            "bash", "-lc", f"bash {_CONTAINER_SCRIPT}",
        ]

        logger.info("Starting PX4 SITL container (image=%s) ...", self.image)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SimulationError(
                f"docker run failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        self.container_id = result.stdout.strip()
        logger.info("Container started: %s", self.container_id[:12])

    def stop(self) -> None:
        """Stop and remove the container."""
        if not self.container_id:
            return
        subprocess.run(["docker", "stop", self.container_id], capture_output=True)
        logger.info("Stopped container %s", self.container_id[:12])
        self.container_id = None

    # ------------------------------------------------------------------
    # Health polling
    # ------------------------------------------------------------------

    async def wait_healthy(self, timeout: float = 120.0) -> None:
        """Wait until both PX4 instances respond to MAVSDK probes.

        Checks the container is still running every second and fails fast
        if it exits before both drones are ready (mirrors runner.py).
        """
        connect_task = asyncio.create_task(self._probe_all(timeout))
        start = time.monotonic()

        try:
            while not connect_task.done():
                if self._container_is_running() is False:
                    connect_task.cancel()
                    tail = self._container_logs_tail()
                    raise SimulationError(
                        f"Container exited after {time.monotonic() - start:.0f}s "
                        f"before MAVSDK was ready.\ndocker logs tail:\n{tail}"
                    )
                await asyncio.sleep(1.0)
            await connect_task  # re-raise any exception from the task
        except asyncio.CancelledError:
            raise

        logger.info(
            "PX4 SITL ready on %s",
            ", ".join(self.mavsdk_addresses),
        )

    async def _probe_all(self, timeout: float) -> None:
        """Concurrently probe both drones."""
        tasks = [
            self._probe_drone(i, addr, timeout)
            for i, addr in enumerate(_MAVSDK_ADDRESSES)
        ]
        await asyncio.gather(*tasks)

    async def _probe_drone(self, drone_id: int, address: str, timeout: float) -> None:
        """Wait until PX4 MAVLink heartbeats arrive on the UDP port.

        Uses a raw UDP socket instead of MAVSDK so that no mavsdk_server
        processes are started during probing.  This avoids port conflicts with
        the mavsdk_server processes that Drone.connect() will start later.
        """
        port = int(address.split(":")[-1])
        deadline = time.monotonic() + timeout
        attempt = 0

        # Brief initial delay so PX4/Gazebo sockets are up
        await asyncio.sleep(2.0)

        while time.monotonic() < deadline:
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            loop = asyncio.get_running_loop()
            received: asyncio.Future = loop.create_future()

            class _Proto(asyncio.DatagramProtocol):
                def datagram_received(self, data, addr):
                    if not received.done():
                        received.set_result(True)

                def error_received(self, exc):
                    if not received.done():
                        received.set_exception(exc)

            try:
                transport, _ = await loop.create_datagram_endpoint(
                    _Proto,
                    local_addr=("0.0.0.0", port),
                )
            except OSError as exc:
                logger.debug(
                    "Drone %d: could not bind port %d (%s), retrying ...",
                    drone_id, port, exc,
                )
                await asyncio.sleep(2.0)
                continue

            try:
                async with asyncio.timeout(min(5.0, remaining)):
                    await received
                logger.info(
                    "Drone %d ready on port %d (attempt %d)",
                    drone_id, port, attempt,
                )
                return
            except TimeoutError:
                logger.debug(
                    "Drone %d port %d: no heartbeat yet (attempt %d)",
                    drone_id, port, attempt,
                )
            finally:
                transport.close()

        raise SimulationError(
            f"Drone {drone_id} ({address}): not ready after {timeout:.0f}s"
        )

    # ------------------------------------------------------------------
    # Helpers (mirrors runner.py)
    # ------------------------------------------------------------------

    def _container_is_running(self) -> Optional[bool]:
        if not self.container_id:
            return None
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container_id],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        state = result.stdout.strip().lower()
        if state == "true":
            return True
        if state == "false":
            return False
        return None

    def _container_logs_tail(self, lines: int = 60) -> str:
        if not self.container_id:
            return ""
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), self.container_id],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return ""
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if stdout and stderr:
            return f"{stdout}\n{stderr}"
        return stdout or stderr
