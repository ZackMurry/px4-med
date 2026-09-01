"""PX4 SITL container lifecycle — single container, N drones (default 5).

Follows the same pattern as runner.py:
  - `docker run -d --rm --network host` with start_multi.sh volume-mounted
  - MAVSDK connectivity polled directly (not Docker healthchecks)
  - Fail-fast if the container exits before both drones are ready
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
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

DEFAULT_NUM_DRONES = 5


@dataclass(frozen=True)
class CustomBattery:
    capacity_mah: int = 5000
    n_cells: int = 4
    v_charged: float = 4.20
    v_empty: float = 3.50


class SimulationError(RuntimeError):
    pass


class DockerManager:
    """Manages one PX4 SITL container running N drone instances."""

    def __init__(
        self,
        image: str = DOCKER_IMAGE,
        log_dir: Optional[Path] = None,
        battery: CustomBattery = CustomBattery(),
        num_drones: int = DEFAULT_NUM_DRONES,
    ) -> None:
        self.image = image
        base_log_dir = Path(log_dir) if log_dir else Path(__file__).parents[3] / "logs"
        self.log_dir = base_log_dir.expanduser().resolve()
        self.battery = battery
        self.num_drones = int(num_drones)
        self.container_id: Optional[str] = None

    @property
    def mavsdk_addresses(self) -> list[str]:
        # udpin: MAVSDK listens for PX4's outgoing MAVLink heartbeats
        return [f"udpin://0.0.0.0:{14540 + i}" for i in range(self.num_drones)]

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the N-drone SITL container."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if not _HOST_SCRIPT.is_file():
            raise FileNotFoundError(
                f"start_multi.sh not found at {_HOST_SCRIPT}"
            )

        # PX4_SIM_SPEED_FACTOR is passed ONLY when explicitly set: its
        # default-'1' passthrough made every instance call gz set_physics
        # during boot, randomly corrupting sibling instances' EKF/telemetry
        # (root cause of the frozen-drone gate failures, found 2026-08-29).
        speed_env = []
        speed = os.environ.get("PX4_SIM_SPEED_FACTOR")
        if speed:
            speed_env = ["-e", f"PX4_SIM_SPEED_FACTOR={speed}"]

        cmd = [
            "docker", "run", "-d", "--rm",
            "--network", "host",
            "-v", f"{_HOST_SCRIPT}:{_CONTAINER_SCRIPT}:ro",
            "-v", f"{self.log_dir}:{_CONTAINER_LOG_DIR}",
            "-e", f"NUM_DRONES={self.num_drones}",
            "-e", "PX4_BASE_INSTANCE=0",
            "-e", "DRONE_MODEL=gz_x500",
            "-e", f"PX4_PARAM_BAT1_CAPACITY={float(self.battery.capacity_mah)}",
            "-e", f"PX4_PARAM_BAT1_N_CELLS={int(self.battery.n_cells)}",
            "-e", f"PX4_PARAM_BAT1_V_CHARGED={float(self.battery.v_charged)}",
            "-e", f"PX4_PARAM_BAT1_V_EMPTY={float(self.battery.v_empty)}",
            # Diagnostic passthrough: keep PX4/gz stdout when the host sets it.
            "-e", f"PX4_VERBOSE_LOGS={os.environ.get('PX4_VERBOSE_LOGS', '0')}",
            *speed_env,
            self.image,
            "bash", "-lc", f"bash {_CONTAINER_SCRIPT}",
        ]

        logger.info(
            "Starting PX4 SITL container (image=%s, battery=%dmAh %dS %.2f->%.2fV) ...",
            self.image,
            self.battery.capacity_mah,
            self.battery.n_cells,
            self.battery.v_charged,
            self.battery.v_empty,
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SimulationError(
                f"docker run failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        self.container_id = result.stdout.strip()
        logger.info("Container started: %s", self.container_id[:12])

    def stop(self) -> None:
        """Stop and remove the container, then delete its PX4 log output."""
        if not self.container_id:
            return
        subprocess.run(["docker", "stop", self.container_id], capture_output=True)
        logger.info("Stopped container %s", self.container_id[:12])
        self.container_id = None
        self._cleanup_logs()

    def _cleanup_logs(self) -> None:
        """Delete per-instance PX4 log dirs so long runs don't fill storage.

        Only removes instance_* entries inside the mounted log dir; other
        files (e.g. runner logs written next to them) are left alone.
        """
        import shutil

        if not self.log_dir.is_dir():
            return
        for entry in self.log_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("instance_"):
                shutil.rmtree(entry, ignore_errors=True)
        logger.info("Cleaned PX4 instance logs under %s", self.log_dir)

    # ------------------------------------------------------------------
    # Instance diagnostics / recovery
    # ------------------------------------------------------------------

    _PX4_BIN_DIR = "/root/PX4-Autopilot/build/px4_sitl_default/bin"

    def instance_diagnostics(self, instance: int) -> str:
        """Return EKF/mavlink state of one PX4 instance via its daemon socket."""
        if not self.container_id:
            return "(no container)"
        cmd = (
            f"cd /tmp/instance_{instance} 2>/dev/null && "
            f"{self._PX4_BIN_DIR}/px4-ekf2 status 2>&1 | head -3; "
            f"cd /tmp/instance_{instance} 2>/dev/null && "
            f"{self._PX4_BIN_DIR}/px4-listener vehicle_gps_position 2>&1 | head -4"
        )
        result = subprocess.run(
            ["docker", "exec", self.container_id, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=30,
        )
        return (result.stdout + result.stderr).strip()

    def restart_instance(self, instance: int) -> None:
        """Kill and relaunch one PX4 instance, re-attaching to its gz model.

        Uses PX4_GZ_MODEL_NAME (attach) instead of PX4_GZ_MODEL_POSE (spawn)
        because the model already exists in the running world. Inherits the
        gz partition from the live gz server process.
        """
        if not self.container_id:
            raise SimulationError("No container to restart instance in")
        script = f"""
set -e
for pid in $(pgrep -x px4); do
  if grep -qa "instance_{instance}" /proc/$pid/cmdline 2>/dev/null; then
    kill -9 $pid || true
  fi
done
sleep 2
GZPID=$(pgrep -f "gz sim" | head -1)
export GZ_PARTITION=$(tr '\\0' '\\n' < /proc/$GZPID/environ | grep ^GZ_PARTITION= | cut -d= -f2)
export GZ_SIM_RESOURCE_PATH=/root/PX4-Autopilot/Tools/simulation/gz/models:/root/PX4-Autopilot/Tools/simulation/gz/worlds
rm -rf /tmp/instance_{instance}
mkdir -p /tmp/instance_{instance}
ln -sfn /root/PX4-Autopilot/build/px4_sitl_default/etc /tmp/instance_{instance}/etc
cd /root/PX4-Autopilot
PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500 PX4_GZ_MODEL_NAME=x500_{instance} \
PX4_HOME_LAT=38.8983889 PX4_HOME_LON=-92.2156389 PX4_HOME_ALT=220.0 HEADLESS=1 \
nohup {self._PX4_BIN_DIR}/px4 -d -i {instance} -w /tmp/instance_{instance} \
  >/dev/null 2>&1 &
echo restarted
"""
        result = subprocess.run(
            ["docker", "exec", self.container_id, "bash", "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        if "restarted" not in result.stdout:
            raise SimulationError(
                f"Instance {instance} restart failed: {result.stderr.strip()}"
            )
        logger.info("Restarted PX4 instance %d (attach to existing model)", instance)

    # ------------------------------------------------------------------
    # Health polling
    # ------------------------------------------------------------------

    async def wait_healthy(self, timeout: float = 240.0) -> None:
        """Wait until every PX4 instance responds to MAVSDK probes.

        Checks the container is still running every second and fails fast
        if it exits before all drones are ready (mirrors runner.py).
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
        """Concurrently probe all drones."""
        tasks = [
            self._probe_drone(i, addr, timeout)
            for i, addr in enumerate(self.mavsdk_addresses)
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
