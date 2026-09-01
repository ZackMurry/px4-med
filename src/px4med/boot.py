"""Reusable SITL fleet boot/convergence helpers.

Encapsulates the hardened boot sequence developed 2026-08-29:
  settle → per-drone ephemeral probe gate → (diagnose + instance restart +
  re-probe on failure). Used by px4med.main and the experiment runner so both
  paths share identical, battle-tested logic.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .docker_manager import DockerManager

logger = logging.getLogger(__name__)

_PROBE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_health.py"

DEFAULT_SETTLE_S = 120.0
PROBE_BUDGET_S = 240
REPROBE_BUDGET_S = 300


def check_ports_free(num_drones: int, base_port: int = 14540) -> None:
    """Fail fast if MAVSDK udpin ports are held by an orphaned previous run."""
    for i in range(num_drones):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind(("0.0.0.0", base_port + i))
        except OSError as exc:
            raise SystemExit(
                f"UDP port {base_port + i} already bound (stale mavsdk_server "
                f"from a previous run?): {exc}. Kill leftover processes and retry."
            )
        finally:
            probe.close()


def _port_holder(port: int) -> str:
    out = subprocess.run(["ss", "-ulpn"], capture_output=True, text=True).stdout
    return next((l for l in out.splitlines() if f":{port} " in l), "(free)")


def kill_stale_udp_holder(port: int) -> None:
    """Kill any lingering mavsdk_server still bound to this udpin port."""
    import re

    out = subprocess.run(["ss", "-ulpn"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{port} " in line and "mavsdk_server" in line:
            m = re.search(r"pid=(\d+)", line)
            if m:
                subprocess.run(["kill", "-9", m.group(1)], capture_output=True)
                logger.warning(
                    "Killed stale mavsdk_server pid %s holding port %d",
                    m.group(1), port,
                )


async def _probe(index: int, budget: int, grpc_port: int, base_port: int) -> int:
    """Run one ephemeral health probe. Never blocks longer than the budget.

    The hard timeout is not redundant with the probe's own `--timeout`: if the
    probe's mavsdk_server child dies, the gRPC stream never yields and the probe
    can hang inside library code. That once stalled the gate for 6 h unnoticed,
    because nothing above this function watches the parent's gate phase. Treat a
    hang as a probe failure so the caller's restart_instance() recovery runs.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(_PROBE_SCRIPT),
        "--port", str(base_port + index),
        "--grpc-port", str(grpc_port),
        "--sysid", str(230 + index),
        "--timeout", str(budget),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        return await asyncio.wait_for(proc.wait(), timeout=budget + 60)
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "Probe for drone %d exceeded %ds wall clock — killing it and "
            "treating as failure", index, budget + 60,
        )
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=15)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Probe for drone %d did not die after kill", index)
        return 1


async def settle_and_gate(
    dm: Optional[DockerManager],
    num_drones: int,
    *,
    base_port: int = 14540,
    settle_s: Optional[float] = None,
) -> None:
    """Settle after boot, then gate every drone on an ephemeral health probe.

    On probe failure: log PX4 internals, restart that instance (fresh mavlink
    links), settle again, re-probe once. Raises RuntimeError if still failing.
    """
    if settle_s is None:
        settle_s = float(os.environ.get("PX4_ATTACH_SETTLE_S", str(DEFAULT_SETTLE_S)))
    logger.info("Settling %.0fs before first MAVSDK contact ...", settle_s)
    await asyncio.sleep(settle_s)

    for i in range(num_drones):
        logger.info("Convergence gate: probing drone %d ...", i)
        rc = await _probe(i, PROBE_BUDGET_S, 50090 + i, base_port)
        if rc != 0 and dm is not None:
            diag = dm.instance_diagnostics(i)
            logger.warning(
                "Convergence gate: drone %d not converged after %ds. "
                "PX4 internals:\n%s\nport holder: %s",
                i, PROBE_BUDGET_S, diag, _port_holder(base_port + i),
            )
            kill_stale_udp_holder(base_port + i)
            logger.warning("Convergence gate: restarting PX4 instance %d", i)
            dm.restart_instance(i)
            await asyncio.sleep(settle_s)
            rc = await _probe(i, REPROBE_BUDGET_S, 50100 + i, base_port)
        if rc != 0:
            diag = dm.instance_diagnostics(i) if dm is not None else "n/a"
            raise RuntimeError(
                f"Convergence gate failed for drone {i} even after instance "
                f"restart. PX4 internals:\n{diag}\n"
                f"port holder: {_port_holder(base_port + i)}"
            )
        logger.info("Convergence gate: drone %d converged.", i)
