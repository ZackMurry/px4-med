#!/usr/bin/env python3
"""Fault-tolerant SITL validation runner for the 5-drone CEDA-FGCS mission.

Parent mode: schedules one-episode worker subprocesses. For each attempt it
boots a fresh SITL container, runs the convergence gate, spawns the worker,
monitors its heartbeat file, retries up to 3 times, and incrementally
refreshes aggregate CSV tables. Resumable: rerun with the same --output-dir
and completed jobs are skipped.

Worker mode: connects to the already-booted-and-gated fleet, runs exactly one
episode via px4med.experiments.run_sitl_episode, and writes per-episode
artifacts + heartbeats.

Rewritten 2026-08-29 (5-drone CEDA-FGCS); 2-drone version in git history.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from px4med.boot import check_ports_free, settle_and_gate
from px4med.docker_manager import DockerManager
from px4med.drone import Drone
from px4med.experiments import (
    EpisodeResult,
    InMemoryMetricsCollector,
    StepResult,
    run_sitl_episode,
    suite_lookup,
    summarize_results,
    write_csv,
    write_episode_csv,
    write_step_csv,
)
from px4med.metrics import StepRecord

logger = logging.getLogger("px4med.runner")

NUM_DRONES = 5
EPISODE_COOLDOWN_S = 45.0


# ── job planning ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class JobSpec:
    suite: str
    scenario: str
    policy: str
    episode: int
    seed: int
    order: int

    @property
    def job_id(self) -> str:
        return (
            f"sitl__{self.order:02d}__{self.suite}__{self.scenario}__"
            f"{self.policy}__ep{self.episode:03d}"
        )


def build_pilot_jobs(seed_base: int) -> list[JobSpec]:
    """Small pilot: nominal comparison across all four policies."""
    sequence = [
        ("baseline_comparison", "nominal", "learned", 2),
        ("baseline_comparison", "nominal", "priority_path", 2),
        ("baseline_comparison", "nominal", "nearest_path", 2),
        ("baseline_comparison", "nominal", "random", 1),
    ]
    return _expand(sequence, seed_base)


def build_core_jobs(seed_base: int) -> list[JobSpec]:
    """Overnight core plan (~10-12h at ~20 min/episode)."""
    sequence = [
        ("baseline_comparison", "nominal", "learned", 6),
        ("baseline_comparison", "nominal", "priority_path", 5),
        ("baseline_comparison", "nominal", "nearest_path", 5),
        ("baseline_comparison", "nominal", "random", 2),
        ("battery_sweep", "battery_60", "learned", 3),
        ("battery_sweep", "battery_60", "priority_path", 3),
    ]
    return _expand(sequence, seed_base)


def _expand(
    sequence: list[tuple[str, str, str, int]],
    seed_base: int,
    episode_offset: int = 0,
) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    order = 1
    offset = 0
    for suite, scenario, policy, count in sequence:
        for episode in range(count):
            jobs.append(JobSpec(
                suite=suite, scenario=scenario, policy=policy,
                episode=episode + episode_offset,
                seed=seed_base + offset, order=order,
            ))
            offset += 1
            order += 1
    return jobs


def build_hazard_jobs(seed_base: int) -> list[JobSpec]:
    """Hazard-density sweep: 4 densities x 2 policies x 4 episodes = 32 jobs.

    Interleaved by density rather than grouped by policy so that a run cut
    short still yields a usable (if shorter) curve at every density instead of
    a complete curve for one policy and nothing for the other.
    """
    sequence = []
    for fraction in (50, 100, 150, 200):
        scenario = f"hazard_{fraction:03d}"
        sequence.append(("hazard_sweep", scenario, "learned", 4))
        sequence.append(("hazard_sweep", scenario, "nearest_path", 4))
    return _expand(sequence, seed_base)


def build_extend_jobs(seed_base: int) -> list[JobSpec]:
    """Top up the two arms whose delivery difference is unresolved.

    Core gave learned 0.697 +- 0.055 (n=6) vs nearest_path 0.656 +- 0.045
    (n=5) — overlapping. A power calculation on that effect (delta 0.041,
    pooled sigma ~0.045) wants n ~= 19 per arm for 80% power at alpha 0.05,
    so this adds 13 and 14 episodes respectively.

    IMPORTANT: run this into its OWN --output-dir, then aggregate across the
    two directories. The runner's resume logic keys on job_id, and these
    job_ids restart their episode numbering at 0, so pointing this plan at the
    core directory would collide with existing jobs and silently skip them.
    Episode indices here start at 100 to keep them unambiguous when the two
    episode sets are concatenated for analysis.
    """
    sequence = [
        ("baseline_comparison", "nominal", "learned", 13),
        ("baseline_comparison", "nominal", "nearest_path", 14),
    ]
    return _expand(sequence, seed_base, episode_offset=100)


def build_latency_jobs(seed_base: int) -> list[JobSpec]:
    """Command-latency robustness: 3 delays x learned x 4 episodes."""
    sequence = [
        ("latency_sweep", f"delay_{delay}", "learned", 4)
        for delay in (1, 2, 4)
    ]
    return _expand(sequence, seed_base)


PLANS = {
    "pilot": build_pilot_jobs,
    "core": build_core_jobs,
    "hazard": build_hazard_jobs,
    "extend": build_extend_jobs,
    "latency": build_latency_jobs,
}


# ── small IO helpers ──────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def append_manifest_row(path: Path, row: dict[str, Any]) -> None:
    import csv as _csv

    exists = path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def configure_logging(log_path: Optional[Path] = None, level: str = "INFO") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


# ── aggregation / resume ──────────────────────────────────────────────────────

def load_completed_results(output_dir: Path) -> tuple[list[EpisodeResult], list[StepResult]]:
    episodes: list[EpisodeResult] = []
    steps: list[StepResult] = []
    jobs_dir = output_dir / "jobs"
    if not jobs_dir.exists():
        return episodes, steps
    for job_dir in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        status_path = job_dir / "status.json"
        episode_json = job_dir / "result" / "episode.json"
        steps_jsonl = job_dir / "result" / "steps.jsonl"
        if not (status_path.exists() and episode_json.exists()):
            continue
        if json.loads(status_path.read_text()).get("status") != "completed":
            continue
        episodes.append(EpisodeResult(**json.loads(episode_json.read_text())))
        if steps_jsonl.exists():
            for line in steps_jsonl.read_text().splitlines():
                if line.strip():
                    steps.append(StepResult(**json.loads(line)))
    return episodes, steps


def refresh_aggregate_outputs(output_dir: Path) -> None:
    episodes, steps = load_completed_results(output_dir)
    tables = output_dir / "tables"
    if episodes:
        write_episode_csv(tables / "episodes.csv", episodes)
        write_csv(tables / "summary.csv", summarize_results(episodes))
    if steps:
        write_step_csv(tables / "steps.csv", steps)


def load_completed_job_ids(output_dir: Path) -> set[str]:
    done: set[str] = set()
    jobs_dir = output_dir / "jobs"
    if not jobs_dir.exists():
        return done
    for job_dir in jobs_dir.iterdir():
        status_path = job_dir / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("status") == "completed":
                done.add(status["job_id"])
    return done


# ── worker mode ───────────────────────────────────────────────────────────────

class HeartbeatMetrics(InMemoryMetricsCollector):
    def __init__(self, heartbeat_path: Path, job_id: str, attempt: int) -> None:
        super().__init__()
        self.heartbeat_path = heartbeat_path
        self.job_id = job_id
        self.attempt = attempt
        self.start_ts = time.time()
        self.deliveries = 0
        self.write_status("starting")

    def log_step(self, record: StepRecord) -> None:
        super().log_step(record)
        self.deliveries += len(record.deliveries)
        self.write_status(
            "running",
            last_step=int(record.step) + 1,
            deliveries_so_far=self.deliveries,
            remaining_patients=int(record.remaining_patients),
        )

    def write_status(self, status: str, **extra: Any) -> None:
        payload = {
            "job_id": self.job_id,
            "attempt": self.attempt,
            "status": status,
            "started_at_iso": datetime.fromtimestamp(
                self.start_ts, tz=timezone.utc
            ).isoformat(),
            "last_update_ts": time.time(),
            "elapsed_s": round(time.time() - self.start_ts, 1),
        }
        payload.update(extra)
        atomic_write_json(self.heartbeat_path, payload)


def run_worker(args: argparse.Namespace) -> int:
    configure_logging(level=args.log_level)
    job_dir = Path(args.job_dir)
    result_dir = job_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = HeartbeatMetrics(Path(args.heartbeat_path), args.job_id, args.attempt)

    try:
        suite, scenario = suite_lookup()[(args.suite, args.scenario)]
    except KeyError:
        raise SystemExit(f"Unknown suite/scenario: {args.suite}/{args.scenario}")

    async def _run() -> EpisodeResult:
        drones = [
            Drone(i, f"udpin://0.0.0.0:{14540 + i}", grpc_port=args.grpc_base_port + i)
            for i in range(NUM_DRONES)
        ]
        heartbeat.write_status("connecting")
        for d in drones:
            await d.connect(timeout=300.0)
        heartbeat.write_status("configuring")
        for d in drones:
            await d.configure_battery_profile(args.battery_drain_rate)
            if args.speed_factor != 1.0:
                await d.configure_speed_profile(args.speed_factor)

        heartbeat.write_status("running")
        result, step_results = await run_sitl_episode(
            drones=drones,
            suite=suite,
            scenario=scenario,
            policy_name=args.policy,
            seed=args.seed,
            episode_idx=args.episode,
            metrics=heartbeat,
            step_hz=args.step_hz,
            model_path=Path(args.model) if args.model else None,
        )
        (result_dir / "episode.json").write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True)
        )
        with (result_dir / "steps.jsonl").open("w") as handle:
            for row in step_results:
                handle.write(json.dumps(asdict(row)) + "\n")
        heartbeat.write_status(
            "completed",
            deliveries=result.patients_delivered,
            died=result.patients_died,
            triage_efficiency=result.triage_efficiency,
        )
        return result

    try:
        result = asyncio.run(_run())
    except Exception as exc:
        heartbeat.write_status("failed", last_error=str(exc))
        logger.exception("Worker failed for %s", args.job_id)
        os._exit(1)
    logger.info(
        "Worker %s done: delivered=%d/%d died=%d eff=%.3f steps=%d",
        args.job_id, result.patients_delivered, result.patients_spawned,
        result.patients_died, result.triage_efficiency, result.steps,
    )
    os._exit(0)


# ── parent mode ───────────────────────────────────────────────────────────────

def monitor_attempt(
    process: subprocess.Popen,
    heartbeat_path: Path,
    *,
    timeout_s: float,
    heartbeat_timeout_s: float,
) -> tuple[int, str, float]:
    start = time.time()
    reason = "process exited"
    while True:
        rc = process.poll()
        now = time.time()
        if rc is not None:
            return rc, reason, now - start
        if now - start > timeout_s:
            process.kill()
            process.wait(timeout=15)
            return 124, f"timeout after {timeout_s:.0f}s", now - start
        if heartbeat_path.exists():
            try:
                hb = json.loads(heartbeat_path.read_text())
                age = now - float(hb.get("last_update_ts", start))
                if age > heartbeat_timeout_s:
                    process.kill()
                    process.wait(timeout=15)
                    return 125, f"stale heartbeat ({age:.0f}s)", now - start
            except json.JSONDecodeError:
                pass
        time.sleep(5.0)


def _cleanup_stale_processes() -> None:
    """Reap mavsdk_server orphans between attempts (ports must be free)."""
    subprocess.run(["pkill", "-9", "-f", "mavsdk/bi[n]"], capture_output=True)
    time.sleep(1.0)


def run_parent(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    for sub in ("jobs", "tables", "logs", "sitl_logs"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "logs" / "runner.log", args.log_level)

    seed_base = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    jobs = PLANS[args.plan](seed_base)
    write_csv(output_dir / "plan.csv", [
        {"plan": args.plan, "order": j.order, "job_id": j.job_id, "suite": j.suite,
         "scenario": j.scenario, "policy": j.policy, "episode": j.episode,
         "seed": j.seed}
        for j in jobs
    ])
    refresh_aggregate_outputs(output_dir)
    completed = load_completed_job_ids(output_dir)
    logger.info("Plan %s: %d jobs (seed base %d), %d already complete",
                args.plan, len(jobs), seed_base, len(completed))

    started = time.time()
    for job in jobs:
        if time.time() - started > args.max_hours * 3600:
            logger.warning("Max runtime budget reached (%.1fh)", args.max_hours)
            break
        if job.job_id in completed:
            logger.info("Skipping completed job %s", job.job_id)
            continue

        job_dir = output_dir / "jobs" / job.job_id
        heartbeat_path = job_dir / "heartbeat.json"
        job_done = False

        for attempt in range(1, 4):
            attempt_dir = job_dir / "attempts" / f"attempt_{attempt:02d}"
            if attempt_dir.exists():
                shutil.rmtree(attempt_dir)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            if heartbeat_path.exists():
                heartbeat_path.unlink()
            atomic_write_json(job_dir / "status.json", {
                "job_id": job.job_id, "status": "running", "attempt": attempt,
                "started_at": now_iso(), "suite": job.suite,
                "scenario": job.scenario, "policy": job.policy,
                "episode": job.episode, "seed": job.seed,
            })
            logger.info("Job %s attempt %d: booting fresh SITL fleet", job.job_id, attempt)

            _cleanup_stale_processes()
            try:
                check_ports_free(NUM_DRONES)
            except SystemExit as exc:
                logger.error("Ports not free before attempt: %s", exc)
                _cleanup_stale_processes()

            dm = DockerManager(log_dir=output_dir / "sitl_logs", num_drones=NUM_DRONES)
            exit_code, reason, duration = 1, "boot failed", 0.0
            try:
                dm.start()
                asyncio.run(dm.wait_healthy())
                asyncio.run(settle_and_gate(dm, NUM_DRONES))

                cmd = [
                    sys.executable, str(Path(__file__).resolve()), "--worker",
                    "--job-id", job.job_id,
                    "--job-dir", str(job_dir),
                    "--heartbeat-path", str(heartbeat_path),
                    "--suite", job.suite, "--scenario", job.scenario,
                    "--policy", job.policy,
                    "--episode", str(job.episode), "--seed", str(job.seed),
                    "--attempt", str(attempt),
                    "--model", str(args.model),
                    "--step-hz", str(args.step_hz),
                    "--grpc-base-port", str(args.grpc_base_port),
                    "--speed-factor", str(args.speed_factor),
                    "--battery-drain-rate", str(args.battery_drain_rate),
                    "--log-level", args.log_level,
                ]
                env = os.environ.copy()
                env.setdefault("PYTHONPATH", str(SRC_ROOT))
                with (attempt_dir / "worker.log").open("w") as log_handle:
                    process = subprocess.Popen(
                        cmd, cwd=REPO_ROOT, stdout=log_handle,
                        stderr=subprocess.STDOUT, text=True, env=env,
                    )
                    exit_code, reason, duration = monitor_attempt(
                        process, heartbeat_path,
                        timeout_s=args.episode_timeout_min * 60,
                        heartbeat_timeout_s=args.heartbeat_timeout_s,
                    )
            except Exception as exc:
                logger.exception("Attempt infrastructure failure: %s", exc)
                reason = f"infra: {exc}"
            finally:
                dm.stop()
                _cleanup_stale_processes()
                time.sleep(args.episode_cooldown_s)

            append_manifest_row(output_dir / "manifest.csv", {
                "timestamp": now_iso(), "job_id": job.job_id, "attempt": attempt,
                "exit_code": exit_code, "duration_s": round(duration, 1),
                "status": "completed" if exit_code == 0 else "failed",
                "reason": reason,
            })
            if exit_code == 0:
                atomic_write_json(job_dir / "status.json", {
                    "job_id": job.job_id, "status": "completed",
                    "attempt": attempt, "completed_at": now_iso(),
                    "duration_s": round(duration, 1),
                })
                refresh_aggregate_outputs(output_dir)
                completed.add(job.job_id)
                job_done = True
                logger.info("Job %s completed in %.1f min", job.job_id, duration / 60)
                break
            logger.error("Job %s attempt %d failed (%s)", job.job_id, attempt, reason)

        if not job_done:
            atomic_write_json(job_dir / "status.json", {
                "job_id": job.job_id, "status": "abandoned",
                "updated_at": now_iso(),
            })
            logger.error("Abandoning job %s after 3 attempts", job.job_id)

    refresh_aggregate_outputs(output_dir)
    logger.info("Plan finished.")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    default_out = Path("results") / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    p = argparse.ArgumentParser(description="5-drone CEDA-FGCS SITL validation runner")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--plan", default="pilot", choices=sorted(PLANS))
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument("--model", type=Path, default=Path("models/ctde_agent_marl_FGCS.pth"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--step-hz", type=float, default=2.0)
    p.add_argument("--grpc-base-port", type=int, default=50051)
    p.add_argument("--speed-factor", type=float, default=3.0)
    p.add_argument("--battery-drain-rate", type=float, default=0.0,
                   help="PX4 SIM battery drain (0 = disabled; the model uses "
                        "its own mission-energy ledger)")
    p.add_argument("--max-hours", type=float, default=12.0)
    p.add_argument("--episode-timeout-min", type=float, default=60.0)
    p.add_argument("--episode-cooldown-s", type=float, default=EPISODE_COOLDOWN_S)
    p.add_argument("--heartbeat-timeout-s", type=float, default=300.0)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # worker-only args
    p.add_argument("--job-id", default=None, help=argparse.SUPPRESS)
    p.add_argument("--job-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--heartbeat-path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--suite", default=None, help=argparse.SUPPRESS)
    p.add_argument("--scenario", default=None, help=argparse.SUPPRESS)
    p.add_argument("--policy", default=None, help=argparse.SUPPRESS)
    p.add_argument("--episode", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--attempt", type=int, default=1, help=argparse.SUPPRESS)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.worker:
        required = (args.job_id, args.job_dir, args.heartbeat_path,
                    args.suite, args.scenario, args.policy, args.episode)
        if any(v is None for v in required):
            raise SystemExit("Missing required worker arguments")
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
