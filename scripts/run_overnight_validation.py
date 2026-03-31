#!/usr/bin/env python3
"""Run a fault-tolerant overnight PX4 SITL validation plan.

Parent mode schedules one-episode subprocess jobs, monitors heartbeats,
retries failures up to three times, and incrementally refreshes aggregate CSV
tables and matplotlib figures.

Worker mode executes exactly one SITL episode using the existing
px4med.experiments helpers and writes per-episode artifacts.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from px4med.coordinator import Coordinator
from px4med.docker_manager import DockerManager
from px4med.drone import Drone
from px4med.environment import WorldEnvironment
from px4med.experiments import EpisodeResult
from px4med.experiments import InMemoryMetricsCollector
from px4med.experiments import ScenarioDef
from px4med.experiments import StepResult
from px4med.experiments import SuiteDef
from px4med.experiments import build_default_suites
from px4med.experiments import build_policy_controller
from px4med.experiments import build_sitl_result
from px4med.experiments import build_step_results
from px4med.experiments import plot_episode_details
from px4med.experiments import plot_suite
from px4med.experiments import summarize_results
from px4med.experiments import write_csv
from px4med.experiments import write_episode_csv
from px4med.experiments import write_step_csv
from px4med.experiments import write_summary_csv
from px4med.metrics import StepRecord
from px4med.policy import PolicyNet


logger = logging.getLogger("px4med.overnight")

PLAN_NAME = "paper_core_12h"
NOMINAL_SUITE = "baseline_comparison"
NOMINAL_SCENARIO = "nominal"
BATTERY_SUITE = "battery_sweep"
BATTERY_SCENARIO = "battery_35"
HAZARD_SUITE = "hazard_sweep"
HAZARD_SCENARIO = "hazard_high"
DELAY_SUITE = "delay_sweep"
DELAY_SCENARIO = "delay_3"
EPISODE_COOLDOWN_S = 180.0


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            path.write_text(existing + "\n", encoding="utf-8")
    rows = []
    if path.exists() and path.stat().st_size > 0:
        return _append_csv_row_existing(path, row)
    rows.append(row)
    write_csv(path, rows)


def _append_csv_row_existing(path: Path, row: dict[str, Any]) -> None:
    import csv

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def configure_logging(log_path: Path | None = None, level: str = "INFO") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def build_core_12h_jobs(seed_base: int) -> list[JobSpec]:
    sequence: list[tuple[str, str, str, int]] = [
        (NOMINAL_SUITE, NOMINAL_SCENARIO, "learned", 5),
        (NOMINAL_SUITE, NOMINAL_SCENARIO, "priority_path", 5),
        (NOMINAL_SUITE, NOMINAL_SCENARIO, "nearest_path", 5),
        (BATTERY_SUITE, BATTERY_SCENARIO, "learned", 2),
        (BATTERY_SUITE, BATTERY_SCENARIO, "priority_path", 2),
        (HAZARD_SUITE, HAZARD_SCENARIO, "learned", 1),
        (HAZARD_SUITE, HAZARD_SCENARIO, "priority_path", 1),
        (DELAY_SUITE, DELAY_SCENARIO, "learned", 1),
    ]
    jobs: list[JobSpec] = []
    seed_offset = 0
    order = 1
    for suite, scenario, policy, episodes in sequence:
        for episode in range(episodes):
            jobs.append(
                JobSpec(
                    suite=suite,
                    scenario=scenario,
                    policy=policy,
                    episode=episode,
                    seed=seed_base + seed_offset,
                    order=order,
                )
            )
            seed_offset += 1
            order += 1
    return jobs


def suite_lookup() -> dict[tuple[str, str], tuple[SuiteDef, ScenarioDef]]:
    lookup: dict[tuple[str, str], tuple[SuiteDef, ScenarioDef]] = {}
    for suite in build_default_suites():
        for scenario in suite.scenarios:
            lookup[(suite.name, scenario.name)] = (suite, scenario)
    return lookup


def write_plan_csv(path: Path, jobs: list[JobSpec]) -> None:
    rows = [
        {
            "plan": PLAN_NAME,
            "order": job.order,
            "job_id": job.job_id,
            "suite": job.suite,
            "scenario": job.scenario,
            "policy": job.policy,
            "episode": job.episode,
            "seed": job.seed,
        }
        for job in jobs
    ]
    write_csv(path, rows)


class HeartbeatMetricsCollector(InMemoryMetricsCollector):
    """Capture step records and persist worker liveness to disk."""

    def __init__(self, heartbeat_path: Path, job: JobSpec, attempt: int) -> None:
        super().__init__()
        self.heartbeat_path = heartbeat_path
        self.job = job
        self.attempt = attempt
        self.delivery_count = 0
        self.start_ts = time.time()
        self.last_step = -1
        self.write_status(status="starting", note="worker initialized")

    def log_step(self, record: StepRecord) -> None:
        super().log_step(record)
        self.last_step = int(record.step)
        self.delivery_count += len(record.deliveries)
        self.write_status(
            status="running",
            last_step=int(record.step) + 1,
            deliveries_so_far=self.delivery_count,
            remaining_patients=int(record.remaining_patients),
            last_known_battery=[float(record.drone0_battery), float(record.drone1_battery)],
        )

    def write_status(self, *, status: str, note: str | None = None, **extra: Any) -> None:
        payload = {
            "job_id": self.job.job_id,
            "suite": self.job.suite,
            "scenario": self.job.scenario,
            "policy": self.job.policy,
            "episode": self.job.episode,
            "seed": self.job.seed,
            "attempt": self.attempt,
            "status": status,
            "started_at": self.start_ts,
            "started_at_iso": datetime.fromtimestamp(self.start_ts, tz=timezone.utc).isoformat(),
            "last_update_ts": time.time(),
            "last_update_iso": now_iso(),
            "elapsed_s": round(time.time() - self.start_ts, 1),
            "last_step": self.last_step + 1 if self.last_step >= 0 else 0,
        }
        if note is not None:
            payload["note"] = note
        payload.update(extra)
        atomic_write_json(self.heartbeat_path, payload)


def _parse_episode_result(payload: dict[str, Any]) -> EpisodeResult:
    return EpisodeResult(**payload)


def _parse_step_result(payload: dict[str, Any]) -> StepResult:
    return StepResult(**payload)


def load_completed_results(output_dir: Path) -> tuple[list[EpisodeResult], list[StepResult]]:
    episode_results: list[EpisodeResult] = []
    step_results: list[StepResult] = []
    jobs_dir = output_dir / "jobs"
    if not jobs_dir.exists():
        return episode_results, step_results

    for job_dir in sorted(path for path in jobs_dir.iterdir() if path.is_dir()):
        status_path = job_dir / "status.json"
        result_dir = job_dir / "result"
        episode_json = result_dir / "episode.json"
        steps_jsonl = result_dir / "steps.jsonl"
        if not status_path.exists() or not episode_json.exists() or not steps_jsonl.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            continue
        episode_results.append(
            _parse_episode_result(json.loads(episode_json.read_text(encoding="utf-8")))
        )
        with steps_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                step_results.append(_parse_step_result(json.loads(line)))
    return episode_results, step_results


def refresh_aggregate_outputs(output_dir: Path) -> None:
    episodes, steps = load_completed_results(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if episodes:
        write_episode_csv(tables_dir / "episodes.csv", episodes)
        summaries = summarize_results(episodes)
        write_summary_csv(tables_dir / "summary.csv", summaries)
        suite_defs = {suite.name: suite for suite in build_default_suites()}
        for suite_name in sorted({episode.suite for episode in episodes}):
            suite_rows = [row for row in summaries if row["suite"] == suite_name]
            if not suite_rows:
                continue
            write_summary_csv(tables_dir / f"{suite_name}_summary.csv", suite_rows)
            plot_suite(suite_defs[suite_name], suite_rows, figures_dir / suite_name)
    if steps:
        write_step_csv(tables_dir / "steps.csv", steps)
    if episodes and steps:
        plot_episode_details(steps, episodes, figures_dir / "episodes")


def write_manifest_row(output_dir: Path, row: dict[str, Any]) -> None:
    append_csv_row(output_dir / "manifest.csv", row)


def write_job_status(job_dir: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(job_dir / "status.json", payload)


def update_live_status(output_dir: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(output_dir / "live_status.json", payload)


def prepare_output_dirs(output_dir: Path) -> None:
    (output_dir / "jobs").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)


def load_completed_job_ids(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    jobs_dir = output_dir / "jobs"
    if not jobs_dir.exists():
        return completed
    for job_dir in jobs_dir.iterdir():
        if not job_dir.is_dir():
            continue
        status_path = job_dir / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            completed.add(status["job_id"])
    return completed


def restart_shared_sitl(dm: DockerManager) -> None:
    logger.warning("Restarting shared PX4 SITL container before continuing")
    dm.stop()
    dm.start()
    asyncio.run(dm.wait_healthy())


def run_worker(args: argparse.Namespace) -> int:
    configure_logging(level=args.log_level)
    heartbeat_path = Path(args.heartbeat_path)
    job_dir = Path(args.job_dir)
    result_dir = job_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    lookup = suite_lookup()
    try:
        suite, scenario = lookup[(args.suite, args.scenario)]
    except KeyError as exc:
        raise SystemExit(f"Unknown suite/scenario: {args.suite}/{args.scenario}") from exc

    job = JobSpec(
        suite=args.suite,
        scenario=args.scenario,
        policy=args.policy,
        episode=args.episode,
        seed=args.seed,
        order=args.order,
    )
    metrics = HeartbeatMetricsCollector(heartbeat_path=heartbeat_path, job=job, attempt=args.attempt)

    learned_policy = PolicyNet(Path(args.model)) if args.policy == "learned" else None
    if learned_policy is not None:
        logger.info("Loaded learned policy from %s on %s", args.model, learned_policy.device)

    async def _async_worker() -> EpisodeResult:
        drones: list[Drone] = []
        try:
            addresses = [f"udpin://0.0.0.0:{14540 + i}" for i in range(2)]
            drones = [
                Drone(i, addresses[i], grpc_port=args.grpc_base_port + i)
                for i in range(len(addresses))
            ]
            metrics.write_status(status="connecting", note="connecting to PX4 drones")
            for drone in drones:
                await drone.connect()
            metrics.write_status(status="configuring", note="configuring PX4 parameters")
            for drone in drones:
                await drone.configure_battery_profile(args.battery_drain_rate)
            if args.speed_factor != 1.0:
                for drone in drones:
                    await drone.configure_speed_profile(args.speed_factor)

            controller = build_policy_controller(
                policy_name=args.policy,
                learned_policy=learned_policy,
                seed=args.seed,
            )
            world = WorldEnvironment(copy.deepcopy(scenario.world))
            coordinator = Coordinator(
                drones=drones,
                policy=controller,
                world=world,
                metrics=metrics,
                step_hz=args.step_hz,
                action_delay_steps=scenario.action_delay_steps,
                enable_cycle_breaking=False,
            )
            metrics.write_status(status="running", note="episode initialized")
            summary = await coordinator.run_episode(
                episode=args.episode,
                max_steps=scenario.max_steps,
            )
            step_results = build_step_results(
                backend="sitl",
                suite=suite.name,
                scenario=scenario.name,
                policy=args.policy,
                episode=args.episode,
                step_records=metrics.step_records,
            )
            result = build_sitl_result(
                suite=suite,
                scenario=scenario,
                policy_name=args.policy,
                seed=args.seed,
                episode_idx=args.episode,
                world=world,
                summary=summary,
                step_records=metrics.step_records,
            )
            write_episode_csv(result_dir / "episode.csv", [result])
            write_step_csv(result_dir / "steps.csv", step_results)
            (result_dir / "episode.json").write_text(
                json.dumps(asdict(result), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            with (result_dir / "steps.jsonl").open("w", encoding="utf-8") as handle:
                for row in step_results:
                    handle.write(json.dumps(asdict(row)) + "\n")
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "job_id": job.job_id,
                        "status": "completed",
                        "completed_at": now_iso(),
                        "episode": asdict(result),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            plot_episode_details(step_results, [result], result_dir / "figures")
            metrics.write_status(
                status="completed",
                note="episode complete",
                deliveries=result.patients_delivered,
                deaths=result.patients_died,
                triage_efficiency=result.triage_efficiency,
                battery_margin_min=result.battery_margin_min,
                wrong_land_attempts=result.wrong_land_attempts,
                mean_tracking_error_m=result.mean_tracking_error_m,
            )
            return result
        finally:
            metrics.write_status(status="running", note="worker cleanup landing drones")
            for drone in drones:
                try:
                    await drone.land()
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    logger.warning("Worker cleanup could not land drone %d: %s", drone.drone_id, exc)

    try:
        result = asyncio.run(_async_worker())
    except Exception as exc:
        metrics.write_status(status="failed", note="worker exception", last_error=str(exc))
        logger.exception("Worker failed for job %s", job.job_id)
        return 1

    logger.info(
        "Worker completed %s delivered=%d/%d died=%d eff=%.3f reward=%.2f steps=%d",
        job.job_id,
        result.patients_delivered,
        result.patients_spawned,
        result.patients_died,
        result.triage_efficiency,
        result.total_reward,
        result.steps,
    )
    return 0


def monitor_attempt(
    *,
    process: subprocess.Popen[str],
    job: JobSpec,
    attempt: int,
    heartbeat_path: Path,
    output_dir: Path,
    timeout_s: float,
    heartbeat_timeout_s: float,
    log_interval_s: float,
) -> tuple[int, str, float]:
    start = time.time()
    next_log = start + 5.0
    failure_reason = "process exited"
    last_heartbeat: dict[str, Any] | None = None

    while True:
        rc = process.poll()
        now = time.time()

        if heartbeat_path.exists():
            try:
                last_heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                last_heartbeat = None

        if rc is not None:
            return rc, failure_reason, now - start

        if now - start > timeout_s:
            failure_reason = f"timeout after {timeout_s:.0f}s"
            process.kill()
            process.wait(timeout=10)
            return 124, failure_reason, now - start

        if last_heartbeat is not None:
            heartbeat_age = now - float(last_heartbeat.get("last_update_ts", start))
            if heartbeat_age > heartbeat_timeout_s:
                failure_reason = f"stale heartbeat ({heartbeat_age:.0f}s)"
                process.kill()
                process.wait(timeout=10)
                return 125, failure_reason, now - start

        if now >= next_log:
            if last_heartbeat is None:
                logger.info(
                    "Job %s attempt %d running %.1fm with no heartbeat yet",
                    job.job_id,
                    attempt,
                    (now - start) / 60.0,
                )
                live = {
                    "job_id": job.job_id,
                    "attempt": attempt,
                    "status": "running",
                    "elapsed_s": round(now - start, 1),
                    "heartbeat": None,
                    "updated_at": now_iso(),
                }
            else:
                logger.info(
                    "Job %s attempt %d running %.1fm status=%s step=%s deliveries=%s remaining=%s battery=%s note=%s error=%s",
                    job.job_id,
                    attempt,
                    (now - start) / 60.0,
                    last_heartbeat.get("status"),
                    last_heartbeat.get("last_step"),
                    last_heartbeat.get("deliveries_so_far"),
                    last_heartbeat.get("remaining_patients"),
                    last_heartbeat.get("last_known_battery"),
                    last_heartbeat.get("note"),
                    last_heartbeat.get("last_error"),
                )
                live = {
                    "job_id": job.job_id,
                    "attempt": attempt,
                    "status": "running",
                    "elapsed_s": round(now - start, 1),
                    "heartbeat": last_heartbeat,
                    "updated_at": now_iso(),
                }
            update_live_status(output_dir, live)
            next_log = now + log_interval_s

        time.sleep(5.0)


def run_parent(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare_output_dirs(output_dir)
    configure_logging(log_path=output_dir / "logs" / "runner.log", level=args.log_level)

    seed_base = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**32)
    jobs = build_core_12h_jobs(seed_base)
    write_plan_csv(output_dir / "plan.csv", jobs)
    refresh_aggregate_outputs(output_dir)

    logger.info("Output directory: %s", output_dir)
    logger.info("Plan: %s (%d jobs, seed base=%d)", PLAN_NAME, len(jobs), seed_base)

    started = time.time()
    completed_ids = load_completed_job_ids(output_dir)

    for job in jobs:
        elapsed = time.time() - started
        if elapsed >= args.max_hours * 3600.0:
            logger.warning("Reached max runtime budget of %.1f hours", args.max_hours)
            break

        job_dir = output_dir / "jobs" / job.job_id
        attempts_dir = job_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        heartbeat_path = job_dir / "heartbeat.json"

        if job.job_id in completed_ids:
            logger.info("Skipping completed job %s", job.job_id)
            continue

        logger.info(
            "Starting job %s [%d/%d]: %s/%s/%s episode=%d seed=%d",
            job.job_id,
            job.order,
            len(jobs),
            job.suite,
            job.scenario,
            job.policy,
            job.episode,
            job.seed,
        )

        job_success = False
        try:
            for attempt in range(1, 4):
                    attempt_dir = attempts_dir / f"attempt_{attempt:02d}"
                    if attempt_dir.exists():
                        shutil.rmtree(attempt_dir)
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    attempt_log = attempt_dir / "worker.log"
                    if heartbeat_path.exists():
                        heartbeat_path.unlink()

                    write_job_status(
                        job_dir,
                        {
                            "job_id": job.job_id,
                            "status": "running",
                            "attempt": attempt,
                            "started_at": now_iso(),
                            "suite": job.suite,
                            "scenario": job.scenario,
                            "policy": job.policy,
                            "episode": job.episode,
                            "seed": job.seed,
                        },
                    )
                    update_live_status(
                        output_dir,
                        {
                            "job_id": job.job_id,
                            "status": "running",
                            "attempt": attempt,
                            "updated_at": now_iso(),
                        },
                    )

                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--job-dir",
                        str(job_dir),
                        "--heartbeat-path",
                        str(heartbeat_path),
                        "--suite",
                        job.suite,
                        "--scenario",
                        job.scenario,
                        "--policy",
                        job.policy,
                        "--episode",
                        str(job.episode),
                        "--seed",
                        str(job.seed),
                        "--order",
                        str(job.order),
                        "--attempt",
                        str(attempt),
                        "--model",
                        str(args.model),
                        "--step-hz",
                        str(args.step_hz),
                        "--grpc-base-port",
                        str(args.grpc_base_port),
                        "--speed-factor",
                        str(args.speed_factor),
                        "--battery-drain-rate",
                        str(args.battery_drain_rate),
                        "--log-level",
                        args.log_level,
                    ]
                    env = os.environ.copy()
                    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))

                    attempt_dm: DockerManager | None = None
                    with attempt_log.open("w", encoding="utf-8") as log_handle:
                        try:
                            if not args.no_docker:
                                attempt_dm = DockerManager(log_dir=output_dir / "sitl_logs")
                                logger.info(
                                    "Starting fresh PX4 SITL container for job %s attempt %d",
                                    job.job_id,
                                    attempt,
                                )
                                attempt_dm.start()
                                asyncio.run(attempt_dm.wait_healthy())

                            process = subprocess.Popen(
                                cmd,
                                cwd=Path(__file__).resolve().parents[1],
                                stdout=log_handle,
                                stderr=subprocess.STDOUT,
                                text=True,
                                env=env,
                            )
                            exit_code, reason, duration_s = monitor_attempt(
                                process=process,
                                job=job,
                                attempt=attempt,
                                heartbeat_path=heartbeat_path,
                                output_dir=output_dir,
                                timeout_s=args.episode_timeout_min * 60.0,
                                heartbeat_timeout_s=args.heartbeat_timeout_s,
                                log_interval_s=args.monitor_interval_s,
                            )
                        finally:
                            if attempt_dm is not None:
                                logger.info(
                                    "Stopping PX4 SITL container for job %s attempt %d",
                                    job.job_id,
                                    attempt,
                                )
                                attempt_dm.stop()
                                logger.info(
                                    "Cooling down %.0fs before the next episode attempt",
                                    args.episode_cooldown_s,
                                )
                                time.sleep(args.episode_cooldown_s)

                    row = {
                        "timestamp": now_iso(),
                        "plan": PLAN_NAME,
                        "job_id": job.job_id,
                        "suite": job.suite,
                        "scenario": job.scenario,
                        "policy": job.policy,
                        "episode": job.episode,
                        "seed": job.seed,
                        "attempt": attempt,
                        "exit_code": exit_code,
                        "duration_s": round(duration_s, 1),
                        "status": "completed" if exit_code == 0 else "failed",
                        "reason": reason,
                        "log_path": str(attempt_log),
                    }
                    write_manifest_row(output_dir, row)

                    if exit_code == 0:
                        logger.info(
                            "Completed job %s attempt %d in %.1fm",
                            job.job_id,
                            attempt,
                            duration_s / 60.0,
                        )
                        write_job_status(
                            job_dir,
                            {
                                "job_id": job.job_id,
                                "status": "completed",
                                "attempt": attempt,
                                "completed_at": now_iso(),
                                "duration_s": round(duration_s, 1),
                                "log_path": str(attempt_log),
                            },
                        )
                        refresh_aggregate_outputs(output_dir)
                        completed_ids.add(job.job_id)
                        update_live_status(
                            output_dir,
                            {
                                "job_id": job.job_id,
                                "status": "completed",
                                "attempt": attempt,
                                "updated_at": now_iso(),
                            },
                        )
                        job_success = True
                        break

                    logger.error(
                        "Job %s attempt %d failed with exit=%d (%s)",
                        job.job_id,
                        attempt,
                        exit_code,
                        reason,
                    )
        except Exception as exc:
            logger.exception("Unexpected parent-side failure while running job %s: %s", job.job_id, exc)

        if not job_success:
            logger.error("Abandoning job %s after 3 failed attempts", job.job_id)
            write_manifest_row(
                output_dir,
                {
                    "timestamp": now_iso(),
                    "plan": PLAN_NAME,
                    "job_id": job.job_id,
                    "suite": job.suite,
                    "scenario": job.scenario,
                    "policy": job.policy,
                    "episode": job.episode,
                    "seed": job.seed,
                    "attempt": 3,
                    "exit_code": "",
                    "duration_s": "",
                    "status": "abandoned",
                    "reason": "3 failed attempts",
                    "log_path": "",
                },
            )
            write_job_status(
                job_dir,
                {
                    "job_id": job.job_id,
                    "status": "abandoned",
                    "attempt": 3,
                    "updated_at": now_iso(),
                },
            )
            update_live_status(
                output_dir,
                {
                    "job_id": job.job_id,
                    "status": "abandoned",
                    "updated_at": now_iso(),
                },
            )
    refresh_aggregate_outputs(output_dir)
    logger.info("Overnight plan finished")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    default_output = Path("results") / f"overnight_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser = argparse.ArgumentParser(description="Run a 12-hour PX4 SITL paper validation plan")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--model", type=Path, default=Path("models/agent_marl9.pth"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-docker", action="store_true", help="Assume SITL is already running")
    parser.add_argument("--step-hz", type=float, default=4.0)
    parser.add_argument("--grpc-base-port", type=int, default=50051)
    parser.add_argument("--speed-factor", type=float, default=3.0)
    parser.add_argument("--battery-drain-rate", type=float, default=180.0)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--episode-timeout-min", type=float, default=45.0)
    parser.add_argument("--episode-cooldown-s", type=float, default=EPISODE_COOLDOWN_S)
    parser.add_argument("--heartbeat-timeout-s", type=float, default=300.0)
    parser.add_argument("--monitor-interval-s", type=float, default=60.0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser.add_argument("--job-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--heartbeat-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--suite", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scenario", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--policy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--episode", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--order", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=int, default=1, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.worker:
        if None in (args.job_dir, args.heartbeat_path, args.suite, args.scenario, args.policy, args.episode, args.order):
            raise SystemExit("Missing required worker arguments")
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
