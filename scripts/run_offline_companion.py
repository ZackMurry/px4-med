#!/usr/bin/env python3
"""Run the offline twin of a SITL suite: same jobs, same seeds, no PX4.

The sim-transfer claim is only clean if both sides see the same worlds. This
reads a completed (or partial) SITL run's `plan.csv` and replays every job
through `run_offline_episode` with the identical (suite, scenario, policy,
seed), writing a parallel run directory with the same `tables/` layout so
`make_paper_figures.py --offline-dir` can pair them.

Cheap: a full 24-job plan is a few minutes of CPU. Do NOT run it while a SITL
fleet is booting — EKF convergence starves (see HANDOFF.md §5).

Usage:
  poetry run python scripts/run_offline_companion.py \
      --sitl-dir results/core_20260829_230535 [--out-dir results/offline_...]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from px4med.experiments import (  # noqa: E402
    FGCSPolicy,
    suite_lookup,
    run_offline_episode,
    summarize_results,
    write_csv,
    write_episode_csv,
    write_step_csv,
)

logger = logging.getLogger("offline_companion")


def read_plan(sitl_dir: Path) -> list[dict[str, str]]:
    plan_path = sitl_dir / "plan.csv"
    if not plan_path.exists():
        raise SystemExit(f"No plan.csv in {sitl_dir}")
    with plan_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sitl-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--only-completed", action="store_true",
                        help="Replay only jobs the SITL run actually finished, "
                             "so the paired comparison uses identical worlds")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    plan = read_plan(args.sitl_dir)
    if args.only_completed:
        done = set()
        jobs_dir = args.sitl_dir / "jobs"
        if jobs_dir.exists():
            for job_dir in jobs_dir.iterdir():
                status = job_dir / "status.json"
                if status.exists():
                    payload = json.loads(status.read_text())
                    if payload.get("status") == "completed":
                        done.add(payload["job_id"])
        before = len(plan)
        plan = [row for row in plan if row["job_id"] in done]
        logger.info("Filtered to %d of %d jobs completed in SITL",
                    len(plan), before)

    out_dir = args.out_dir or (
        args.sitl_dir.parent / f"offline_twin_{args.sitl_dir.name}"
    )
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    lookup = suite_lookup()
    # One policy load for the whole sweep — the checkpoint is 29 MB.
    learned = FGCSPolicy(weights_path=args.model) if any(
        row["policy"] == "learned" for row in plan) else None

    results, all_steps = [], []
    start = time.time()
    for i, row in enumerate(plan, 1):
        key = (row["suite"], row["scenario"])
        if key not in lookup:
            logger.warning("Skipping unknown suite/scenario %s", key)
            continue
        suite, scenario = lookup[key]
        seed = int(row["seed"])
        episode = int(row.get("episode", 0) or 0)
        result, steps = run_offline_episode(
            suite, scenario, row["policy"], seed=seed, episode_idx=episode,
            learned_policy=learned if row["policy"] == "learned" else None,
            model_path=args.model,
        )
        results.append(result)
        all_steps.extend(steps)
        logger.info(
            "[%2d/%2d] %-14s seed=%-11d deliv=%2d triage=%.3f succ=%.0f "
            "landed=%d wind_avoid=%.3f (%d opp)",
            i, len(plan), row["policy"], seed, result.patients_delivered,
            result.triage_efficiency, result.mission_success,
            result.drones_landed, result.wind_avoidance_rate,
            result.wind_avoidance_opportunities,
        )
        # Incremental, so an interrupted sweep still leaves usable tables.
        write_episode_csv(out_dir / "tables" / "episodes.csv", results)
        write_csv(out_dir / "tables" / "summary.csv", summarize_results(results))

    if all_steps:
        write_step_csv(out_dir / "tables" / "steps.csv", all_steps)
    (out_dir / "provenance.json").write_text(json.dumps({
        "paired_with": str(args.sitl_dir),
        "jobs": len(results),
        "only_completed": args.only_completed,
        "duration_s": round(time.time() - start, 1),
    }, indent=2))

    logger.info("Wrote %d offline episodes to %s in %.1f min",
                len(results), out_dir, (time.time() - start) / 60)
    for row in summarize_results(results):
        logger.info(
            "  %-14s n=%s deliv=%.3f triage=%.3f succ=%.2f",
            row["policy"], row["episodes"],
            float(row["delivery_rate_mean"]),
            float(row["triage_efficiency_mean"]),
            float(row["mission_success_mean"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
