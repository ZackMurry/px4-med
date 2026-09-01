#!/usr/bin/env python3
"""Large-n offline sweep — tighten the offline side of any comparison.

Offline episodes cost seconds, so the abstract-env numbers can have far more
episodes than the SITL side. With n=50 per policy the offline CI becomes
negligible next to the SITL CI, which means the offline-vs-SITL transfer gap is
attributable to SITL variance alone rather than to noise on both sides.

Independent of `run_offline_companion.py`: that one mirrors a specific SITL
run's plan.csv at matched seeds (for paired comparison), whereas this sweeps
arbitrary (suite, scenario, policy) cells at whatever n you ask for.

CPU-only but not free — `nearest_path` runs ~75 s/episode because of its
safe-return Dijkstra. Do NOT run this while a SITL fleet is booting
(HANDOFF.md §5): EKF convergence starves.

Usage:
  poetry run python scripts/run_offline_sweep.py --episodes 50 \
      --suite baseline_comparison --scenario nominal
  poetry run python scripts/run_offline_sweep.py --episodes 20 \
      --suite hazard_sweep --scenario all --policies learned,nearest_path
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from px4med.experiments import (  # noqa: E402
    FGCSPolicy,
    build_default_suites,
    run_offline_episode,
    summarize_results,
    write_csv,
    write_episode_csv,
)

logger = logging.getLogger("offline_sweep")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--suite", default="baseline_comparison")
    parser.add_argument(
        "--scenario", default="nominal",
        help="scenario name, or 'all' for every scenario in the suite")
    parser.add_argument(
        "--policies", default=None,
        help="comma-separated; defaults to the suite's own policy list")
    parser.add_argument("--seed-base", type=int, default=900000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    suites = {s.name: s for s in build_default_suites()}
    if args.suite not in suites:
        raise SystemExit(f"Unknown suite {args.suite}; have {sorted(suites)}")
    suite = suites[args.suite]

    if args.scenario == "all":
        scenarios = list(suite.scenarios)
    else:
        scenarios = [s for s in suite.scenarios if s.name == args.scenario]
        if not scenarios:
            raise SystemExit(
                f"Unknown scenario {args.scenario} in {args.suite}; have "
                f"{[s.name for s in suite.scenarios]}"
            )

    policies = (
        [p.strip() for p in args.policies.split(",") if p.strip()]
        if args.policies else list(suite.policies)
    )

    out_dir = args.out_dir or (
        REPO_ROOT / "results"
        / f"offline_sweep_{args.suite}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    learned = FGCSPolicy(weights_path=args.model) if "learned" in policies else None

    results = []
    start = time.time()
    total = len(scenarios) * len(policies) * args.episodes
    done = 0
    for scenario in scenarios:
        for policy in policies:
            for episode in range(args.episodes):
                seed = args.seed_base + done
                result, _ = run_offline_episode(
                    suite, scenario, policy, seed=seed, episode_idx=episode,
                    learned_policy=learned if policy == "learned" else None,
                    model_path=args.model,
                )
                results.append(result)
                done += 1
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - start
                    logger.info(
                        "[%3d/%3d] %s/%s %-13s  elapsed %.1f min, eta %.1f min",
                        done, total, scenario.name, policy, "",
                        elapsed / 60,
                        (elapsed / done) * (total - done) / 60,
                    )
                    # Incremental write so an interrupted sweep is still usable
                    write_episode_csv(out_dir / "tables" / "episodes.csv", results)
                    write_csv(out_dir / "tables" / "summary.csv",
                              summarize_results(results))

    write_episode_csv(out_dir / "tables" / "episodes.csv", results)
    write_csv(out_dir / "tables" / "summary.csv", summarize_results(results))
    (out_dir / "provenance.json").write_text(json.dumps({
        "backend": "offline",
        "suite": args.suite,
        "scenarios": [s.name for s in scenarios],
        "policies": policies,
        "episodes_per_cell": args.episodes,
        "seed_base": args.seed_base,
        "duration_s": round(time.time() - start, 1),
    }, indent=2))

    logger.info("Wrote %d episodes to %s in %.1f min",
                len(results), out_dir, (time.time() - start) / 60)
    for row in summarize_results(results):
        logger.info(
            "  %-18s %-13s n=%-3s deliv=%.3f +-%.3f  triage=%.3f +-%.3f  "
            "wind_avoid=%.3f",
            row["scenario"], row["policy"], row["episodes"],
            float(row["delivery_rate_mean"]), float(row["delivery_rate_ci95"]),
            float(row["triage_efficiency_mean"]),
            float(row["triage_efficiency_ci95"]),
            float(row["wind_avoidance_rate_mean"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
