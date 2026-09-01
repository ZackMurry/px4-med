#!/usr/bin/env python3
"""Find episodes where a drone never actually flew.

There is an intermittent SITL fault (see HANDOFF.md §5b/§5c) where a drone
arms, reports airborne, passes the convergence gate — and then its telemetry
reads the map origin for the whole episode. The coordinator syncs the world
from that telemetry, so the env believes the drone is parked in a corner and
the fleet is silently one drone short. Such an episode is NOT a valid 5-drone
measurement: it also scores mission_success = 0 because the inert drone can
never reach its pad.

`tables/steps.csv` records every drone's grid cell at every step, so this is
detectable after the fact with no changes to the running suite.

A drone is called INERT when it occupies <= `--max-cells` distinct grid cells
for the whole episode (default 2 — one cell, plus one for boundary jitter).
Because a *legitimately* landed drone also stops moving, only cells visited
before the drone's last movement count; a drone that flew and then parked on
its pad is fine, while one pinned at the origin from step 1 is not.

Usage:
  poetry run python scripts/detect_inert_drones.py --run-dir results/core_...
  ... --json          machine-readable summary
  ... --print-rerun   emit the job_ids to re-run, one per line
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

ORIGIN_CELLS = {(0, 0), (0, 1), (1, 0), (1, 1)}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_positions(raw: str) -> list[tuple[int, int]]:
    cells = []
    for token in (raw or "").split(";"):
        if ":" not in token:
            continue
        x, y = token.split(":", 1)
        try:
            cells.append((int(x), int(y)))
        except ValueError:
            continue
    return cells


def analyse_episode(rows: list[dict[str, str]], max_cells: int) -> dict:
    rows = sorted(rows, key=lambda r: int(r["step"]))
    tracks: list[list[tuple[int, int]]] = []
    for row in rows:
        cells = parse_positions(row.get("positions", ""))
        if not tracks:
            tracks = [[] for _ in cells]
        for i, cell in enumerate(cells):
            if i < len(tracks):
                tracks[i].append(cell)

    drones = []
    for i, track in enumerate(tracks):
        if not track:
            drones.append({"drone": i, "inert": True, "reason": "no positions",
                           "distinct_cells": 0, "displacement": 0,
                           "start": None, "end": None})
            continue
        distinct = len(set(track))
        displacement = sum(
            abs(b[0] - a[0]) + abs(b[1] - a[1])
            for a, b in zip(track, track[1:])
        )
        # Steps until the drone last changed cell — a drone that flew and then
        # sat on its pad should not be judged on its parked tail.
        last_move = 0
        for step_idx in range(1, len(track)):
            if track[step_idx] != track[step_idx - 1]:
                last_move = step_idx
        inert = distinct <= max_cells
        reason = ""
        if inert:
            reason = (
                "pinned at map origin" if set(track) <= ORIGIN_CELLS
                else f"only {distinct} distinct cell(s)"
            )
        drones.append({
            "drone": i,
            "inert": inert,
            "reason": reason,
            "distinct_cells": distinct,
            "displacement": displacement,
            "start": list(track[0]),
            "end": list(track[-1]),
            "last_move_step": last_move,
            "steps": len(track),
        })
    return {"drones": drones,
            "inert_drones": [d["drone"] for d in drones if d["inert"]],
            "steps": len(rows)}


def job_id_for(plan: list[dict[str, str]], suite, scenario, policy, episode
               ) -> Optional[str]:
    for row in plan:
        if (row["suite"] == suite and row["scenario"] == scenario
                and row["policy"] == policy and row["episode"] == str(episode)):
            return row["job_id"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-cells", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--print-rerun", action="store_true")
    args = parser.parse_args()

    steps = read_csv(args.run_dir / "tables" / "steps.csv")
    if not steps:
        print(f"no steps.csv under {args.run_dir}/tables — nothing to check")
        return 1
    plan = read_csv(args.run_dir / "plan.csv")
    episodes = {
        (r["suite"], r["scenario"], r["policy"], r["episode"]): r
        for r in read_csv(args.run_dir / "tables" / "episodes.csv")
    }

    grouped: dict[tuple, list[dict[str, str]]] = {}
    for row in steps:
        key = (row["suite"], row["scenario"], row["policy"], row["episode"])
        grouped.setdefault(key, []).append(row)

    report, contaminated = [], []
    for key in sorted(grouped):
        suite, scenario, policy, episode = key
        analysis = analyse_episode(grouped[key], args.max_cells)
        job_id = job_id_for(plan, suite, scenario, policy, episode)
        episode_row = episodes.get(key)
        entry = {
            "job_id": job_id,
            "suite": suite, "scenario": scenario,
            "policy": policy, "episode": episode,
            "steps": analysis["steps"],
            "inert_drones": analysis["inert_drones"],
            "delivered": episode_row.get("patients_delivered") if episode_row else None,
            "triage_efficiency": episode_row.get("triage_efficiency") if episode_row else None,
            "drones_landed": episode_row.get("drones_landed") if episode_row else None,
            "detail": analysis["drones"],
        }
        report.append(entry)
        if analysis["inert_drones"]:
            contaminated.append(entry)

    if args.print_rerun:
        for entry in contaminated:
            if entry["job_id"]:
                print(entry["job_id"])
        return 0

    if args.json:
        print(json.dumps({"episodes": report,
                          "contaminated": [e["job_id"] for e in contaminated]},
                         indent=2))
        return 0

    print(f"Checked {len(report)} episode(s) in {args.run_dir.name}\n")
    header = (f"{'policy':14s} {'ep':>3s} {'steps':>6s} {'deliv':>6s} "
              f"{'landed':>6s}  inert drones")
    print(header)
    print("-" * len(header))
    for entry in report:
        inert = entry["inert_drones"]
        flag = ",".join(str(d) for d in inert) if inert else "-"
        print(f"{entry['policy']:14s} {entry['episode']:>3s} "
              f"{entry['steps']:>6d} {str(entry['delivered']):>6s} "
              f"{str(entry['drones_landed']):>6s}  {flag}")
        for drone in entry["detail"]:
            if drone["inert"]:
                print(f"{'':>14s}     drone {drone['drone']}: {drone['reason']}"
                      f" (start {drone['start']} end {drone['end']},"
                      f" displacement {drone['displacement']})")

    total = len(report)
    bad = len(contaminated)
    print()
    print(f"CONTAMINATED: {bad}/{total} episode(s)"
          + (f" ({100.0 * bad / total:.0f}%)" if total else ""))
    if contaminated:
        print("\nRe-run these job_ids (see --print-rerun for a bare list):")
        for entry in contaminated:
            print(f"  {entry['job_id']}  (inert: {entry['inert_drones']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
