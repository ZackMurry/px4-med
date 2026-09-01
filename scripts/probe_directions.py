#!/usr/bin/env python3
"""Directional sanity probes for the observation builder ↔ policy pairing.

Constructs minimal worlds (no hazards, no obstacles) with a single pending
patient placed N/S/W/E of drone 0 and checks the policy's greedy action moves
toward it. Also probes the landing phase: drone adjacent to its pad should
move toward the pad, and on the pad should land.
Failures indicate a coordinate/orientation mismatch, not a weak policy.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from px4med.environment import WorldEnvironment
from px4med.fgcs_policy import FGCSPolicy
from px4med.fgcs_state import build_observation

ACTION_NAMES = ("north", "south", "west", "east", "hover", "land")


def make_world(**mission) -> WorldEnvironment:
    config = {
        "num_obstacles": 0,
        "hazards": {"num_wind_zones": 0, "num_low_signal_zones": 0},
        "mission": {"initial_patients": 1, "spawn_total": 1, "max_steps": 500, **mission},
        "agent_start_positions": [[50, 50], [10, 10], [10, 90], [90, 10], [20, 50]],
        "patients": [{"grid": [50, 40], "weight": 3}],
    }
    w = WorldEnvironment(config)
    w.reset()
    return w


def probe(policy: FGCSPolicy) -> int:
    failures = 0

    # ── patient-direction probes for drone 0 at (50,50) ──────────────────────
    cases = {
        "north": (50, 40),
        "south": (50, 60),
        "west": (40, 50),
        "east": (60, 50),
    }
    for expect, (px, py) in cases.items():
        w = make_world()
        p = w.patients[0]
        p.grid_x, p.grid_y = float(px), float(py)
        action = policy.select_actions(build_observation(w))[0]
        verdict = "OK " if ACTION_NAMES[action] == expect else "FAIL"
        if ACTION_NAMES[action] != expect:
            failures += 1
        print(f"[{verdict}] patient {expect} of drone → action {ACTION_NAMES[action]}")

    # ── landing-phase probes ──────────────────────────────────────────────────
    w = make_world()
    w.patients[0].delivered = True
    w._spawned_count = w.spawn_total
    assert w.landing_phase()
    pad = w.landing_grid(0)

    # On pad: mask forces land.
    w.agent_grids[0] = pad
    action = policy.select_actions(build_observation(w))[0]
    verdict = "OK " if action == 5 else "FAIL"
    if action != 5:
        failures += 1
    print(f"[{verdict}] landing phase on pad → action {ACTION_NAMES[action]}")

    # Adjacent to pad (west of it): should move east toward the pad.
    w.agent_grids[0] = (pad[0] - 3, pad[1])
    action = policy.select_actions(build_observation(w))[0]
    verdict = "OK " if ACTION_NAMES[action] == "east" else "FAIL"
    if ACTION_NAMES[action] != "east":
        failures += 1
    print(f"[{verdict}] landing phase, pad 3 east → action {ACTION_NAMES[action]}")

    # Adjacent to pad (north of it): should move south toward the pad.
    w.agent_grids[0] = (pad[0], pad[1] - 3)
    action = policy.select_actions(build_observation(w))[0]
    verdict = "OK " if ACTION_NAMES[action] == "south" else "FAIL"
    if ACTION_NAMES[action] != "south":
        failures += 1
    print(f"[{verdict}] landing phase, pad 3 south → action {ACTION_NAMES[action]}")

    return failures


def main() -> int:
    policy = FGCSPolicy(device="cpu")
    failures = probe(policy)
    print(f"\n{'ALL PROBES PASSED' if failures == 0 else f'{failures} PROBES FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
