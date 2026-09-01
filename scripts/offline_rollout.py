#!/usr/bin/env python3
"""Offline sanity rollout: run the CEDA-FGCS policy in the true world (no PX4).

Drives the collaborator's own training Environment via TrueWorld — the same
world the SITL experiments use — and gives a behavioral smoke signal
(deliveries, triage efficiency, landing phase, terminal landings).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from px4med.baselines import make_baseline
from px4med.fgcs_policy import FGCSPolicy
from px4med.true_world import TrueWorld


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--policy", default="learned",
        choices=["learned", "priority_path", "nearest_path", "random"],
    )
    args = parser.parse_args()

    random.seed(args.seed)
    learned = FGCSPolicy(device=args.device) if args.policy == "learned" else None
    if learned is not None:
        print(f"Policy loaded on {learned.device}; actions={learned.action_names}")

    for ep in range(args.episodes):
        world = TrueWorld({"mission": {"max_steps": args.max_steps}})
        world.reset()
        policy = learned if learned is not None else make_baseline(args.policy, args.seed + ep, world)
        total_reward = 0.0
        action_counts = [0] * 6
        landing_phase_step = None
        step = 0
        for step in range(args.max_steps):
            actions = policy.select_actions(world.observation())
            for a in actions:
                action_counts[a] += 1
            data = world.step(actions)
            total_reward += sum(data["rewards"])
            if landing_phase_step is None and world.landing_phase():
                landing_phase_step = step
            if not args.quiet and step % 50 == 0:
                print(
                    f"  step {step:3d} | actions={actions} pos={world.agent_grids} "
                    f"pending={world.pending_count()} delivered="
                    f"{sum(1 for p in world.patients if p.delivered)} "
                    f"landed={world.landed} depleted={world.depleted}"
                )
            if data["done"]:
                break

        delivered = sum(1 for p in world.patients if p.delivered)
        died = sum(1 for p in world.patients if p.died)
        spawned = sum(1 for p in world.patients if p.active)
        triage = world.triage_summary()
        print(
            f"Episode {ep}: steps={step + 1} spawned={spawned} delivered={delivered} "
            f"died={died} triage_eff={triage['triage_efficiency']:.3f} "
            f"landed={sum(world.landed)}/{world.num_drones} "
            f"depleted={sum(world.depleted)} reward={total_reward:.1f} "
            f"landing_phase_at={landing_phase_step} "
            f"action_hist(N,S,W,E,H,L)={action_counts}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
