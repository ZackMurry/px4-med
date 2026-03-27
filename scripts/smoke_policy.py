#!/usr/bin/env python3
"""Level-2 smoke test: verify CentralQNet architecture without a .pth file.

Run from the repo root:
    python scripts/smoke_policy.py

Or with an actual model file:
    python scripts/smoke_policy.py --model models/ctde_agent_marl9.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test CentralQNet forward pass")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to .pth file (omit to test with random weights)",
    )
    args = parser.parse_args()

    import torch
    from px4med.policy import CentralQNet, PolicyNet

    if args.model:
        print(f"Loading weights from {args.model} …")
        policy = PolicyNet(args.model)
        net = policy.net
        print(f"  device : {policy.device}")
    else:
        print("No model file given — testing with random weights")
        net = CentralQNet()
        net.eval()

    # Joint state: 140 floats per agent × 2
    dummy = torch.zeros(1, CentralQNet.JOINT_STATE_DIM)
    with torch.no_grad():
        out = net(dummy)

    print(f"  input  : {list(dummy.shape)}  (joint state = 140×2)")
    print(f"  output : {list(out.shape)}    (Q-values for {CentralQNet.ACTION_DIM} actions)")
    print(f"  Q-vals : {out[0].tolist()}")
    print(f"  action : {out.argmax().item()}  (greedy)")

    if args.model:
        # Also exercise PolicyNet.select_action for both agents
        state = [0.0] * 140
        a = policy.select_action(state, state)
        print(f"  PolicyNet.select_action → {a}")

    print("OK")


if __name__ == "__main__":
    main()
