"""Loads the trained CTDE .pth model and runs greedy inference. No training logic."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class CentralQNet(nn.Module):
    """Mirror of AneeshMARL5.py CentralQNet — must stay in sync with training code."""

    JOINT_STATE_DIM = 280  # 140 floats per agent × 2 agents
    ACTION_DIM = 5

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.JOINT_STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.ACTION_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PolicyNet:
    """Thin inference wrapper around CentralQNet."""

    def __init__(self, model_path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CentralQNet().to(self.device)
        self.net.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.net.eval()

    def select_action(self, state_self: list[float], state_other: list[float]) -> int:
        """Return greedy action (0–4) for one agent given both agents' state vectors."""
        joint = torch.FloatTensor(state_self + state_other).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.net(joint).argmax().item())
