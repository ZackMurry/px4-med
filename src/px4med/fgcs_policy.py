"""Loads the CEDA-FGCS-PX4 policy package (models/CEDA-FGCS.py) for inference.

The module filename contains a hyphen, so it is loaded via importlib as the
model README prescribes. Only the shared local DQN is used; QMIX stays off.
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_DIR = _REPO_ROOT / "models"
DEFAULT_MODULE_PATH = DEFAULT_PACKAGE_DIR / "CEDA-FGCS.py"
DEFAULT_WEIGHTS_PATH = DEFAULT_PACKAGE_DIR / "ctde_agent_marl_FGCS.pth"


def load_ceda_module(module_path: Path = DEFAULT_MODULE_PATH):
    spec = spec_from_file_location("ceda_fgcs_px4", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CEDA-FGCS module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FGCSPolicy:
    """Thin wrapper exposing select_actions(observation) -> list[int]."""

    def __init__(
        self,
        weights_path: Path = DEFAULT_WEIGHTS_PATH,
        module_path: Path = DEFAULT_MODULE_PATH,
        device: str = "auto",
    ) -> None:
        self.module = load_ceda_module(module_path)
        self.policy = self.module.CEDAFGCSPX4Policy(
            weights_path, device=device, load_mixer=False
        )
        self.device = self.policy.device
        self.num_agents = self.module.NUM_AGENTS
        self.action_names = self.module.ACTION_NAMES

    def select_actions(self, observation: Mapping[str, object]) -> list[int]:
        return [int(a) for a in self.policy.select_actions(observation)]
