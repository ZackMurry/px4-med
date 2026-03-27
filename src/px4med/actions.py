"""Maps discrete action integers to MAVSDK NED position offsets.

Action space (mirrors AneeshMARL5.py Environment.step()):
  0  up    → grid y - 1 → NED north + STEP_M
  1  down  → grid y + 1 → NED north - STEP_M
  2  left  → grid x - 1 → NED east  - STEP_M
  3  right → grid x + 1 → NED east  + STEP_M
  4  land  → call drone.land() (not a positional offset)
"""
from __future__ import annotations

from dataclasses import dataclass

STEP_M: float = 2.0       # 1 grid cell = 2 m
CRUISE_DOWN_M: float = -5.0  # NED down (negative = above ground, 5 m AGL)


@dataclass(frozen=True)
class WaypointOffset:
    """Delta in NED metres to apply to the drone's current position."""
    d_north: float
    d_east: float
    d_down: float = 0.0


_ACTION_MAP: dict[int, WaypointOffset] = {
    0: WaypointOffset(+STEP_M,  0.0),   # up    → north
    1: WaypointOffset(-STEP_M,  0.0),   # down  → south
    2: WaypointOffset(0.0, -STEP_M),    # left  → west
    3: WaypointOffset(0.0, +STEP_M),    # right → east
}


def action_to_offset(action: int) -> WaypointOffset:
    """Return NED delta for move actions 0–3.

    Raises ValueError for action 4 (land) — call drone.land() directly.
    """
    if action == 4:
        raise ValueError("Action 4 is land — call drone.land() instead of action_to_offset()")
    if action not in _ACTION_MAP:
        raise ValueError(f"Unknown action: {action!r} (expected 0–4)")
    return _ACTION_MAP[action]


def is_land_action(action: int) -> bool:
    return action == 4
