"""Maps discrete action integers to MAVSDK NED position offsets.

Action space (CEDA-FGCS-PX4):
  0  north → grid y - 1 → NED north + STEP_M
  1  south → grid y + 1 → NED north - STEP_M
  2  west  → grid x - 1 → NED east  - STEP_M
  3  east  → grid x + 1 → NED east  + STEP_M
  4  hover → hold current position setpoint
  5  land  → call drone.land() (not a positional offset)
"""
from __future__ import annotations

from dataclasses import dataclass

STEP_M: float = 2.0       # 1 grid cell = 2 m
CRUISE_DOWN_M: float = -20.0  # NED down (negative = above ground, 20 m AGL)

ACTION_HOVER = 4
ACTION_LAND = 5


@dataclass(frozen=True)
class WaypointOffset:
    """Delta in NED metres to apply to the drone's current position."""
    d_north: float
    d_east: float
    d_down: float = 0.0


_ACTION_MAP: dict[int, WaypointOffset] = {
    0: WaypointOffset(+STEP_M,  0.0),   # north
    1: WaypointOffset(-STEP_M,  0.0),   # south
    2: WaypointOffset(0.0, -STEP_M),    # west
    3: WaypointOffset(0.0, +STEP_M),    # east
    4: WaypointOffset(0.0, 0.0),        # hover
}


def action_to_offset(action: int) -> WaypointOffset:
    """Return NED delta for move/hover actions 0–4.

    Raises ValueError for action 5 (land) — call drone.land() directly.
    """
    if action == ACTION_LAND:
        raise ValueError("Action 5 is land — call drone.land() instead of action_to_offset()")
    if action not in _ACTION_MAP:
        raise ValueError(f"Unknown action: {action!r} (expected 0–5)")
    return _ACTION_MAP[action]


def is_land_action(action: int) -> bool:
    return action == ACTION_LAND
