"""Builds per-agent 140-float state vectors from live MAVSDK telemetry.

Layout (must exactly match AneeshMARL5.py Environment.get_state()):
  [0]       agent_id               (0.0 or 1.0)
  [1]       x_norm                 (grid_x / GRID_SIZE)
  [2]       y_norm                 (grid_y / GRID_SIZE)
  [3]       battery_norm           (sim_battery_pct / MAX_BATTERY)
  [4]       landed_flag            (1.0 if virtually landed else 0.0)
  [5]       other_dx               ((other_x - x) / GRID_SIZE)
  [6]       other_dy               ((other_y - y) / GRID_SIZE)
  [7]       lz_dir_x               (unit vector x toward landing zone)
  [8]       lz_dir_y               (unit vector y toward landing zone)
  [9:65]    patient features       8 patients × 7 floats
              [gx_norm, gy_norm, dir_x, dir_y, delivered_flag, timer_norm, weight_norm]
  [65:90]   5×5 local obstacle grid   (row-major dy∈[-2,2], dx∈[-2,2])
  [90:115]  5×5 local wind grid
  [115:140] 5×5 local low-signal grid
  Total: 140 floats; joint input to CentralQNet = 280 (self ‖ other)
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .drone import Telemetry
    from .environment import WorldEnvironment

GRID_SIZE = 50
MAX_BATTERY = 100.0
MAX_PATIENT_TIMER = 250
MAX_PATIENT_WEIGHT = 3
METERS_PER_CELL = 2.0   # 1 grid cell = 2 m in NED space


def ned_to_grid(north_m: float, east_m: float) -> tuple[float, float]:
    """Convert NED metres (origin = PX4 home) to fractional grid coordinates.

    Mapping chosen to match training env action semantics:
      grid x = east_m / METERS_PER_CELL   (action 3 = right = +x = +east)
      grid y = -north_m / METERS_PER_CELL (action 0 = up   = -y = +north)
    """
    return east_m / METERS_PER_CELL, -north_m / METERS_PER_CELL


def _direction_vector(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
    """Unit vector from (x0, y0) toward (x1, y1) in grid space. Returns (0,0) if coincident."""
    dx, dy = x1 - x0, y1 - y0
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > 0:
        return dx / dist, dy / dist
    return 0.0, 0.0


def build_state(
    agent_idx: int,
    telem_self: "Telemetry",
    telem_other: "Telemetry",
    world: "WorldEnvironment",
) -> list[float]:
    """Return 140-float state vector matching AneeshMARL5.py Environment.get_state()."""
    grid_size = world.config.get("grid", {}).get("size", GRID_SIZE)

    # ── grid positions ────────────────────────────────────────────────────────
    # Continuous (for normalised features) and rounded (for 5×5 local grids)
    x_f, y_f = ned_to_grid(telem_self.north_m, telem_self.east_m)
    ox_f, oy_f = ned_to_grid(telem_other.north_m, telem_other.east_m)
    x_i, y_i = round(x_f), round(y_f)

    # ── base features [0:9] ───────────────────────────────────────────────────
    lz_north, lz_east = world.landing_zones[agent_idx]
    lz_gx, lz_gy = ned_to_grid(lz_north, lz_east)
    lz_dir_x, lz_dir_y = _direction_vector(x_f, y_f, lz_gx, lz_gy)

    base: list[float] = [
        float(agent_idx),
        x_f / grid_size,
        y_f / grid_size,
        world.batteries[agent_idx] / MAX_BATTERY,
        1.0 if world.landed[agent_idx] else 0.0,
        (ox_f - x_f) / grid_size,
        (oy_f - y_f) / grid_size,
        lz_dir_x,
        lz_dir_y,
    ]

    # ── patient features [9:65] — 8 patients × 7 floats ──────────────────────
    patient_features: list[float] = []
    for p in world.patients:
        w_norm = p.weight / MAX_PATIENT_WEIGHT
        if not p.active:
            patient_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, w_norm])
        elif p.delivered:
            patient_features.extend([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, w_norm])
        else:
            dir_x, dir_y = _direction_vector(x_f, y_f, p.grid_x, p.grid_y)
            patient_features.extend([
                p.grid_x / grid_size,
                p.grid_y / grid_size,
                dir_x,
                dir_y,
                0.0,
                p.timer / MAX_PATIENT_TIMER,
                w_norm,
            ])

    # ── 5×5 local grids [65:90 / 90:115 / 115:140] ───────────────────────────
    obs_grid: list[float] = []
    wind_grid: list[float] = []
    ls_grid: list[float] = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cx, cy = x_i + dx, y_i + dy
            oob = cx < 0 or cx >= grid_size or cy < 0 or cy >= grid_size
            obs_grid.append(1.0 if oob or (cx, cy) in world.obstacles else 0.0)
            wind_grid.append(1.0 if not oob and (cx, cy) in world.wind_zones else 0.0)
            ls_grid.append(1.0 if not oob and (cx, cy) in world.low_signal_zones else 0.0)

    state = base + patient_features + obs_grid + wind_grid + ls_grid
    assert len(state) == 140, f"State length {len(state)} != 140"
    return state
