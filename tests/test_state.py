"""Verifies state vector construction matches training env output given mock telemetry.

Tests are ordered from simplest to most complete:
  1. Length / slice offsets
  2. Individual feature groups with known values
  3. Full cross-validation against AneeshMARL5.Environment (obstacles + zones cleared)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Make both src/ and repo root importable
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

from px4med.drone import Telemetry
from px4med.environment import (
    MAX_PATIENT_TIMER,
    MAX_PATIENT_WEIGHT,
    WorldEnvironment,
)
from px4med.state import METERS_PER_CELL, build_state

# ── helpers ──────────────────────────────────────────────────────────────────

_CFG = {"grid": {"size": 50, "meters_per_cell": 2.0}}


def _world(wind: set | None = None, ls: set | None = None) -> WorldEnvironment:
    w = WorldEnvironment(_CFG)
    w.reset()
    w.wind_zones = wind if wind is not None else set()
    w.low_signal_zones = ls if ls is not None else set()
    return w


def _telem(gx: float, gy: float, battery: float = 100.0, landed: bool = False) -> Telemetry:
    """Build Telemetry from integer grid coords so NED ↔ grid round-trips exactly."""
    return Telemetry(
        north_m=-gy * METERS_PER_CELL,
        east_m=gx * METERS_PER_CELL,
        down_m=-5.0,
        battery_pct=battery,
        is_landed=landed,
    )


# ── length / slice offsets ────────────────────────────────────────────────────

def test_state_vector_length():
    s = build_state(0, _telem(1, 1), _telem(1, 13), _world())
    assert len(s) == 140


def test_slice_offsets():
    """Verify the section boundaries are correct by checking known-zero regions."""
    w = _world()
    s = build_state(0, _telem(25, 25), _telem(25, 1), w)
    # agent features: 9 floats
    assert len(s[0:9]) == 9
    # patient features: 8 × 7 = 56 floats
    assert len(s[9:65]) == 56
    # each local grid: 25 floats
    assert len(s[65:90]) == 25   # obs
    assert len(s[90:115]) == 25  # wind
    assert len(s[115:140]) == 25  # low-signal


# ── agent feature group [0:9] ─────────────────────────────────────────────────

def test_agent_id_zero():
    s = build_state(0, _telem(5, 10), _telem(20, 20), _world())
    assert s[0] == pytest.approx(0.0)


def test_agent_id_one():
    s = build_state(1, _telem(5, 10), _telem(20, 20), _world())
    assert s[0] == pytest.approx(1.0)


def test_position_normalised():
    s = build_state(0, _telem(10, 20), _telem(30, 30), _world())
    assert s[1] == pytest.approx(10 / 50)   # x_norm
    assert s[2] == pytest.approx(20 / 50)   # y_norm


def test_battery_normalised():
    s = build_state(0, _telem(5, 5, battery=60.0), _telem(10, 10), _world())
    assert s[3] == pytest.approx(60.0 / 100.0)


def test_landed_flag():
    s_flying = build_state(0, _telem(5, 5, landed=False), _telem(1, 1), _world())
    s_landed = build_state(0, _telem(5, 5, landed=True), _telem(1, 1), _world())
    assert s_flying[4] == pytest.approx(0.0)
    assert s_landed[4] == pytest.approx(1.0)


def test_other_relative_offset():
    s = build_state(0, _telem(10, 20), _telem(14, 25), _world())
    assert s[5] == pytest.approx((14 - 10) / 50)  # other_dx
    assert s[6] == pytest.approx((25 - 20) / 50)  # other_dy


def test_landing_zone_direction():
    """Agent pointing straight east toward its landing zone."""
    w = _world()
    # Override landing zone so agent is directly west of it
    w.landing_zones[0] = (0.0, 20.0)   # north=0, east=20
    # Agent at grid (5, 0) → north=0, east=10
    s = build_state(0, _telem(5, 0), _telem(1, 1), w)
    assert s[7] == pytest.approx(1.0, abs=1e-5)   # lz_dir_x = east
    assert s[8] == pytest.approx(0.0, abs=1e-5)


# ── patient feature group [9:65] ─────────────────────────────────────────────

def test_inactive_patient_all_zeros_except_weight():
    w = _world()
    w.patients[5].active = False
    w.patients[5].weight = 2
    s = build_state(0, _telem(1, 1), _telem(5, 5), w)
    base = 9 + 5 * 7
    assert s[base:base + 6] == pytest.approx([0.0] * 6, abs=1e-6)
    assert s[base + 6] == pytest.approx(2 / MAX_PATIENT_WEIGHT)


def test_delivered_patient_flag():
    w = _world()
    w.patients[0].active = True
    w.patients[0].delivered = True
    w.patients[0].weight = 3
    s = build_state(0, _telem(1, 1), _telem(5, 5), w)
    # delivered: [0, 0, 0, 0, 1.0, 0, weight_norm]
    assert s[9 + 4] == pytest.approx(1.0)   # delivered_flag
    assert s[9 + 0] == pytest.approx(0.0)   # gx_norm zero
    assert s[9 + 6] == pytest.approx(3 / MAX_PATIENT_WEIGHT)


def test_active_patient_position_and_timer():
    w = _world()
    p = w.patients[0]
    p.active = True
    p.delivered = False
    p.grid_x, p.grid_y = 13.0, 13.0
    p.timer = 125   # half of MAX_PATIENT_TIMER
    p.weight = 1
    s = build_state(0, _telem(1, 1), _telem(5, 5), w)
    assert s[9 + 0] == pytest.approx(13 / 50)          # gx_norm
    assert s[9 + 1] == pytest.approx(13 / 50)          # gy_norm
    assert s[9 + 4] == pytest.approx(0.0)               # not delivered
    assert s[9 + 5] == pytest.approx(125 / MAX_PATIENT_TIMER)
    assert s[9 + 6] == pytest.approx(1 / MAX_PATIENT_WEIGHT)


def test_active_patient_direction_vector_unit_length():
    w = _world()
    p = w.patients[0]
    p.active, p.delivered = True, False
    p.grid_x, p.grid_y = 10.0, 5.0
    s = build_state(0, _telem(1, 1), _telem(5, 5), w)
    dx, dy = s[9 + 2], s[9 + 3]
    assert math.sqrt(dx**2 + dy**2) == pytest.approx(1.0, abs=1e-5)


# ── local 5×5 grids [65:115] ──────────────────────────────────────────────────

def test_interior_obs_grid_all_zeros():
    """Interior agent position → no OOB cells → obs_grid all zero."""
    s = build_state(0, _telem(25, 25), _telem(1, 1), _world())
    assert s[65:90] == pytest.approx([0.0] * 25, abs=1e-6)


def test_oob_cells_appear_as_obstacles():
    """Agent at (1,1): cells with x<0 or y<0 must be 1.0 in obs_grid."""
    s = build_state(0, _telem(1, 1), _telem(5, 5), _world())
    obs = s[65:90]
    flat = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cx, cy = 1 + dx, 1 + dy
            expected = 1.0 if (cx < 0 or cy < 0 or cx >= 50 or cy >= 50) else 0.0
            assert obs[flat] == pytest.approx(expected), \
                f"obs[{flat}] at ({cx},{cy}): expected {expected}, got {obs[flat]}"
            flat += 1


def test_wind_zone_in_local_grid():
    """A wind zone one cell north of the agent (dy=-1, dx=0) → flat index 7."""
    w = _world(wind={(5, 4)})   # agent at grid (5,5); zone at (5,4) = dy=-1,dx=0
    s = build_state(0, _telem(5, 5), _telem(1, 1), w)
    wind = s[90:115]
    expected = [0.0] * 25
    expected[7] = 1.0           # (dy+2)*5 + (dx+2) = 1*5+2 = 7
    assert wind == pytest.approx(expected, abs=1e-6)


def test_low_signal_zone_in_local_grid():
    """Low-signal zone two cells east (dy=0, dx=2) → flat index 14."""
    w = _world(ls={(7, 5)})     # agent at (5,5); zone at (7,5) = dx=+2,dy=0
    s = build_state(0, _telem(5, 5), _telem(1, 1), w)
    ls = s[115:140]
    expected = [0.0] * 25
    expected[14] = 1.0          # (0+2)*5 + (2+2) = 2*5+4 = 14
    assert ls == pytest.approx(expected, abs=1e-6)


# ── cross-validation against AneeshMARL5.Environment ─────────────────────────

def test_state_matches_training_env():
    """Full round-trip: our build_state() must equal training get_state() (no obstacles/zones)."""
    from AneeshMARL5 import Environment as TrainingEnv

    tr = TrainingEnv(fixed_layout=True)
    tr.obstacles = set()
    tr.wind_zones = []
    tr.low_signal_zones = []

    config = {
        "grid": {"size": 50, "meters_per_cell": 2.0},
        "patients": [
            {"grid": list(pos), "weight": int(tr.patient_weights[i])}
            for i, pos in enumerate(tr.patient_positions)
        ],
        "landing_zones": [{"grid": list(lz)} for lz in tr.landing_zones],
    }
    w = WorldEnvironment(config)
    w.reset()
    # Mirror training env patient state exactly
    for i, p in enumerate(w.patients):
        p.active = tr.patient_active[i]
        p.delivered = tr.patients_delivered[i]
        p.timer = tr.patient_timers[i]
        p.weight = int(tr.patient_weights[i])
    w.wind_zones = set()
    w.low_signal_zones = set()

    for agent_idx in range(2):
        gx, gy = tr.agents[agent_idx]
        ogx, ogy = tr.agents[1 - agent_idx]

        tself = Telemetry(
            north_m=-gy * METERS_PER_CELL,
            east_m=gx * METERS_PER_CELL,
            down_m=-5.0,
            battery_pct=tr.batteries[agent_idx],
            is_landed=tr.landed[agent_idx],
        )
        tother = Telemetry(
            north_m=-ogy * METERS_PER_CELL,
            east_m=ogx * METERS_PER_CELL,
            down_m=-5.0,
            battery_pct=tr.batteries[1 - agent_idx],
            is_landed=tr.landed[1 - agent_idx],
        )

        tr_state = tr.get_state(agent_idx)
        our_state = build_state(agent_idx, tself, tother, w)

        assert len(our_state) == 140

        # Base features + patient features must be identical
        assert our_state[:65] == pytest.approx(tr_state[:65], abs=1e-5), \
            f"Agent {agent_idx}: base+patient mismatch\nours={our_state[:65]}\nref ={tr_state[:65]}"

        # obs_grid (65:90): training env has random obstacles, SITL has none — skip

        # wind + ls grids: both cleared to zero, must match
        assert our_state[90:] == pytest.approx(tr_state[90:], abs=1e-5), \
            f"Agent {agent_idx}: wind/ls grid mismatch"
