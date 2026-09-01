"""Shape and semantics tests for the CEDA-FGCS-PX4 observation builder."""
from __future__ import annotations

import random

import numpy as np
import pytest

from px4med.environment import (
    ACTION_HOVER,
    ACTION_LAND,
    MAX_PATIENTS,
    WorldEnvironment,
)
from px4med.fgcs_state import build_observation


@pytest.fixture
def world() -> WorldEnvironment:
    random.seed(1234)
    w = WorldEnvironment({"mission": {"max_steps": 200}})
    w.reset()
    return w


def test_observation_shapes(world: WorldEnvironment) -> None:
    obs = build_observation(world)
    assert obs["drones"].shape == (5, 22)
    assert obs["patients"].shape == (MAX_PATIENTS, 10)
    assert obs["patient_masks"].shape == (MAX_PATIENTS,)
    assert obs["pending_patient_masks"].shape == (MAX_PATIENTS,)
    assert obs["local_grids"].shape == (5, 3, 21, 21)
    assert obs["mission"].shape == (12,)
    assert obs["action_masks"].shape == (5, 6)
    for key in ("drones", "patients", "local_grids", "mission"):
        assert obs[key].dtype == np.float32
        assert np.isfinite(obs[key]).all()
    for key in ("patient_masks", "pending_patient_masks", "action_masks"):
        assert obs[key].dtype == bool


def test_pending_masks_subset(world: WorldEnvironment) -> None:
    obs = build_observation(world)
    assert not (obs["pending_patient_masks"] & ~obs["patient_masks"]).any()
    assert obs["patient_masks"].sum() == world.initial_patients


def test_every_drone_has_valid_action(world: WorldEnvironment) -> None:
    obs = build_observation(world)
    assert obs["action_masks"].any(axis=-1).all()
    # In rescue phase (full battery, off-pad) land must be unavailable.
    assert not obs["action_masks"][:, ACTION_LAND].any()
    assert obs["action_masks"][:, :ACTION_LAND].all()


def test_inactive_patient_rows_zero(world: WorldEnvironment) -> None:
    obs = build_observation(world)
    inactive = ~obs["patient_masks"]
    assert (obs["patients"][inactive] == 0.0).all()


def test_landing_phase_masks(world: WorldEnvironment) -> None:
    # Resolve every patient and exhaust the spawn budget.
    world._spawned_count = world.spawn_total
    for p in world.patients:
        if p.active:
            p.delivered = True
    assert world.landing_phase()

    # Put drone 0 on its pad: land only. Drone 1 elsewhere: moves + hover.
    world.agent_grids[0] = world.landing_grid(0)
    world.agent_grids[1] = (5, 5)
    obs = build_observation(world)
    assert obs["action_masks"][0].tolist() == [False] * 5 + [True]
    assert obs["action_masks"][1].tolist() == [True] * 5 + [False]
    assert obs["mission"][8] == 1.0
    # Phase flag (drone feature 21) set for every live drone in landing phase.
    assert (obs["drones"][:, 21] == 1.0).all()


def test_energy_return_activates_land_on_pad(world: WorldEnvironment) -> None:
    # Not landing phase, but battery below the 20-point threshold on pad 0.
    assert not world.landing_phase()
    world.batteries[0] = 15.0
    world.agent_grids[0] = world.landing_grid(0)
    obs = build_observation(world)
    assert world.return_required(0)
    assert obs["action_masks"][0].tolist() == [False] * 5 + [True]
    assert obs["drones"][0, 21] == 1.0
    # A healthy drone off-pad keeps rescue-phase masks and phase flag 0.
    assert obs["drones"][1, 21] == 0.0


def test_landed_and_depleted_hover_only(world: WorldEnvironment) -> None:
    world.landed[2] = True
    world.depleted[3] = True
    obs = build_observation(world)
    hover_only = [False] * ACTION_HOVER + [True] + [False]
    assert obs["action_masks"][2].tolist() == hover_only
    assert obs["action_masks"][3].tolist() == hover_only
    assert obs["drones"][2, 3] == 1.0   # landed flag
    assert obs["drones"][3, 4] == 1.0   # depleted flag


def test_local_grid_boundary(world: WorldEnvironment) -> None:
    world.agent_grids[0] = (0, 0)
    obs = build_observation(world)
    grid = obs["local_grids"][0, 0]
    # Rows above (dy<0, i.e. north) and cols left (dx<0) of (0,0) are OOB.
    assert grid[:10, :].all() or grid[:10, :].sum() == 10 * 21  # dy=-10..-1 rows fully OOB
    assert grid[:, :10].all()


def test_step_advances_bookkeeping(world: WorldEnvironment) -> None:
    actions = [ACTION_HOVER] * 5
    data = world.step(actions)
    assert data["done"] is False
    assert world.prev_actions == actions
    obs = build_observation(world)
    assert (obs["drones"][:, 11 + ACTION_HOVER] == 1.0).all()
    # Ledger drained by the clean-step rate (no wind at start: zones empty
    # until first refresh interval elapses, but refresh may add wind), so
    # batteries must have decreased by at least the clean rate.
    assert all(
        b <= world.initial_battery - world.battery_drain_per_step + 1e-9
        for b in world.batteries
    )


def test_episode_continues_after_depletion(world: WorldEnvironment) -> None:
    world.batteries = [0.1] + [100.0] * 4
    data = world.step([0, ACTION_HOVER, ACTION_HOVER, ACTION_HOVER, ACTION_HOVER])
    assert world.depleted[0]
    assert data["done"] is False


def test_service_debt_fractions(world: WorldEnvironment) -> None:
    debt = world.service_debt_fractions()
    assert len(debt) == 3
    # Nothing delivered yet → full debt for every class that has spawned members.
    for frac in debt:
        assert 0.0 <= frac <= 1.0
    # Deliver every patient → debt goes to zero.
    for p in world.patients:
        if p.active:
            p.delivered = True
    assert world.service_debt_fractions() == [0.0, 0.0, 0.0]


def test_safe_return_margin_reasonable(world: WorldEnvironment) -> None:
    # Full battery near the start corner: margin should be positive but less
    # than full battery (route + 18 buffer must be subtracted).
    margin = world.safe_return_margin(0)
    assert margin < world.initial_battery
    # Start (2,2) → pad (97,97): ≥190 hops ≥ 38 energy + 18 buffer ≤ 44 margin
    assert margin < 50.0
    assert margin > -100.0
