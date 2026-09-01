"""Hazard/energy counters must mirror the training script's accumulation.

The env emits these per step and never accumulates them; only
`TrainingEpisodeTracker` does, and it defines the rates as
`1 - selections / max(1, opportunities)`. If our accumulation drifts from
that, our avoidance numbers stop being comparable to the curriculum gates
(stage 2 requires >= 0.98 wind and low-signal avoidance).
"""
from __future__ import annotations

from px4med.true_world import (
    HAZARD_KEYS,
    accumulate_hazard,
    hazard_fields,
    new_hazard_totals,
)


def test_scalar_and_per_agent_counters_accumulate():
    totals = new_hazard_totals()
    accumulate_hazard(totals, {
        "wind_avoidance_opportunities": 3,
        "wind_hazard_selections": 1,
        "wind_exposure_steps": [1, 0, 0, 2, 0],   # per-agent list -> summed
        "reserve_violation_flags": [1, 1, 0, 0, 0],
    })
    accumulate_hazard(totals, {
        "wind_avoidance_opportunities": 1,
        "wind_exposure_steps": [0, 0, 1, 0, 0],
    })
    fields = hazard_fields(totals)
    assert fields["wind_avoidance_opportunities"] == 4
    assert fields["wind_hazard_selections"] == 1
    assert fields["wind_avoidance_rate"] == 0.75      # 1 - 1/4
    assert fields["wind_exposure_steps"] == 4
    assert fields["reserve_violations"] == 2


def test_rate_is_vacuous_one_without_opportunities():
    # Mirrors the training formula's max(1, ...) guard: no opportunities scores
    # 1.0, which is why a rate must always be read next to its denominator.
    fields = hazard_fields(new_hazard_totals())
    assert fields["wind_avoidance_rate"] == 1.0
    assert fields["low_signal_avoidance_rate"] == 1.0
    assert fields["wind_avoidance_opportunities"] == 0


def test_missing_and_malformed_step_data_is_ignored():
    totals = new_hazard_totals()
    accumulate_hazard(totals, None)
    accumulate_hazard(totals, {})
    accumulate_hazard(totals, {"wind_avoidance_opportunities": None})
    assert all(v == 0 for v in totals.values())


def test_every_declared_key_is_reported():
    # Guards against adding a key to HAZARD_KEYS without surfacing it.
    totals = {key: 1 for key in HAZARD_KEYS}
    fields = hazard_fields(totals)
    assert fields["wind_exposure_steps"] == 1
    assert fields["forced_terminal_landings"] == 1
    assert fields["low_signal_movement_failures"] == 1
    # 1 - 1/1 == 0.0 when every counter is 1
    assert fields["wind_avoidance_rate"] == 0.0


def test_episode_result_exposes_hazard_columns():
    import dataclasses

    from px4med.experiments import EpisodeResult

    names = {f.name for f in dataclasses.fields(EpisodeResult)}
    for key in hazard_fields(None):
        assert key in names, f"{key} missing from EpisodeResult"
