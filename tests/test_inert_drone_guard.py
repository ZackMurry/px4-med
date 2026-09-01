"""The start-position guard must catch the inert-telemetry fault (HANDOFF §5c).

Regression test built from real data: in core_20260829_230535 job 1 attempt 2,
four drones settled on their assigned start cells exactly while drone 0 sat at
(0, 0), 109 cells from its target. The coordinator warned and continued, which
produced a silently 4-drone episode — 47/50 delivered but mission_success and
all_landed forced to 0. That data looks plausible and is not a valid 5-drone
measurement, so the attempt must fail and be retried instead.
"""
from __future__ import annotations

from px4med.coordinator import (
    _INERT_START_MAX_CELLS,
    InertDroneError,
    find_stranded_drones,
)

# Verbatim from the worker log of the observed failure.
OBSERVED_TARGETS = [(29, 80), (11, 49), (1, 89), (88, 5), (73, 75)]
OBSERVED_ACTUAL = [(0, 0), (11, 49), (1, 89), (88, 5), (73, 75)]


def test_catches_the_observed_fault():
    stranded = find_stranded_drones(OBSERVED_ACTUAL, OBSERVED_TARGETS)
    assert stranded == [(0, 109)]


def test_exact_settle_is_clean():
    assert find_stranded_drones(OBSERVED_TARGETS, OBSERVED_TARGETS) == []


def test_braking_overshoot_is_not_flagged():
    # Healthy drones settle on the exact cell; allow generous slack for a
    # drone still braking so ordinary overshoot never fails an attempt.
    overshot = [
        (target[0] + 3, target[1] - 2) for target in OBSERVED_TARGETS
    ]
    assert find_stranded_drones(overshot, OBSERVED_TARGETS) == []


def test_threshold_boundary_is_exclusive():
    target = [(50, 50)]
    at_limit = [(50 + _INERT_START_MAX_CELLS, 50)]
    just_over = [(50 + _INERT_START_MAX_CELLS + 1, 50)]
    assert find_stranded_drones(at_limit, target) == []
    assert find_stranded_drones(just_over, target) == [(0, _INERT_START_MAX_CELLS + 1)]


def test_reports_every_offender():
    targets = [(50, 50), (10, 10), (90, 90)]
    actual = [(0, 0), (10, 10), (0, 0)]
    assert find_stranded_drones(actual, targets) == [(0, 100), (2, 180)]


def test_error_type_is_distinguishable():
    # The worker logs the exception; a dedicated type keeps this failure mode
    # greppable in worker logs and distinct from arm/telemetry errors.
    assert issubclass(InertDroneError, RuntimeError)
