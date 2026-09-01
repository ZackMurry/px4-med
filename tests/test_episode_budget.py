"""The episode budget must track the training config exactly.

Regression guard for a real bug: the experiment suites ran 500-step episodes
against a policy trained with MAX_STEPS=800. The mission observation encodes
`episode_step / current_episode_deadline()`, so the shorter budget silently
fed the policy a 1.6x-fast mission clock, and long SITL episodes were
truncated mid-landing.
"""
from __future__ import annotations

from px4med.episode_budget import (
    LANDING_GRACE_STEPS,
    LOOP_STEP_CAP,
    MISSION_MAX_STEPS,
)


def test_budget_matches_training_module():
    from px4med.true_world import load_training_module

    m = load_training_module()
    assert MISSION_MAX_STEPS == m.MAX_STEPS
    assert LANDING_GRACE_STEPS == m.DEFAULT_LANDING_GRACE_STEPS


def test_loop_cap_covers_worst_case_landing_deadline():
    # Worst case: patients resolve on the final mission step, so the env's
    # landing_deadline is MISSION_MAX_STEPS + LANDING_GRACE_STEPS.
    assert LOOP_STEP_CAP == MISSION_MAX_STEPS + LANDING_GRACE_STEPS


def test_suites_use_the_full_budget():
    from px4med.experiments import build_default_suites

    for suite in build_default_suites():
        for scenario in suite.scenarios:
            mission = scenario.world["mission"]
            assert mission["max_steps"] == MISSION_MAX_STEPS, suite.name
            assert mission["landing_grace_steps"] == LANDING_GRACE_STEPS, suite.name
            # The driver loop must outlast the env's own termination.
            assert scenario.max_steps == LOOP_STEP_CAP, suite.name
