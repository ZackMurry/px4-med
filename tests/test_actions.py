"""Verifies action→waypoint mapping is correct and reversible."""
from px4med.actions import action_to_offset, is_land_action, STEP_M
import pytest


def test_move_actions_produce_offsets():
    n = action_to_offset(0)
    assert n.d_north == STEP_M and n.d_east == 0.0

    s = action_to_offset(1)
    assert s.d_north == -STEP_M and s.d_east == 0.0

    w = action_to_offset(2)
    assert w.d_east == -STEP_M and w.d_north == 0.0

    e = action_to_offset(3)
    assert e.d_east == STEP_M and e.d_north == 0.0


def test_land_action_raises():
    with pytest.raises(ValueError):
        action_to_offset(4)


def test_is_land_action():
    assert is_land_action(4)
    assert not is_land_action(0)
