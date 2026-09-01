"""Verifies action→waypoint mapping for the 6-action CEDA-FGCS space."""
from px4med.actions import ACTION_HOVER, ACTION_LAND, action_to_offset, is_land_action, STEP_M
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


def test_hover_is_zero_offset():
    h = action_to_offset(ACTION_HOVER)
    assert h.d_north == 0.0 and h.d_east == 0.0


def test_land_action_raises():
    with pytest.raises(ValueError):
        action_to_offset(ACTION_LAND)


def test_is_land_action():
    assert is_land_action(ACTION_LAND)
    assert not is_land_action(ACTION_HOVER)
    assert not is_land_action(0)
