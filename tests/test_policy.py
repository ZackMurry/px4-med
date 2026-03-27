"""Loads model, feeds a known state, asserts deterministic action output."""
import pytest
from pathlib import Path


MODEL_PATH = Path(__file__).parents[1] / "models" / "ctde_agent_marl9.pth"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model file not present")
def test_policy_forward_pass():
    from px4med.policy import PolicyNet
    policy = PolicyNet(MODEL_PATH)
    dummy_state = [0.0] * 140
    action = policy.select_action(dummy_state, dummy_state)
    assert 0 <= action <= 4


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model file not present")
def test_policy_is_deterministic():
    from px4med.policy import PolicyNet
    policy = PolicyNet(MODEL_PATH)
    state = [0.1] * 140
    a1 = policy.select_action(state, state)
    a2 = policy.select_action(state, state)
    assert a1 == a2
