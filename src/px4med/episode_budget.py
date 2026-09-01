"""Episode step budget — the one place it is defined.

These MUST match the training config in `models/CEDA-FGCS-new.py`
(`MAX_STEPS` / `DEFAULT_LANDING_GRACE_STEPS`), because the mission
observation the policy consumes encodes

    min(1.0, episode_step / current_episode_deadline())

where the deadline is `MISSION_MAX_STEPS` during the rescue phase and
`resolution_step + LANDING_GRACE_STEPS` once every patient is resolved. Run a
smaller mission budget and the policy perceives the clock running
proportionally faster than it ever did in training — off-distribution in a way
that is invisible in the logs.

The env self-terminates on rescue_timeout (mission budget exhausted with
patients unresolved) or landing_timeout (grace exhausted after resolution), so
the driver loop must be allowed to reach `LOOP_STEP_CAP`; capping it at the
mission budget alone truncates episodes that resolve late and under-reports
`all_landed` / `mission_success`.

Kept free of heavy imports so `main.py` can read it at argparse time without
pulling in torch.
"""
from __future__ import annotations

MISSION_MAX_STEPS = 800
LANDING_GRACE_STEPS = 400
LOOP_STEP_CAP = MISSION_MAX_STEPS + LANDING_GRACE_STEPS
