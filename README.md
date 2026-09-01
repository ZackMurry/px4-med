# px4med

Runs the trained CEDA-FGCS-PX4 CTDE multi-agent RL policy on five PX4 SITL
drones. The scenario is a five-UAV medical delivery mission with up to 50
dynamically spawning patients, triage acuity progression, wind/low-signal
hazards, and a mission-energy ledger, on a 100×100 abstract grid. This repo
deploys the policy via MAVSDK against real PX4 SITL instances running in
Docker.

All drones run as separate PX4 SITL instances inside a single Docker
container, sharing one headless Gazebo server (started directly — no
per-container rebuild). The Python control loop connects to each instance over
MAVSDK, polls telemetry at ~2 Hz, rebuilds the training environment's dict
observation (`src/px4med/fgcs_state.py`), and dispatches waypoint commands
from the policy's masked greedy action output.

The model package lives in `models/` (`CEDA-FGCS.py` loader + checkpoint +
contract README). Verify it with:

```bash
python3 models/CEDA-FGCS.py --device cpu --show-metadata --smoke-test
```

## Requirements

- Docker
- Python 3.11+ with Poetry (torch comes from the system installation via
  `system-site-packages`; see the note in `pyproject.toml`)

## Running

```bash
# install deps
poetry install

# run one SITL episode (sets up the docker container for you)
poetry run px4med --episodes 1

# skip the Docker lifecycle if SITL is already running externally
poetry run px4med --no-docker --episodes 1
```

PX4 console output is discarded by default to avoid filling storage
(`PX4_VERBOSE_LOGS=1` in the container env re-enables it), onboard ULog
logging is disabled (`SDLOG_MODE=-1`), and per-instance log dirs are deleted
after each container stop.

## Offline sanity checks (no SITL)

```bash
# policy + world + observation builder end-to-end rollout
poetry run python scripts/offline_rollout.py --episodes 1

# directional/landing probes (catches coordinate-convention bugs)
poetry run python scripts/probe_directions.py
```

## Validation Experiments

The experiment suites (`px4med-experiments`, `px4med-overnight-validation`)
are still being ported from the previous 2-drone model and are not currently
runnable.

## Ports

| Drone | MAVSDK UDP | MAV_SYS_ID |
| ----- | ---------- | ---------- |
| 0     | 14540      | 1          |
| 1     | 14541      | 2          |
| 2     | 14542      | 3          |
| 3     | 14543      | 4          |
| 4     | 14544      | 5          |

## Tests

```bash
poetry run pytest
```

All tests are pure unit tests and don't need SITL or the model checkpoint.
