# px4med

Runs a trained CTDE multi-agent RL policy on two PX4 SITL drones. The scenario is a two-UAV medical delivery task with dynamic patient triage using a policy trained in a custom gym environment. This repo deploys it via MAVSDK against real PX4 SITL instances running in Docker.

Both drones run as separate PX4 SITL instances inside a single Docker container, sharing one headless Gazebo session. The Python control loop connects to each instance over MAVSDK, polls telemetry at ~2 Hz, rebuilds the training environment's state vector, and dispatches waypoint commands based on the policy's greedy action output.

## Requirements

- Docker
- Python 3.11 with Poetry

## Running

```bash
# install deps
poetry install

# run simulation (sets up docker image for you)
poetry run px4med
```

If you already have SITL running externally (e.g. for debugging), skip the Docker lifecycle:

```bash
poetry run px4med --no-docker --episodes 1
```

## Ports

| Drone | Sim TCP | MAVSDK UDP |
| ----- | ------- | ---------- |
| 0     | 4560    | 14540      |
| 1     | 4561    | 14541      |

## Tests

```bash
poetry run pytest
```

Most tests are pure unit tests and don't need SITL. `test_state_matches_training_env` requires `AneeshMARL5.py` in the repo root.
