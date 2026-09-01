# CEDA-FGCS-PX4

This folder packages the final CEDA-FGCS-PX4 policy for five-vehicle PX4
simulation. It contains the exact shared local DQN used for decentralized
execution, the training-only QMIX mixer for offline diagnostics, and the final
12,000-episode checkpoint.

The module is intentionally inference-only. It does not contain the grid-world
trainer, a ROS 2 node, or a PX4 flight controller. A simulator adapter must
construct the observation described below and translate each discrete policy
action into a bounded PX4 setpoint or landing request.

## Package contents

```text
CEDA-FGCS-PX4/
├── CEDA-FGCS.py
├── README.md
└── weights/
    └── ctde_agent_marl_FGCS.pth
```

The checkpoint contains the online and target shared-DQN parameters, online and
target QMIX parameters, optimizer state, and training metadata. PX4 execution
uses only `policy_state_dict`; QMIX is not an action-selection policy.

## Packaged model

| Property | Value |
|---|---:|
| Model | `CEDA-FGCS-PX4` |
| Training episodes | 12,000 |
| Environment steps | 4,835,653 |
| Learner updates | 149,551 |
| Drones | 5 |
| Maximum patients | 50 |
| Abstract grid | 100 × 100 |
| Actions | north, south, west, east, hover, land |
| Clean battery endurance | 500 policy steps |
| Checkpoint SHA-256 | `3d0df78d5291edb730bfb507568d2a9c0c5b0f03cf70842a14e329bc1abd6275` |

## Requirements and package verification

Use Python 3.9 or newer with NumPy and PyTorch 2.x. CUDA is optional for
inference.

```bash
python3 -m pip install numpy torch
cd /path/to/CEDA-FGCS-PX4
python3 CEDA-FGCS.py --device cpu --show-metadata --smoke-test
```

To verify the training-only QMIX tensors as well:

```bash
python3 CEDA-FGCS.py \
  --device cpu \
  --show-metadata \
  --smoke-test \
  --mixer-diagnostic
```

The smoke observation is synthetic. These commands verify checkpoint identity,
strict tensor loading, tensor shapes, a DQN forward pass, greedy action
selection, and optionally a QMIX forward pass; they do not reproduce mission
performance.

## Integration architecture

```text
PX4 telemetry + synchronized mission/hazard simulator state
                              │
                              ▼
                  observation builder
                              │
                              ▼
             shared local DQN (one row per drone)
                              │
                              ▼
         action index → bounded position target or land request
                              │
                              ▼
            ROS 2 Offboard adapter → PX4 SITL vehicles
```

For physically decentralized execution, run the same shared DQN on each vehicle
and use only that vehicle's action row. Each replica still needs the patient set
and the other-drone entity table because triage decisions and collision avoidance
were learned from those inputs. The model never needs the QMIX mixer during
execution.

A single process may evaluate all five rows once per synchronized decision tick
and route the actions to five SITL namespaces. That is computationally efficient,
but it is a centralized deployment process even though the action network itself
uses decentralized observations.

## Observation contract

Call `CEDAFGCSPX4Policy.select_actions(observation)` with one synchronized
dictionary. Continuous arrays must be finite and are converted to `float32`;
masks are converted to Boolean tensors.

| Key | Shape | Meaning |
|---|---:|---|
| `drones` | `(5, 22)` | One state row per drone |
| `patients` | `(50, 10)` | One row per patient slot |
| `patient_masks` | `(50,)` | Active/spawned patient slots |
| `pending_patient_masks` | `(50,)` | Active unresolved patient slots |
| `local_grids` | `(5, 3, 21, 21)` | Per-drone obstacle, wind, and low-signal maps |
| `mission` | `(12,)` | Shared mission progress and service debt |
| `action_masks` | `(5, 6)` | Valid actions in model action order |

`pending_patient_masks` must be a subset of `patient_masks`, and every drone
must have at least one valid action.

### Drone feature order

The 22 drone features are:

| Index | Feature |
|---:|---|
| 0–1 | Grid x and y divided by 100 |
| 2 | Model battery percentage divided by 100 |
| 3 | Landed flag |
| 4 | Irreversible drone-died/battery-depleted flag |
| 5–6 | Assigned landing-zone x and y divided by 100 |
| 7–8 | Previous grid displacement `(dx, dy)` in `{-1, 0, 1}` |
| 9 | Previous-step collision flag |
| 10 | Collision streak divided by the training cap of 4 |
| 11–16 | Previous action as a six-value one-hot vector |
| 17 | Obstacle-aware landing distance divided by that pad map's maximum reachable distance |
| 18 | Safe-return battery margin divided by 100 and clipped to `[-1, 1]` |
| 19 | Current cell is in a wind zone |
| 20 | Current cell is in a low-signal zone |
| 21 | Global landing phase or individual energy-return phase |

The other four drones remain inputs to each local policy row. Do not remove
those entities: the drone Set Transformer uses them for decentralized collision
avoidance and coordination.

### Patient feature order

The 10 patient features are:

| Index | Feature |
|---:|---|
| 0–1 | Grid x and y divided by 100 |
| 2 | Remaining timer divided by 300 |
| 3 | Current triage weight divided by 3 |
| 4 | Initial triage weight divided by 3 |
| 5 | Active flag |
| 6 | Pending/unresolved flag |
| 7 | Successfully delivered flag |
| 8 | Died flag |
| 9 | Elapsed response time divided by that patient's initial timer, clipped to `[0, 1]` |

Inactive rows must remain zero with both patient masks false. The final training
stage initializes patients with a 220-step timer, but timer feature 2 is still
normalized by the global training maximum of 300.

### Local-grid channels

Each 21 × 21 grid is centered on its drone with radius 10:

1. obstacle or outside-world boundary;
2. wind zone;
3. low-signal zone.

Values use the exact grid orientation from training: decreasing y is north and
increasing x is east. Cells outside the 100 × 100 world are marked as obstacles.
The model uses both immediate candidate cells and directional corridor features,
so do not resize, rotate, transpose, or blur these maps.

### Mission feature order

The 12 mission features are:

| Index | Feature |
|---:|---|
| 0 | Episode step divided by the active episode deadline |
| 1 | Remaining spawn countdown divided by the active spawn interval plus jitter |
| 2 | Spawned patients divided by 50 |
| 3 | Pending patients divided by 50 |
| 4 | Successfully delivered patients divided by 50 |
| 5 | Dead patients divided by 50 |
| 6 | Landed drones divided by 5 |
| 7 | All patients have spawned |
| 8 | All patients are resolved; this activates the landing head |
| 9–11 | W1, W2, and W3 service-debt fractions |

For triage class `w`, compute service debt as:

```text
target_count = target_rate[w] × spawned_count[w]
debt_count = max(0, target_count - delivered_count[w])
debt_fraction = debt_count / max(1, target_count)
```

The target rates are W1 `0.50`, W2 `0.70`, and W3 `0.90`. Patient death does not
erase service debt.

### Action-mask semantics

The action order is north, south, west, east, hover, land.

- Operational drone away from a permitted landing state: movement and hover are
  available; land is unavailable.
- Live drone at its assigned pad after all patients resolve or after its energy
  return activates: land is the only action.
- Landed or dead drone: hover is the only action.
- Boundary, obstacle, and occupied-cell movement actions were not masked during
  training. Avoidance was learned through observation and reward.

To reproduce the trained policy, retain these mask semantics. If a separate PX4
safety supervisor overrides an unsafe setpoint, log that intervention because it
changes the executed policy from the learned-policy evaluation.

## Reproducing the energy state

The trained battery is a mission-energy ledger, not an uncalibrated physical
battery model. For simulation fidelity, initialize every drone at `100.0` and
apply:

- clean operational step: `0.20` percentage points;
- additional drain when the resulting cell is in wind: `2.30` points;
- standby step on the assigned landing zone: `0.02` points.

Do not substitute PX4 `BatteryStatus.remaining` directly unless the simulator
has been calibrated to these same decision-step semantics.

The safe-return feature uses an obstacle- and hazard-aware expected-energy map.
For a transition from `origin` to `destination`:

```text
p_success = 1.0
if origin is low signal: p_success *= 0.50
if origin is wind:       p_success *= 0.85

expected_failures = (1 - p_success) / p_success
failed_cost = 0.20 + (2.30 if origin is wind else 0)
success_cost = 0.20 + (2.30 if destination is wind else 0)
transition_cost = success_cost + expected_failures × failed_cost
```

Run Dijkstra from each assigned landing pad using those directed transition
costs and treating obstacles as impassable. Then compute:

```text
required_return_battery = expected_route_cost + 18.0
safe_return_margin = battery - required_return_battery
return_required = battery <= 20.0 or safe_return_margin <= 0
```

Drone feature 18 is `clip(safe_return_margin / 100, -1, 1)`. Feature 21 is true
when `return_required` is true or all patients are resolved.

## Loading from a ROS 2 node

Because the Python filename contains a hyphen, load it with `importlib`:

```python
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

package = Path("/path/to/CEDA-FGCS-PX4")
spec = spec_from_file_location("ceda_fgcs_px4", package / "CEDA-FGCS.py")
ceda = module_from_spec(spec)
spec.loader.exec_module(ceda)

policy = ceda.CEDAFGCSPX4Policy(
    package / "weights" / "ctde_agent_marl_FGCS.pth",
    device="cpu",
    load_mixer=False,
)

# observation = build_synchronized_observation(...)
fleet_actions = policy.select_actions(observation)
vehicle_action = fleet_actions[vehicle_index]

# Equivalent helper when each process owns one vehicle:
vehicle_action = policy.select_agent_action(observation, vehicle_index)
```

## Mapping actions to PX4 local NED

Choose a fixed `meters_per_cell`, horizontal grid origin, and flight altitude.
One consistent mapping is:

```text
NED north = origin_north - grid_y × meters_per_cell
NED east  = origin_east  + grid_x × meters_per_cell
```

| Model action | Grid change | Local-NED target change |
|---|---:|---:|
| north | `(0, -1)` | `north += meters_per_cell` |
| south | `(0, +1)` | `north -= meters_per_cell` |
| west | `(-1, 0)` | `east -= meters_per_cell` |
| east | `(+1, 0)` | `east += meters_per_cell` |
| hover | none | hold the current bounded setpoint |
| land | none | request PX4 landing at the assigned pad |

Keep the high-level policy decision loop separate from the PX4 Offboard
heartbeat. Publish `OffboardControlMode` and the active `TrajectorySetpoint`
continuously (the official example uses a 100 ms timer), but request a new grid
action only after the previous cell target reaches the configured position and
velocity tolerance. Do not convert a grid action into an open-loop velocity
command.

The current PX4 example streams ten setpoints before switching to Offboard and
arming, and PX4 exits Offboard if the control-mode stream falls below roughly
2 Hz. Continue publishing the held setpoint between model decisions and confirm
commands through `VehicleCommandAck`.

## Multi-vehicle PX4 requirements

- Maintain one permanent mapping between CEDA agent indices `0..4`, PX4
  instances, ROS 2 namespaces, and `MAV_SYS_ID` values.
- PX4 simulation instances greater than zero normally use `px4_<instance>`
  namespaces; the first instance has no namespace by default. Setting
  `PX4_UXRCE_DDS_NS` explicitly for every vehicle avoids this asymmetry.
- Set each `VehicleCommand.target_system` to the intended vehicle's
  `MAV_SYS_ID`; otherwise PX4 may ignore the command.
- Use a `px4_msgs` branch compatible with the PX4 firmware. PX4 v1.16 and newer
  can use the PX4 ROS 2 Message Translation Node for versioned messages.
- Use simulator time consistently and synchronize the vehicle, patient, hazard,
  and network state before every policy decision.

Current official references:

- [PX4 ROS 2 user guide](https://docs.px4.io/main/en/ros2/user_guide)
- [PX4 multi-vehicle ROS 2 simulation](https://docs.px4.io/main/en/ros2/multi_vehicle)
- [PX4 ROS 2 Offboard example](https://docs.px4.io/main/en/ros2/offboard_control)
- [PX4 uXRCE-DDS bridge](https://docs.px4.io/main/en/middleware/uxrce_dds)

## Adapter responsibilities

The external simulator/ROS 2 adapter must provide:

- synchronized positions, landed/dead states, and previous actions for all five
  drones;
- the mission-energy ledger and safe-return maps;
- patient positions, spawn states, timers, triage history, response ages, and
  outcomes;
- 21 × 21 obstacle, wind, and low-signal maps centered on each drone;
- W1/W2/W3 service debt and other mission features;
- high-level action completion and collision feedback;
- correct ROS 2 namespace, target-system, arming, Offboard, setpoint, and landing
  handling for each PX4 vehicle.

## Safety scope

This package is research software intended for simulation. The policy produces
high-level discrete decisions and is not a replacement for PX4 stabilization,
state estimation, failsafes, geofencing, collision prevention, or a certified
flight-safety system. Hardware testing requires an independent safety review and
a supervised, recoverable test plan.
