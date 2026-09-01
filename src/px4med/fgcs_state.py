"""Builds the CEDA-FGCS-PX4 (final) dict observation from the live world state.

Contract (models/README.md, verified against checkpoint 3d0df78d…):
  drones                (5, 22) float32
  patients              (50, 10) float32
  patient_masks         (50,)   bool   — spawned slots (resolved ones stay true)
  pending_patient_masks (50,)   bool   — spawned & unresolved
  local_grids           (5, 3, 21, 21) float32 — obstacle|boundary, wind, low-signal
  mission               (12,)   float32
  action_masks          (5, 6)  bool

Drone feature order (README §Drone feature order):
  0 x/100, 1 y/100, 2 battery/100, 3 landed, 4 depleted,
  5 pad_x/100, 6 pad_y/100, 7 prev_dx, 8 prev_dy (raw cells in {-1,0,1}),
  9 prev collision, 10 collision streak / 4,
  11–16 previous-action one-hot (N,S,W,E,hover,land),
  17 obstacle-aware landing distance / pad map max reachable distance,
  18 safe-return battery margin / 100, clipped [−1, 1],
  19 in wind, 20 in low signal,
  21 global landing phase or individual energy-return phase

Patient feature order:
  0 x/100, 1 y/100, 2 timer/300, 3 weight/3, 4 initial weight/3,
  5 active, 6 pending, 7 delivered, 8 died,
  9 elapsed response time / initial timer, clipped [0, 1]

Mission feature order:
  0 step/deadline, 1 spawn countdown / active spawn interval,
  2 spawned/50, 3 pending/50, 4 delivered/50, 5 died/50,
  6 landed drones / 5, 7 all spawned, 8 all resolved (landing head),
  9–11 W1/W2/W3 service-debt fractions
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .environment import WorldEnvironment

MAX_BATTERY = 100.0
MAX_PATIENT_TIMER = 300
MAX_PATIENTS = 50
NUM_ACTIONS = 6
LOCAL_GRID_RADIUS = 10
COLLISION_STREAK_CAP = 4


def build_observation(world: "WorldEnvironment") -> dict[str, np.ndarray]:
    grid_size = world.grid_size
    n = world.num_drones
    side = 2 * LOCAL_GRID_RADIUS + 1

    drones = np.zeros((n, 22), dtype=np.float32)
    local_grids = np.zeros((n, 3, side, side), dtype=np.float32)
    action_masks = np.zeros((n, NUM_ACTIONS), dtype=bool)

    landing_phase = world.landing_phase()

    for i in range(n):
        x, y = world.agent_grids[i]
        pad_x, pad_y = world.landing_grid(i)
        path_dist, path_max = world.pad_path_distance(i, (x, y))
        margin = world.safe_return_margin(i)

        drones[i, 0] = x / grid_size
        drones[i, 1] = y / grid_size
        drones[i, 2] = world.batteries[i] / MAX_BATTERY
        drones[i, 3] = 1.0 if world.landed[i] else 0.0
        drones[i, 4] = 1.0 if world.depleted[i] else 0.0
        drones[i, 5] = pad_x / grid_size
        drones[i, 6] = pad_y / grid_size
        drones[i, 7] = float(world.prev_displacements[i][0])
        drones[i, 8] = float(world.prev_displacements[i][1])
        drones[i, 9] = 1.0 if world.prev_collisions[i] else 0.0
        drones[i, 10] = min(world.collision_streaks[i], COLLISION_STREAK_CAP) / COLLISION_STREAK_CAP
        prev_action = int(world.prev_actions[i])
        if 0 <= prev_action < NUM_ACTIONS:
            drones[i, 11 + prev_action] = 1.0
        drones[i, 17] = path_dist / max(1, path_max)
        drones[i, 18] = float(np.clip(margin / MAX_BATTERY, -1.0, 1.0))
        drones[i, 19] = 1.0 if (x, y) in world.wind_zones else 0.0
        drones[i, 20] = 1.0 if (x, y) in world.low_signal_zones else 0.0
        drones[i, 21] = 1.0 if (landing_phase or world.return_required(i)) else 0.0

        # Local grids: row index = dy (decreasing y = north), col index = dx.
        for dy in range(-LOCAL_GRID_RADIUS, LOCAL_GRID_RADIUS + 1):
            for dx in range(-LOCAL_GRID_RADIUS, LOCAL_GRID_RADIUS + 1):
                cx, cy = x + dx, y + dy
                row, col = dy + LOCAL_GRID_RADIUS, dx + LOCAL_GRID_RADIUS
                oob = cx < 0 or cx >= grid_size or cy < 0 or cy >= grid_size
                if oob or (cx, cy) in world.obstacles:
                    local_grids[i, 0, row, col] = 1.0
                if not oob and (cx, cy) in world.wind_zones:
                    local_grids[i, 1, row, col] = 1.0
                if not oob and (cx, cy) in world.low_signal_zones:
                    local_grids[i, 2, row, col] = 1.0

        action_masks[i] = world.action_mask(i)

    # ── patients (50, 10) ─────────────────────────────────────────────────────
    patients = np.zeros((MAX_PATIENTS, 10), dtype=np.float32)
    patient_masks = np.zeros(MAX_PATIENTS, dtype=bool)
    pending_masks = np.zeros(MAX_PATIENTS, dtype=bool)
    for p in world.patients[:MAX_PATIENTS]:
        if not p.active:
            continue
        patient_masks[p.idx] = True
        pending_masks[p.idx] = p.pending
        patients[p.idx, 0] = p.grid_x / grid_size
        patients[p.idx, 1] = p.grid_y / grid_size
        patients[p.idx, 2] = max(0, p.timer) / MAX_PATIENT_TIMER
        patients[p.idx, 3] = p.weight / 3.0
        patients[p.idx, 4] = p.initial_weight / 3.0
        patients[p.idx, 5] = 1.0
        patients[p.idx, 6] = 1.0 if p.pending else 0.0
        patients[p.idx, 7] = 1.0 if p.delivered else 0.0
        patients[p.idx, 8] = 1.0 if p.died else 0.0
        patients[p.idx, 9] = float(
            np.clip(p.steps_elapsed / max(1, p.initial_timer), 0.0, 1.0)
        )

    # ── mission (12,) ─────────────────────────────────────────────────────────
    spawned = sum(1 for p in world.patients if p.active)
    pending = world.pending_count()
    delivered = sum(1 for p in world.patients if p.delivered)
    died = sum(1 for p in world.patients if p.died)
    landed_count = sum(1 for i in range(n) if world.landed[i])
    all_spawned = world.all_spawned()
    debt = world.service_debt_fractions()
    mission = np.array([
        min(1.0, world._step_count / max(1, world.max_steps)),
        0.0 if all_spawned else world._new_patient_timer / max(1, world._active_spawn_interval),
        spawned / MAX_PATIENTS,
        pending / MAX_PATIENTS,
        delivered / MAX_PATIENTS,
        died / MAX_PATIENTS,
        landed_count / max(1, n),
        1.0 if all_spawned else 0.0,
        1.0 if landing_phase else 0.0,
        debt[0],
        debt[1],
        debt[2],
    ], dtype=np.float32)

    return {
        "drones": drones,
        "patients": patients,
        "patient_masks": patient_masks,
        "pending_patient_masks": pending_masks,
        "local_grids": local_grids,
        "mission": mission,
        "action_masks": action_masks,
    }
