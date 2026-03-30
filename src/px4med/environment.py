"""World state manager mirroring the training environment transition logic.

The SITL loop still executes actions through PX4, but this module owns the
training-environment state that the policy was trained against:
obstacles, hazard zones, patient progression, virtual battery, landing
eligibility, and reward accounting.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

# ── constants mirroring AneeshMARL5.py ───────────────────────────────────────

GRID_SIZE = 50
METERS_PER_CELL = 2.0       # 1 grid cell = 2 m in NED space

MAX_PATIENT_TIMER = 250
MAX_PATIENT_WEIGHT = 3
NEW_PATIENT_SPAWN_INTERVAL = 75
MAX_PATIENTS = 8
NUM_PATIENTS = 4            # patients active at episode start

NUM_WIND_ZONES = 15
NUM_LOW_SIGNAL_ZONES = 10
WIND_APPEAR_INTERVAL = 30
LOW_SIGNAL_APPEAR_INTERVAL = 30
NUM_OBSTACLES = 200

MAX_BATTERY = 100.0
BATTERY_DRAIN_PER_STEP = 0.1
BATTERY_DRAIN_IN_WIND = 0.5
LOW_BATTERY_THRESHOLD = 20.0
LOW_SIGNAL_FAILURE_PROB = 0.3

GOAL_REWARD = 100.0
STEP_PENALTY = -0.2
CLEAN_STEP_BONUS = 0.1
COLLISION_PENALTY = -1000.0
AGENT_COLLISION_PENALTY = -1000.0
BATTERY_DEPLETION_PENALTY = -50.0
LOW_BATTERY_PENALTY = -0.5
WIND_PENALTY = -2.0
LOW_SIGNAL_PENALTY = -8.0
SHAPING_FACTOR = 1.5
PATIENT_DEATH_PENALTY = -30.0
LANDING_REWARD = 150.0
LAND_WRONG_PENALTY = -2.0
CLOSENESS_PENALTY = -10.0
CLOSENESS_RADIUS = 4
STEP_CLIP = 5.0

# Fixed-layout defaults matching AneeshMARL5.py Environment(fixed_layout=True)
_DEFAULT_PATIENT_GRIDS = [
    (13, 13), (13, 1), (25, 25), (25, 1),
    (35, 10), (10, 35), (40, 30), (30, 40),
]
_DEFAULT_LANDING_ZONE_GRIDS = [(48, 48), (48, 45)]
_DEFAULT_START_GRIDS = [(1, 1), (1, 13)]


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class Patient:
    idx: int
    grid_x: float
    grid_y: float
    north_m: float
    east_m: float
    weight: int
    timer: int = MAX_PATIENT_TIMER
    active: bool = True
    delivered: bool = False
    actually_delivered: bool = False   # True only when delivered by a drone (not timed out)
    steps_elapsed: int = 0
    decay_a: float = 0.0
    decay_b: float = 0.0
    thresh_serious: float = 0.0
    thresh_critical: float = 0.0


# ── world environment ─────────────────────────────────────────────────────────

class WorldEnvironment:
    """Tracks dynamic world state using the same abstractions as `train.py`."""

    DELIVERY_RADIUS_M: float = 2.0   # 1 grid cell radius

    def __init__(self, config: dict) -> None:
        self.config = config
        self.patients: list[Patient] = []
        self.wind_zones: set[tuple[int, int]] = set()
        self.low_signal_zones: set[tuple[int, int]] = set()
        self.obstacles: set[tuple[int, int]] = set()
        # One entry per drone: (north_m, east_m)
        self.landing_zones: list[tuple[float, float]] = []
        self.start_grids: list[tuple[int, int]] = list(_DEFAULT_START_GRIDS)
        self.agent_grids: list[tuple[int, int]] = list(_DEFAULT_START_GRIDS)
        self.batteries: list[float] = [MAX_BATTERY, MAX_BATTERY]
        self.landed: list[bool] = [False, False]

        self._new_patient_timer: int = NEW_PATIENT_SPAWN_INTERVAL
        self._wind_timer: int = WIND_APPEAR_INTERVAL
        self._ls_timer: int = LOW_SIGNAL_APPEAR_INTERVAL
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Initialise world state from config. Safe to call between episodes."""
        grid_cfg = self.config.get("grid", {})
        mpc = float(grid_cfg.get("meters_per_cell", METERS_PER_CELL))
        self.start_grids = [
            tuple(p)
            for p in self.config.get("agent_start_positions", _DEFAULT_START_GRIDS)
        ]
        self.agent_grids = list(self.start_grids)
        self.batteries = [MAX_BATTERY, MAX_BATTERY]
        self.landed = [False, False]

        # Obstacles
        obs_cfgs = self.config.get("obstacles")
        if obs_cfgs:
            self.obstacles = {tuple(obs["grid"]) for obs in obs_cfgs}
        else:
            self.obstacles = self._generate_obstacles()

        # Landing zones
        lz_cfgs = self.config.get(
            "landing_zones",
            [{"grid": list(g)} for g in _DEFAULT_LANDING_ZONE_GRIDS],
        )
        self.landing_zones = []
        for lz in lz_cfgs:
            gx, gy = lz["grid"]
            self.landing_zones.append((-gy * mpc, gx * mpc))   # (north_m, east_m)

        # Patients — pad to MAX_PATIENTS
        patient_cfgs = self.config.get(
            "patients",
            [{"grid": list(g)} for g in _DEFAULT_PATIENT_GRIDS],
        )
        self.patients = []
        for i, pc in enumerate(patient_cfgs[:MAX_PATIENTS]):
            gx, gy = pc["grid"]
            w = pc.get("weight", random.randint(1, MAX_PATIENT_WEIGHT))
            decay_a, decay_b, thresh_serious, thresh_critical = self._sample_decay_params(w)
            self.patients.append(Patient(
                idx=i,
                grid_x=float(gx), grid_y=float(gy),
                north_m=-gy * mpc, east_m=gx * mpc,
                weight=w, timer=MAX_PATIENT_TIMER,
                active=(i < NUM_PATIENTS),
                delivered=False, actually_delivered=False,
                steps_elapsed=0,
                decay_a=decay_a,
                decay_b=decay_b,
                thresh_serious=thresh_serious,
                thresh_critical=thresh_critical,
            ))
        while len(self.patients) < MAX_PATIENTS:
            i = len(self.patients)
            self.patients.append(Patient(
                idx=i, grid_x=0.0, grid_y=0.0,
                north_m=0.0, east_m=0.0,
                weight=1, timer=MAX_PATIENT_TIMER,
                active=False, delivered=False, actually_delivered=False,
                steps_elapsed=0,
                decay_a=0.0,
                decay_b=0.0,
                thresh_serious=0.0,
                thresh_critical=0.0,
            ))

        self.wind_zones = set()
        self.low_signal_zones = set()
        self._new_patient_timer = NEW_PATIENT_SPAWN_INTERVAL
        self._wind_timer = WIND_APPEAR_INTERVAL
        self._ls_timer = LOW_SIGNAL_APPEAR_INTERVAL
        self._step_count = 0

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def refresh_hazards(self) -> None:
        """Refresh wind/low-signal zones using the same cadence as training."""
        self._step_count += 1
        grid_size = self.config.get("grid", {}).get("size", GRID_SIZE)
        interior = [
            (x, y)
            for x in range(2, grid_size - 2)
            for y in range(2, grid_size - 2)
        ]

        # Wind zones — refresh on same interval as training env
        if self._wind_timer > 0:
            self._wind_timer -= 1
        else:
            self.wind_zones = set(
                random.sample(interior, min(NUM_WIND_ZONES, len(interior)))
            )
            self._wind_timer = WIND_APPEAR_INTERVAL

        # Low-signal zones
        if self._ls_timer > 0:
            self._ls_timer -= 1
        else:
            self.low_signal_zones = set(
                random.sample(interior, min(NUM_LOW_SIGNAL_ZONES, len(interior)))
            )
            self._ls_timer = LOW_SIGNAL_APPEAR_INTERVAL

    def step(self, actions: list[int]) -> dict:
        """Apply one training-env transition from the current world state."""
        self.refresh_hazards()
        grid_size = int(self.config.get("grid", {}).get("size", GRID_SIZE))

        old_positions = list(self.agent_grids)
        old_shaping_dist: list[int] = []
        for i in range(2):
            nearest = self.nearest_undelivered_patient(old_positions[i])
            if nearest is not None:
                old_shaping_dist.append(
                    self.manhattan_distance(old_positions[i], self.patient_grid(nearest))
                )
            else:
                old_shaping_dist.append(
                    self.manhattan_distance(old_positions[i], self.landing_grid(i))
                )

        step_rewards = [STEP_PENALTY, STEP_PENALTY]
        milestone_rewards = [0.0, 0.0]
        step_data = {
            "wind_entries": [0, 0],
            "low_signal_entries": [0, 0],
            "obstacle_collisions": 0,
            "agent_collisions": 0,
            "actions_executed": list(actions),
            "landing_attempts": [False, False],
            "landed_this_step": [False, False],
            "deliveries": [],
            "sim_positions": list(old_positions),
            "rewards": [0.0, 0.0],
            "done": False,
        }

        new_positions: list[tuple[int, int]] = []
        for agent_idx, action in enumerate(actions):
            x, y = self.agent_grids[agent_idx]
            if self.landed[agent_idx]:
                new_positions.append((x, y))
                continue

            if action == 4:
                step_data["landing_attempts"][agent_idx] = True
                if (x, y) == self.landing_grid(agent_idx):
                    self.landed[agent_idx] = True
                    step_data["landed_this_step"][agent_idx] = True
                    milestone_rewards[agent_idx] += LANDING_REWARD
                else:
                    step_rewards[agent_idx] += LAND_WRONG_PENALTY
                new_positions.append((x, y))
                continue

            in_low_signal = (x, y) in self.low_signal_zones
            if in_low_signal and random.random() < LOW_SIGNAL_FAILURE_PROB:
                new_x, new_y = x, y
            else:
                if action == 0:
                    new_x, new_y = x, y - 1
                elif action == 1:
                    new_x, new_y = x, y + 1
                elif action == 2:
                    new_x, new_y = x - 1, y
                elif action == 3:
                    new_x, new_y = x + 1, y
                else:
                    new_x, new_y = x, y

            if (
                new_x < 0
                or new_x >= grid_size
                or new_y < 0
                or new_y >= grid_size
                or (new_x, new_y) in self.obstacles
            ):
                milestone_rewards[agent_idx] += COLLISION_PENALTY
                step_data["obstacle_collisions"] += 1
                new_x, new_y = x, y

            new_positions.append((new_x, new_y))

        active = [i for i in range(2) if not self.landed[i]]
        if len(active) == 2 and new_positions[active[0]] == new_positions[active[1]]:
            for i in active:
                milestone_rewards[i] += AGENT_COLLISION_PENALTY
                new_positions[i] = old_positions[i]
            step_data["agent_collisions"] += 1

        self.agent_grids = new_positions
        step_data["sim_positions"] = list(new_positions)

        active_agents = [i for i in range(2) if not self.landed[i]]
        if len(active_agents) == 2:
            dist = abs(new_positions[0][0] - new_positions[1][0]) + abs(new_positions[0][1] - new_positions[1][1])
            if dist < CLOSENESS_RADIUS:
                for i in active_agents:
                    step_rewards[i] += CLOSENESS_PENALTY

        for i in range(2):
            if self.landed[i]:
                continue

            in_wind = self.agent_grids[i] in self.wind_zones
            if in_wind:
                self.batteries[i] -= BATTERY_DRAIN_PER_STEP + BATTERY_DRAIN_IN_WIND
                step_rewards[i] += WIND_PENALTY
                step_data["wind_entries"][i] += 1
            else:
                self.batteries[i] -= BATTERY_DRAIN_PER_STEP

            if self.agent_grids[i] in self.low_signal_zones:
                step_rewards[i] += LOW_SIGNAL_PENALTY
                step_data["low_signal_entries"][i] += 1

            if 0 < self.batteries[i] < LOW_BATTERY_THRESHOLD:
                step_rewards[i] += LOW_BATTERY_PENALTY

            if self.batteries[i] <= 0:
                milestone_rewards[i] += BATTERY_DEPLETION_PENALTY
                self.batteries[i] = 0.0

        self._advance_patients(milestone_rewards)

        for i in range(2):
            if self.landed[i]:
                continue
            for p in self.patients:
                if not p.active or p.delivered:
                    continue
                if self.agent_grids[i] == self.patient_grid(p.idx):
                    timer_ratio = p.timer / MAX_PATIENT_TIMER
                    milestone_rewards[i] += GOAL_REWARD * timer_ratio * p.weight
                    p.delivered = True
                    p.actually_delivered = True
                    step_data["deliveries"].append(p.idx)

        for i in range(2):
            if self.landed[i]:
                continue
            nearest = self.nearest_undelivered_patient(self.agent_grids[i])
            if nearest is not None:
                new_dist = self.manhattan_distance(self.agent_grids[i], self.patient_grid(nearest))
            else:
                new_dist = self.manhattan_distance(self.agent_grids[i], self.landing_grid(i))
            step_rewards[i] += SHAPING_FACTOR * (old_shaping_dist[i] - new_dist)

        rewards = [
            max(-STEP_CLIP, min(STEP_CLIP, step_rewards[i])) + milestone_rewards[i]
            for i in range(2)
        ]
        done = all(self.landed) or any(b <= 0 for b in self.batteries)

        step_data["rewards"] = rewards
        step_data["done"] = done
        return step_data

    def _advance_patients(self, milestone_rewards: list[float]) -> None:
        """Advance patient timers and spawn logic after movement, as in training."""
        self._new_patient_timer -= 1
        if self._new_patient_timer <= 0:
            self._new_patient_timer = NEW_PATIENT_SPAWN_INTERVAL
            for p in self.patients:
                if not p.active and not p.delivered:
                    p.active = True
                    p.timer = MAX_PATIENT_TIMER
                    p.weight = random.randint(1, MAX_PATIENT_WEIGHT)
                    p.steps_elapsed = 0
                    (
                        p.decay_a,
                        p.decay_b,
                        p.thresh_serious,
                        p.thresh_critical,
                    ) = self._sample_decay_params(p.weight)
                    break

        for p in self.patients:
            if not p.active or p.delivered:
                continue
            p.timer -= 1
            if p.timer <= 0:
                milestone_rewards[0] += PATIENT_DEATH_PENALTY / 2
                milestone_rewards[1] += PATIENT_DEATH_PENALTY / 2
                p.delivered = True
                continue

            p.steps_elapsed += 1
            survival = 1.0 / (1.0 + math.exp(p.decay_a * p.steps_elapsed - p.decay_b))
            if survival < p.thresh_critical:
                p.weight = 3
            elif survival < p.thresh_serious:
                p.weight = 2
            else:
                p.weight = 1

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_grid_pos(self, north_m: float, east_m: float) -> tuple[int, int]:
        """Convert NED metres to the nearest integer grid cell (x, y)."""
        mpc = float(self.config.get("grid", {}).get("meters_per_cell", METERS_PER_CELL))
        return round(east_m / mpc), round(-north_m / mpc)

    def grid_to_ned(self, grid_x: int, grid_y: int) -> tuple[float, float]:
        """Convert integer grid position to NED metres."""
        mpc = float(self.config.get("grid", {}).get("meters_per_cell", METERS_PER_CELL))
        return -grid_y * mpc, grid_x * mpc

    def landing_grid(self, agent_idx: int) -> tuple[int, int]:
        north_m, east_m = self.landing_zones[agent_idx]
        return self.get_grid_pos(north_m, east_m)

    def patient_grid(self, patient_idx: int) -> tuple[int, int]:
        patient = self.patients[patient_idx]
        return round(patient.grid_x), round(patient.grid_y)

    def manhattan_distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def nearest_undelivered_patient(self, pos: tuple[int, int]) -> Optional[int]:
        best_idx: Optional[int] = None
        best_dist = math.inf
        for p in self.patients:
            if not p.active or p.delivered:
                continue
            dist = self.manhattan_distance(pos, self.patient_grid(p.idx))
            if dist < best_dist:
                best_dist = dist
                best_idx = p.idx
        return best_idx

    def triage_summary(self) -> dict[str, float | int]:
        triage_data: dict[str, float | int] = {
            "delivered_w1": 0,
            "delivered_w2": 0,
            "delivered_w3": 0,
            "died_w1": 0,
            "died_w2": 0,
            "died_w3": 0,
            "weighted_delivery_score": 0.0,
            "max_possible_weighted_score": 0.0,
            "triage_efficiency": 0.0,
        }
        for p in self.patients:
            if not p.active:
                continue
            weight = int(p.weight)
            triage_data["max_possible_weighted_score"] += weight
            if p.actually_delivered:
                triage_data[f"delivered_w{weight}"] += 1
                triage_data["weighted_delivery_score"] += weight
            elif p.delivered:
                triage_data[f"died_w{weight}"] += 1
        if triage_data["max_possible_weighted_score"] > 0:
            triage_data["triage_efficiency"] = (
                triage_data["weighted_delivery_score"] / triage_data["max_possible_weighted_score"]
            )
        return triage_data

    def _sample_decay_params(self, initial_weight: int) -> tuple[float, float, float, float]:
        """Match train.py logistic acuity progression for each patient."""
        if initial_weight == 1:
            decay_a = random.uniform(0.02, 0.05)
            decay_b = random.uniform(3.0, 5.0)
            thresh_serious = random.uniform(0.50, 0.70)
            thresh_critical = random.uniform(0.15, 0.30)
        elif initial_weight == 2:
            decay_a = random.uniform(0.05, 0.10)
            decay_b = random.uniform(2.0, 3.5)
            thresh_serious = random.uniform(0.45, 0.65)
            thresh_critical = random.uniform(0.15, 0.30)
        else:
            decay_a = random.uniform(0.10, 0.20)
            decay_b = random.uniform(1.0, 2.5)
            thresh_serious = random.uniform(0.40, 0.60)
            thresh_critical = random.uniform(0.10, 0.25)

        thresh_critical = min(thresh_critical, thresh_serious - 0.05)
        return decay_a, decay_b, thresh_serious, thresh_critical

    def _generate_obstacles(self) -> set[tuple[int, int]]:
        """Match the training obstacle generator for the fixed-layout world."""
        grid_size = int(self.config.get("grid", {}).get("size", GRID_SIZE))
        protected = {
            (1, 1), (1, 13), (13, 13), (13, 1), (25, 25), (25, 1), (48, 48)
        }
        obstacles: set[tuple[int, int]] = set()

        for y in range(3, 12):
            if y not in [5, 6, 9, 10]:
                obstacles.add((7, y))

        while len(obstacles) < NUM_OBSTACLES:
            x = random.randint(2, grid_size - 3)
            y = random.randint(2, grid_size - 3)
            if (x, y) in protected:
                continue
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            blocked = sum(
                1
                for nx, ny in neighbors
                if (nx, ny) in obstacles or nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size
            )
            if blocked < 3:
                obstacles.add((x, y))

        return obstacles
