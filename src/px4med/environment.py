"""World state manager mirroring the CEDA-FGCS-PX4 (final) training environment.

The SITL loop still executes actions through PX4, but this module owns the
training-environment state that the policy was trained against: obstacles,
hazard zones, patient progression, the mission-energy ledger, landing
eligibility, and reward accounting.

Targets the 5-drone / 50-patient / 100×100 CEDA-FGCS-PX4 model
(models/CEDA-FGCS.py, checkpoint 3d0df78d…). Action space:
0=north 1=south 2=west 3=east 4=hover 5=land.

Contract-backed semantics (models/README.md "Reproducing the energy state"):
  - battery ledger: 0.20/step clean, +2.30 when the resulting cell is in wind,
    0.02 standby on the assigned pad; NOT PX4 BatteryStatus.
  - movement failure: p_success *= 0.50 in low signal, *= 0.85 in wind
  - safe-return margin from a Dijkstra expected-energy map + 18.0 buffer;
    return_required when battery <= 20 or margin <= 0

PoC ASSUMPTIONS (not derivable from the shipped package; confirm with the
model's author):
  - dynamic spawn cadence/budget (SPAWN_INTERVAL/SPAWN_TOTAL) and jitter
  - hazard rectangle count/size distribution and refresh cadence
  - obstacle count and layout generator (connected-random in training)
  - start grids and landing pad layout for the 5 drones
  - reward constants (diagnostic only; paper metrics don't depend on them)
"""
from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

# ── constants mirroring the CEDA-FGCS-PX4 deployment contract ─────────────────

GRID_SIZE = 100
METERS_PER_CELL = 2.0       # 1 grid cell = 2 m in NED space

NUM_DRONES = 5
MAX_PATIENT_TIMER = 300     # normalization maximum (training global max)
INITIAL_PATIENT_TIMER = 220 # final-stage spawn timer (README)
MAX_PATIENT_WEIGHT = 3
MAX_PATIENTS = 50           # observation slots (fixed by the model)
INITIAL_PATIENTS = 20       # active at episode start (ckpt curriculum stage 4)
SPAWN_TOTAL = 30            # total patients spawned per episode (incl. initial)
SPAWN_INTERVAL = 40         # steps between dynamic spawns

NUM_WIND_ZONES = 12         # hazard rectangles per refresh
NUM_LOW_SIGNAL_ZONES = 8
HAZARD_RECT_MIN = 3         # rectangle edge length range (cells)
HAZARD_RECT_MAX = 8
WIND_APPEAR_INTERVAL = 30
LOW_SIGNAL_APPEAR_INTERVAL = 30
NUM_OBSTACLES = 400

MAX_BATTERY = 100.0
BATTERY_DRAIN_PER_STEP = 0.20
BATTERY_DRAIN_IN_WIND = 2.30
BATTERY_DRAIN_STANDBY = 0.02
LOW_BATTERY_THRESHOLD = 20.0
SAFE_RETURN_BUFFER = 18.0
LOW_SIGNAL_MOVE_SUCCESS = 0.50
WIND_MOVE_SUCCESS = 0.85

# Triage service-debt target delivery rates per class (ckpt metadata)
TRIAGE_TARGET_RATES = {1: 0.5, 2: 0.7, 3: 0.9}

GOAL_REWARD = 100.0
STEP_PENALTY = -0.2
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
CLOSENESS_RADIUS = 4        # matches CEDA-FGCS.py CLOSENESS_RADIUS
STEP_CLIP = 5.0

ACTION_NORTH = 0
ACTION_SOUTH = 1
ACTION_WEST = 2
ACTION_EAST = 3
ACTION_HOVER = 4
ACTION_LAND = 5
MOVE_DELTAS = {
    ACTION_NORTH: (0, -1),
    ACTION_SOUTH: (0, 1),
    ACTION_WEST: (-1, 0),
    ACTION_EAST: (1, 0),
}

# Start grids along the west edge; pads clustered in the south-east corner.
_DEFAULT_START_GRIDS = [(2, 2), (2, 14), (2, 26), (2, 38), (2, 50)]
_DEFAULT_LANDING_ZONE_GRIDS = [(97, 97), (97, 93), (97, 89), (93, 97), (93, 93)]


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class Patient:
    idx: int
    grid_x: float
    grid_y: float
    north_m: float
    east_m: float
    weight: int
    initial_weight: int = 1
    timer: int = INITIAL_PATIENT_TIMER
    initial_timer: int = INITIAL_PATIENT_TIMER
    active: bool = False       # slot spawned this episode
    delivered: bool = False    # resolved by drone delivery
    died: bool = False         # resolved by timer expiry
    steps_elapsed: int = 0
    decay_a: float = 0.0
    decay_b: float = 0.0
    thresh_serious: float = 0.0
    thresh_critical: float = 0.0

    @property
    def pending(self) -> bool:
        return self.active and not self.delivered and not self.died

    @property
    def resolved(self) -> bool:
        return self.delivered or self.died

    # Legacy alias (older metrics code): delivery-by-drone.
    @property
    def actually_delivered(self) -> bool:
        return self.delivered


# ── world environment ─────────────────────────────────────────────────────────

class WorldEnvironment:
    """Tracks dynamic world state for the 5-drone CEDA-FGCS-PX4 mission."""

    DELIVERY_RADIUS_M: float = 2.0   # 1 grid cell radius

    def __init__(self, config: dict) -> None:
        self.config = config
        hazard_cfg = config.get("hazards", {})
        battery_cfg = config.get("battery", {})
        mission_cfg = config.get("mission", {})

        self.num_drones = int(config.get("num_drones", NUM_DRONES))
        self.grid_size = int(config.get("grid", {}).get("size", GRID_SIZE))
        self.num_wind_zones = int(hazard_cfg.get("num_wind_zones", NUM_WIND_ZONES))
        self.num_low_signal_zones = int(hazard_cfg.get("num_low_signal_zones", NUM_LOW_SIGNAL_ZONES))
        self.wind_appear_interval = int(hazard_cfg.get("wind_appear_interval", WIND_APPEAR_INTERVAL))
        self.low_signal_appear_interval = int(
            hazard_cfg.get("low_signal_appear_interval", LOW_SIGNAL_APPEAR_INTERVAL)
        )
        self.num_obstacles = int(config.get("num_obstacles", NUM_OBSTACLES))

        self.initial_battery = float(battery_cfg.get("initial", MAX_BATTERY))
        self.battery_drain_per_step = float(
            battery_cfg.get("drain_per_step", BATTERY_DRAIN_PER_STEP)
        )
        self.battery_drain_in_wind = float(
            battery_cfg.get("drain_in_wind", BATTERY_DRAIN_IN_WIND)
        )
        self.battery_drain_standby = float(
            battery_cfg.get("drain_standby", BATTERY_DRAIN_STANDBY)
        )
        self.low_battery_threshold = float(
            battery_cfg.get("low_battery_threshold", LOW_BATTERY_THRESHOLD)
        )
        self.safe_return_buffer = float(
            battery_cfg.get("safe_return_buffer", SAFE_RETURN_BUFFER)
        )

        self.initial_patients = int(mission_cfg.get("initial_patients", INITIAL_PATIENTS))
        self.spawn_total = int(mission_cfg.get("spawn_total", SPAWN_TOTAL))
        self.spawn_interval = int(mission_cfg.get("spawn_interval", SPAWN_INTERVAL))
        self.spawn_jitter = int(mission_cfg.get("spawn_jitter", 0))
        self.max_steps = int(mission_cfg.get("max_steps", 800))

        self.patients: list[Patient] = []
        self.wind_zones: set[tuple[int, int]] = set()
        self.low_signal_zones: set[tuple[int, int]] = set()
        self.obstacles: set[tuple[int, int]] = set()
        # One entry per drone: (north_m, east_m)
        self.landing_zones: list[tuple[float, float]] = []
        self.start_grids: list[tuple[int, int]] = list(_DEFAULT_START_GRIDS)
        self.agent_grids: list[tuple[int, int]] = list(_DEFAULT_START_GRIDS)
        self.batteries: list[float] = [self.initial_battery] * self.num_drones
        self.landed: list[bool] = [False] * self.num_drones
        self.depleted: list[bool] = [False] * self.num_drones

        # Per-drone bookkeeping the observation builder needs
        self.prev_displacements: list[tuple[int, int]] = [(0, 0)] * self.num_drones
        self.prev_actions: list[int] = [ACTION_HOVER] * self.num_drones
        self.prev_collisions: list[bool] = [False] * self.num_drones
        self.collision_streaks: list[int] = [0] * self.num_drones

        # Obstacle-aware pad maps (per drone), refreshed on reset/hazard change
        self.pad_hop_maps: list[dict[tuple[int, int], int]] = []
        self.pad_hop_max: list[int] = []
        self.pad_energy_maps: list[dict[tuple[int, int], float]] = []

        self._new_patient_timer: int = self.spawn_interval
        self._active_spawn_interval: int = self.spawn_interval
        self._wind_timer: int = self.wind_appear_interval
        self._ls_timer: int = self.low_signal_appear_interval
        self._step_count: int = 0
        self._spawned_count: int = 0
        self._energy_maps_dirty: bool = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Initialise world state from config. Safe to call between episodes."""
        grid_size = self.grid_size
        mpc = float(self.config.get("grid", {}).get("meters_per_cell", METERS_PER_CELL))
        n = self.num_drones
        self.start_grids = [
            tuple(p)
            for p in self.config.get("agent_start_positions", _DEFAULT_START_GRIDS[:n])
        ]
        self.agent_grids = list(self.start_grids)
        self.batteries = [self.initial_battery] * n
        self.landed = [False] * n
        self.depleted = [False] * n
        self.prev_displacements = [(0, 0)] * n
        self.prev_actions = [ACTION_HOVER] * n
        self.prev_collisions = [False] * n
        self.collision_streaks = [0] * n

        # Landing zones (resolve before obstacles so pads stay clear)
        lz_cfgs = self.config.get(
            "landing_zones",
            [{"grid": list(g)} for g in _DEFAULT_LANDING_ZONE_GRIDS[:n]],
        )
        self.landing_zones = []
        pad_grids: list[tuple[int, int]] = []
        for lz in lz_cfgs:
            gx, gy = lz["grid"]
            pad_grids.append((int(gx), int(gy)))
            self.landing_zones.append((-gy * mpc, gx * mpc))   # (north_m, east_m)
        self._pad_grids = pad_grids

        # Obstacles
        obs_cfgs = self.config.get("obstacles")
        if obs_cfgs is not None:
            self.obstacles = {tuple(obs["grid"]) for obs in obs_cfgs}
        else:
            self.obstacles = self._generate_obstacles(pad_grids)

        # Patients — MAX_PATIENTS slots; initial_patients active immediately
        patient_cfgs = self.config.get("patients")
        self.patients = []
        self._spawned_count = 0
        if patient_cfgs is not None:
            for i, pc in enumerate(patient_cfgs[:MAX_PATIENTS]):
                gx, gy = pc["grid"]
                w = pc.get("weight", random.randint(1, MAX_PATIENT_WEIGHT))
                self.patients.append(self._make_patient(i, float(gx), float(gy), w, mpc, active=True))
                self._spawned_count += 1
        else:
            occupied = set(self.obstacles) | set(pad_grids) | set(self.start_grids)
            for i in range(self.initial_patients):
                gx, gy = self._sample_free_cell(grid_size, occupied)
                occupied.add((gx, gy))
                w = random.randint(1, MAX_PATIENT_WEIGHT)
                self.patients.append(self._make_patient(i, float(gx), float(gy), w, mpc, active=True))
                self._spawned_count += 1
        while len(self.patients) < MAX_PATIENTS:
            i = len(self.patients)
            self.patients.append(self._make_patient(i, 0.0, 0.0, 1, mpc, active=False))

        self.wind_zones = set()
        self.low_signal_zones = set()
        self._active_spawn_interval = self.spawn_interval + (
            random.randint(-self.spawn_jitter, self.spawn_jitter) if self.spawn_jitter else 0
        )
        self._new_patient_timer = self._active_spawn_interval
        self._wind_timer = self.wind_appear_interval
        self._ls_timer = self.low_signal_appear_interval
        self._step_count = 0

        # Obstacle-aware hop-distance maps (BFS over free cells) — static per episode
        self.pad_hop_maps = []
        self.pad_hop_max = []
        for pad in pad_grids:
            hop_map = self._bfs_distance_map(pad, grid_size)
            self.pad_hop_maps.append(hop_map)
            self.pad_hop_max.append(max(hop_map.values()) if hop_map else 1)
        # Expected-energy maps depend on hazards — computed lazily on demand
        self._energy_maps_dirty = True

    def _make_patient(
        self, idx: int, gx: float, gy: float, weight: int, mpc: float, *, active: bool
    ) -> Patient:
        decay_a, decay_b, thresh_serious, thresh_critical = self._sample_decay_params(weight)
        return Patient(
            idx=idx,
            grid_x=gx, grid_y=gy,
            north_m=-gy * mpc, east_m=gx * mpc,
            weight=weight, initial_weight=weight,
            timer=INITIAL_PATIENT_TIMER, initial_timer=INITIAL_PATIENT_TIMER,
            active=active,
            steps_elapsed=0,
            decay_a=decay_a, decay_b=decay_b,
            thresh_serious=thresh_serious, thresh_critical=thresh_critical,
        )

    def _sample_free_cell(
        self, grid_size: int, occupied: set[tuple[int, int]]
    ) -> tuple[int, int]:
        for _ in range(10_000):
            x = random.randint(1, grid_size - 2)
            y = random.randint(1, grid_size - 2)
            if (x, y) not in occupied:
                return x, y
        raise RuntimeError("Could not sample a free cell for patient placement")

    def _bfs_distance_map(
        self, pad: tuple[int, int], grid_size: int
    ) -> dict[tuple[int, int], int]:
        """Shortest obstacle-avoiding step count from every reachable cell to pad."""
        dist: dict[tuple[int, int], int] = {pad: 0}
        queue: deque[tuple[int, int]] = deque([pad])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nxt = (cx + dx, cy + dy)
                if (
                    0 <= nxt[0] < grid_size
                    and 0 <= nxt[1] < grid_size
                    and nxt not in self.obstacles
                    and nxt not in dist
                ):
                    dist[nxt] = dist[(cx, cy)] + 1
                    queue.append(nxt)
        return dist

    # ------------------------------------------------------------------
    # Expected-energy safe-return maps (README "Reproducing the energy state")
    # ------------------------------------------------------------------

    def _transition_cost(self, origin: tuple[int, int], dest: tuple[int, int]) -> float:
        p_success = 1.0
        if origin in self.low_signal_zones:
            p_success *= LOW_SIGNAL_MOVE_SUCCESS
        if origin in self.wind_zones:
            p_success *= WIND_MOVE_SUCCESS
        expected_failures = (1.0 - p_success) / p_success
        failed_cost = self.battery_drain_per_step + (
            self.battery_drain_in_wind if origin in self.wind_zones else 0.0
        )
        success_cost = self.battery_drain_per_step + (
            self.battery_drain_in_wind if dest in self.wind_zones else 0.0
        )
        return success_cost + expected_failures * failed_cost

    def _dijkstra_energy_map(self, pad: tuple[int, int]) -> dict[tuple[int, int], float]:
        """Expected route energy from every cell to pad over directed costs.

        Runs Dijkstra outward from the pad; since costs are per directed
        transition origin→destination toward the pad, relaxing from `cur` to
        neighbor `nxt` uses cost(nxt → cur-direction step) = cost(origin=nxt,
        dest=cur)."""
        grid_size = self.grid_size
        dist: dict[tuple[int, int], float] = {pad: 0.0}
        heap: list[tuple[float, tuple[int, int]]] = [(0.0, pad)]
        while heap:
            d, cur = heapq.heappop(heap)
            if d > dist.get(cur, math.inf):
                continue
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nxt = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size):
                    continue
                if nxt in self.obstacles:
                    continue
                nd = d + self._transition_cost(nxt, cur)
                if nd < dist.get(nxt, math.inf):
                    dist[nxt] = nd
                    heapq.heappush(heap, (nd, nxt))
        return dist

    def _refresh_energy_maps(self) -> None:
        self.pad_energy_maps = [
            self._dijkstra_energy_map(pad) for pad in self._pad_grids
        ]
        self._energy_maps_dirty = False

    def safe_return_margin(self, agent_idx: int) -> float:
        """battery − (expected route cost + buffer); README safe-return spec."""
        if self._energy_maps_dirty:
            self._refresh_energy_maps()
        energy_map = self.pad_energy_maps[agent_idx]
        gx, gy = self.agent_grids[agent_idx]
        route_cost = energy_map.get((int(gx), int(gy)))
        if route_cost is None:
            # Unreachable pad: treat as the worst reachable route cost
            route_cost = max(energy_map.values()) if energy_map else MAX_BATTERY
        required = route_cost + self.safe_return_buffer
        return self.batteries[agent_idx] - required

    def return_required(self, agent_idx: int) -> bool:
        return (
            self.batteries[agent_idx] <= self.low_battery_threshold
            or self.safe_return_margin(agent_idx) <= 0.0
        )

    # ------------------------------------------------------------------
    # Mission phase / masks / service debt
    # ------------------------------------------------------------------

    def all_spawned(self) -> bool:
        return self._spawned_count >= self.spawn_total

    def pending_count(self) -> int:
        return sum(1 for p in self.patients if p.pending)

    def landing_phase(self) -> bool:
        """True once every spawned patient is resolved and no more will spawn."""
        return self.all_spawned() and self.pending_count() == 0

    def drone_phase_flag(self, agent_idx: int) -> bool:
        """Drone feature 21: global landing phase or individual energy return."""
        return self.landing_phase() or self.return_required(agent_idx)

    def service_debt_fractions(self) -> list[float]:
        """W1/W2/W3 service-debt fractions (mission features 9–11)."""
        spawned = {1: 0, 2: 0, 3: 0}
        delivered = {1: 0, 2: 0, 3: 0}
        for p in self.patients:
            if not p.active:
                continue
            w = int(p.initial_weight)
            spawned[w] += 1
            if p.delivered:
                delivered[w] += 1
        fractions = []
        for w in (1, 2, 3):
            target = TRIAGE_TARGET_RATES[w] * spawned[w]
            debt = max(0.0, target - delivered[w])
            fractions.append(debt / max(1.0, target))
        return fractions

    def action_mask(self, idx: int) -> list[bool]:
        """Valid actions per CEDA-FGCS-PX4 README mask semantics."""
        if self.landed[idx] or self.depleted[idx]:
            return [False, False, False, False, True, False]
        at_pad = tuple(self.agent_grids[idx]) == self.landing_grid(idx)
        if at_pad and self.drone_phase_flag(idx):
            return [False, False, False, False, False, True]
        return [True, True, True, True, True, False]

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _sample_hazard_rects(self, count: int) -> set[tuple[int, int]]:
        """Hazard rectangles; may overlap obstacles (ckpt metadata)."""
        cells: set[tuple[int, int]] = set()
        grid_size = self.grid_size
        for _ in range(count):
            w = random.randint(HAZARD_RECT_MIN, HAZARD_RECT_MAX)
            h = random.randint(HAZARD_RECT_MIN, HAZARD_RECT_MAX)
            x0 = random.randint(0, max(0, grid_size - w - 1))
            y0 = random.randint(0, max(0, grid_size - h - 1))
            for x in range(x0, x0 + w):
                for y in range(y0, y0 + h):
                    cells.add((x, y))
        return cells

    def refresh_hazards(self) -> None:
        """Refresh wind/low-signal rectangles on the training cadence."""
        self._step_count += 1

        if self._wind_timer > 0:
            self._wind_timer -= 1
        else:
            self.wind_zones = self._sample_hazard_rects(self.num_wind_zones)
            self._wind_timer = self.wind_appear_interval
            self._energy_maps_dirty = True

        if self._ls_timer > 0:
            self._ls_timer -= 1
        else:
            self.low_signal_zones = self._sample_hazard_rects(self.num_low_signal_zones)
            self._ls_timer = self.low_signal_appear_interval
            self._energy_maps_dirty = True

    def step(self, actions: list[int]) -> dict:
        """Apply one training-env transition from the current world state."""
        self.refresh_hazards()
        grid_size = self.grid_size
        n = self.num_drones

        old_positions = list(self.agent_grids)
        old_shaping_dist: list[int] = []
        for i in range(n):
            nearest = self.nearest_undelivered_patient(old_positions[i])
            if nearest is not None:
                old_shaping_dist.append(
                    self.manhattan_distance(old_positions[i], self.patient_grid(nearest))
                )
            else:
                old_shaping_dist.append(
                    self.manhattan_distance(old_positions[i], self.landing_grid(i))
                )

        step_rewards = [STEP_PENALTY] * n
        milestone_rewards = [0.0] * n
        step_data = {
            "wind_entries": [0] * n,
            "low_signal_entries": [0] * n,
            "obstacle_collisions": 0,
            "agent_collisions": 0,
            "actions_executed": list(actions),
            "landing_attempts": [False] * n,
            "landed_this_step": [False] * n,
            "deliveries": [],
            "sim_positions": list(old_positions),
            "rewards": [0.0] * n,
            "done": False,
        }
        collided_this_step = [False] * n

        new_positions: list[tuple[int, int]] = []
        for agent_idx, action in enumerate(actions):
            x, y = self.agent_grids[agent_idx]
            if self.landed[agent_idx] or self.depleted[agent_idx]:
                new_positions.append((x, y))
                continue

            if action == ACTION_LAND:
                step_data["landing_attempts"][agent_idx] = True
                if (x, y) == self.landing_grid(agent_idx):
                    self.landed[agent_idx] = True
                    step_data["landed_this_step"][agent_idx] = True
                    milestone_rewards[agent_idx] += LANDING_REWARD
                else:
                    step_rewards[agent_idx] += LAND_WRONG_PENALTY
                new_positions.append((x, y))
                continue

            if action == ACTION_HOVER:
                new_positions.append((x, y))
                continue

            # Movement failure model (README safe-return spec probabilities)
            p_success = 1.0
            if (x, y) in self.low_signal_zones:
                p_success *= LOW_SIGNAL_MOVE_SUCCESS
            if (x, y) in self.wind_zones:
                p_success *= WIND_MOVE_SUCCESS
            if random.random() > p_success:
                new_x, new_y = x, y
            else:
                dx, dy = MOVE_DELTAS.get(action, (0, 0))
                new_x, new_y = x + dx, y + dy

            if (
                new_x < 0
                or new_x >= grid_size
                or new_y < 0
                or new_y >= grid_size
                or (new_x, new_y) in self.obstacles
            ):
                milestone_rewards[agent_idx] += COLLISION_PENALTY
                step_data["obstacle_collisions"] += 1
                collided_this_step[agent_idx] = True
                new_x, new_y = x, y

            new_positions.append((new_x, new_y))

        # Pairwise agent-collision resolution among airborne drones.
        # Special case: two drones can end up sharing a cell via SITL telemetry
        # drift (impossible in training). Reverting both would deadlock them
        # forever, so an already-overlapping pair lets the lower index move
        # while the higher index yields (matches the model's yield feature).
        active = [i for i in range(n) if not self.landed[i] and not self.depleted[i]]
        colliding: set[int] = set()
        for ai in range(len(active)):
            for bi in range(ai + 1, len(active)):
                a, b = active[ai], active[bi]
                if new_positions[a] != new_positions[b]:
                    continue
                if old_positions[a] == old_positions[b]:
                    # Already overlapping: only the higher index holds still,
                    # and no crash penalty — this is drift recovery.
                    new_positions[b] = old_positions[b]
                else:
                    colliding.add(a)
                    colliding.add(b)
        if colliding:
            step_data["agent_collisions"] += len(colliding) // 2
            for i in colliding:
                milestone_rewards[i] += AGENT_COLLISION_PENALTY
                new_positions[i] = old_positions[i]
                collided_this_step[i] = True

        self.agent_grids = new_positions
        step_data["sim_positions"] = list(new_positions)

        # Closeness penalty between any airborne pair
        for ai in range(len(active)):
            for bi in range(ai + 1, len(active)):
                a, b = active[ai], active[bi]
                dist = self.manhattan_distance(new_positions[a], new_positions[b])
                if dist < CLOSENESS_RADIUS:
                    step_rewards[a] += CLOSENESS_PENALTY
                    step_rewards[b] += CLOSENESS_PENALTY

        # Mission-energy ledger and hazards
        for i in range(n):
            if self.depleted[i]:
                continue
            if self.landed[i]:
                self.batteries[i] = max(
                    0.0, self.batteries[i] - self.battery_drain_standby
                )
                continue

            in_wind = self.agent_grids[i] in self.wind_zones
            if in_wind:
                self.batteries[i] -= self.battery_drain_per_step + self.battery_drain_in_wind
                step_rewards[i] += WIND_PENALTY
                step_data["wind_entries"][i] += 1
            else:
                self.batteries[i] -= self.battery_drain_per_step

            if self.agent_grids[i] in self.low_signal_zones:
                step_rewards[i] += LOW_SIGNAL_PENALTY
                step_data["low_signal_entries"][i] += 1

            if 0 < self.batteries[i] < self.low_battery_threshold:
                step_rewards[i] += LOW_BATTERY_PENALTY

            if self.batteries[i] <= 0:
                # Drone is irreversibly dead but the episode continues
                # (irrecoverable_failure_termination=False in training).
                milestone_rewards[i] += BATTERY_DEPLETION_PENALTY
                self.batteries[i] = 0.0
                self.depleted[i] = True

        self._advance_patients(milestone_rewards)

        # Deliveries
        for i in range(n):
            if self.landed[i] or self.depleted[i]:
                continue
            for p in self.patients:
                if not p.pending:
                    continue
                if self.agent_grids[i] == self.patient_grid(p.idx):
                    timer_ratio = p.timer / MAX_PATIENT_TIMER
                    milestone_rewards[i] += GOAL_REWARD * timer_ratio * p.weight
                    p.delivered = True
                    step_data["deliveries"].append(p.idx)

        # Distance shaping
        for i in range(n):
            if self.landed[i] or self.depleted[i]:
                continue
            nearest = self.nearest_undelivered_patient(self.agent_grids[i])
            if nearest is not None:
                new_dist = self.manhattan_distance(self.agent_grids[i], self.patient_grid(nearest))
            else:
                new_dist = self.manhattan_distance(self.agent_grids[i], self.landing_grid(i))
            step_rewards[i] += SHAPING_FACTOR * (old_shaping_dist[i] - new_dist)

        # Per-drone bookkeeping for the observation builder
        for i in range(n):
            self.prev_displacements[i] = (
                new_positions[i][0] - old_positions[i][0],
                new_positions[i][1] - old_positions[i][1],
            )
            self.prev_actions[i] = actions[i]
            self.prev_collisions[i] = collided_this_step[i]
            if collided_this_step[i]:
                self.collision_streaks[i] = min(self.collision_streaks[i] + 1, 4)
            else:
                self.collision_streaks[i] = 0

        rewards = [
            max(-STEP_CLIP, min(STEP_CLIP, step_rewards[i])) + milestone_rewards[i]
            for i in range(n)
        ]
        done = all(self.landed[i] or self.depleted[i] for i in range(n))

        step_data["rewards"] = rewards
        step_data["done"] = done
        return step_data

    def _advance_patients(self, milestone_rewards: list[float]) -> None:
        """Advance patient timers, acuity decay, and budgeted dynamic spawning."""
        n = self.num_drones
        if not self.all_spawned():
            self._new_patient_timer -= 1
            if self._new_patient_timer <= 0:
                self._active_spawn_interval = self.spawn_interval + (
                    random.randint(-self.spawn_jitter, self.spawn_jitter)
                    if self.spawn_jitter else 0
                )
                self._new_patient_timer = self._active_spawn_interval
                for p in self.patients:
                    if not p.active and not p.resolved:
                        mpc = float(self.config.get("grid", {}).get("meters_per_cell", METERS_PER_CELL))
                        occupied = set(self.obstacles) | {
                            self.patient_grid(q.idx) for q in self.patients if q.active
                        }
                        gx, gy = self._sample_free_cell(self.grid_size, occupied)
                        p.grid_x, p.grid_y = float(gx), float(gy)
                        p.north_m, p.east_m = -gy * mpc, gx * mpc
                        p.active = True
                        p.timer = INITIAL_PATIENT_TIMER
                        p.initial_timer = INITIAL_PATIENT_TIMER
                        p.weight = random.randint(1, MAX_PATIENT_WEIGHT)
                        p.initial_weight = p.weight
                        p.steps_elapsed = 0
                        (
                            p.decay_a,
                            p.decay_b,
                            p.thresh_serious,
                            p.thresh_critical,
                        ) = self._sample_decay_params(p.weight)
                        self._spawned_count += 1
                        break

        for p in self.patients:
            if not p.pending:
                continue
            p.timer -= 1
            if p.timer <= 0:
                for i in range(n):
                    milestone_rewards[i] += PATIENT_DEATH_PENALTY / n
                p.died = True
                p.timer = 0
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
            if not p.pending:
                continue
            dist = self.manhattan_distance(pos, self.patient_grid(p.idx))
            if dist < best_dist:
                best_dist = dist
                best_idx = p.idx
        return best_idx

    def pad_path_distance(self, agent_idx: int, pos: tuple[int, int]) -> tuple[float, int]:
        """(obstacle-aware hop count to pad, map max) — unreachable = map max."""
        hop_map = self.pad_hop_maps[agent_idx]
        max_dist = max(1, self.pad_hop_max[agent_idx])
        return float(hop_map.get((int(pos[0]), int(pos[1])), max_dist)), max_dist

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
            if p.delivered:
                triage_data[f"delivered_w{weight}"] += 1
                triage_data["weighted_delivery_score"] += weight
            elif p.died:
                triage_data[f"died_w{weight}"] += 1
        if triage_data["max_possible_weighted_score"] > 0:
            triage_data["triage_efficiency"] = (
                triage_data["weighted_delivery_score"] / triage_data["max_possible_weighted_score"]
            )
        return triage_data

    def _sample_decay_params(self, initial_weight: int) -> tuple[float, float, float, float]:
        """Logistic acuity progression parameters per initial triage weight."""
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

    def _generate_obstacles(self, pad_grids: list[tuple[int, int]]) -> set[tuple[int, int]]:
        """Random obstacles avoiding starts, pads, and fully-enclosed pockets."""
        grid_size = self.grid_size
        protected = set(self.start_grids) | set(pad_grids)
        obstacles: set[tuple[int, int]] = set()

        while len(obstacles) < self.num_obstacles:
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
