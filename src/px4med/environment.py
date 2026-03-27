"""World state manager — patient timers, delivery detection, hazard zones.

Mirrors the dynamic world state from AneeshMARL5.py Environment, driven by
config + clock ticks rather than the training env's RL step loop. All
positions are stored in both grid space and NED metres so state.py can
read either without re-conversion.
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

# Fixed-layout defaults matching AneeshMARL5.py Environment(fixed_layout=True)
_DEFAULT_PATIENT_GRIDS = [
    (13, 13), (13, 1), (25, 25), (25, 1),
    (35, 10), (10, 35), (40, 30), (30, 40),
]
_DEFAULT_LANDING_ZONE_GRIDS = [(48, 48), (48, 45)]


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


# ── world environment ─────────────────────────────────────────────────────────

class WorldEnvironment:
    """Tracks dynamic world state independent of PX4 SITL."""

    DELIVERY_RADIUS_M: float = 2.0   # 1 grid cell radius

    def __init__(self, config: dict) -> None:
        self.config = config
        self.patients: list[Patient] = []
        self.wind_zones: set[tuple[int, int]] = set()
        self.low_signal_zones: set[tuple[int, int]] = set()
        # One entry per drone: (north_m, east_m)
        self.landing_zones: list[tuple[float, float]] = []

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
            self.patients.append(Patient(
                idx=i,
                grid_x=float(gx), grid_y=float(gy),
                north_m=-gy * mpc, east_m=gx * mpc,
                weight=w, timer=MAX_PATIENT_TIMER,
                active=(i < NUM_PATIENTS),
                delivered=False, actually_delivered=False,
            ))
        while len(self.patients) < MAX_PATIENTS:
            i = len(self.patients)
            self.patients.append(Patient(
                idx=i, grid_x=0.0, grid_y=0.0,
                north_m=0.0, east_m=0.0,
                weight=1, timer=MAX_PATIENT_TIMER,
                active=False, delivered=False, actually_delivered=False,
            ))

        self.wind_zones = set()
        self.low_signal_zones = set()
        self._new_patient_timer = NEW_PATIENT_SPAWN_INTERVAL
        self._wind_timer = WIND_APPEAR_INTERVAL
        self._ls_timer = LOW_SIGNAL_APPEAR_INTERVAL
        self._step_count = 0

    # ------------------------------------------------------------------
    # Per-step update (call once per control loop tick)
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance patient timers, spawn new patients, refresh hazard zones."""
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

        # Spawn new patient
        self._new_patient_timer -= 1
        if self._new_patient_timer <= 0:
            self._new_patient_timer = NEW_PATIENT_SPAWN_INTERVAL
            for p in self.patients:
                if not p.active and not p.delivered:
                    p.active = True
                    p.timer = MAX_PATIENT_TIMER
                    p.weight = random.randint(1, MAX_PATIENT_WEIGHT)
                    break

        # Advance patient timers
        for p in self.patients:
            if not p.active or p.delivered:
                continue
            p.timer -= 1
            if p.timer <= 0:
                p.delivered = True   # timed out (not actually_delivered)

    # ------------------------------------------------------------------
    # Delivery detection
    # ------------------------------------------------------------------

    def check_delivery(self, drone_idx: int, north_m: float, east_m: float) -> Optional[int]:
        """Return patient idx if the drone is within DELIVERY_RADIUS_M, else None."""
        for p in self.patients:
            if not p.active or p.delivered:
                continue
            dist = math.sqrt((north_m - p.north_m) ** 2 + (east_m - p.east_m) ** 2)
            if dist <= self.DELIVERY_RADIUS_M:
                p.delivered = True
                p.actually_delivered = True
                return p.idx
        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_grid_pos(self, north_m: float, east_m: float) -> tuple[int, int]:
        """Convert NED metres to the nearest integer grid cell (x, y)."""
        mpc = float(self.config.get("grid", {}).get("meters_per_cell", METERS_PER_CELL))
        return round(east_m / mpc), round(-north_m / mpc)
