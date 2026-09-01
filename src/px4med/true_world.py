"""Wraps the collaborator's actual training Environment as the SITL world.

`models/CEDA-FGCS-new.py` is the real training script (environment + rewards +
curriculum). Using its `Environment` directly eliminates every reconstruction
gap: observations come from his `get_state()`, transitions/rewards/masks from
his `step()`. This adapter exposes the same duck-typed interface the
Coordinator, baselines, and experiments layer already consume from the old
`WorldEnvironment`.

Verified 2026-08-29: policy in this env scores 48/50 delivered, triage 0.96,
mission_success at step 335 (vs 1-8/30 in the reconstructed world).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Optional

_TRAIN_PATH = Path(__file__).resolve().parents[2] / "models" / "CEDA-FGCS-new.py"
_module = None


def load_training_module():
    global _module
    if _module is None:
        os.environ.setdefault("CEDA_HEADLESS", "1")
        spec = spec_from_file_location("ceda_train", _TRAIN_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load training module from {_TRAIN_PATH}")
        mod = module_from_spec(spec)
        sys.modules["ceda_train"] = mod
        spec.loader.exec_module(mod)
        _module = mod
    return _module


@dataclass
class PatientView:
    """Lightweight patient view for baselines and result tallies."""
    idx: int
    grid_x: float
    grid_y: float
    weight: int
    initial_weight: int
    timer: int
    active: bool
    pending: bool
    delivered: bool
    died: bool


class TrueWorld:
    """Duck-typed WorldEnvironment replacement backed by the training env."""

    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        m = load_training_module()
        self._m = m
        mission_cfg = self.config.get("mission", {})
        self.max_steps = int(mission_cfg.get("max_steps", m.MAX_STEPS))
        grace = int(mission_cfg.get("landing_grace_steps", m.DEFAULT_LANDING_GRACE_STEPS))
        stage = int(mission_cfg.get("curriculum_stage", len(m.CURRICULUM_STAGES) - 1))
        self._stage = stage
        self.env = m.Environment(
            fixed_layout=False,
            episode_max_steps=self.max_steps,
            landing_grace_steps=grace,
        )
        self.num_drones = m.NUM_AGENTS
        self.grid_size = m.GRID_SIZE
        # Optional world overrides (hazard/battery sweeps). Applied post-reset
        # in reset(); keep keys minimal and explicit.
        self._battery_initial = self.config.get("battery", {}).get("initial")
        # Hazard density: the env scales its rectangle counts by
        # hazard_fraction (NUM_WIND_ZONE_RECTANGLES=6,
        # NUM_LOW_SIGNAL_ZONE_RECTANGLES=5) and re-reads the attribute on every
        # periodic regeneration, so setting it post-reset holds for the whole
        # episode. Training used 0.5 at stage 0 and 1.0 at stages 1-2, so
        # anything above 1.0 is explicitly off-distribution extrapolation.
        self._hazard_fraction = self.config.get("hazard", {}).get("fraction")

    # ── lifecycle ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.env.reset(curriculum_stage=self._stage)
        if self._hazard_fraction is not None:
            # Force one regeneration so the new density applies from step 0;
            # the forced call also re-arms the periodic timers.
            self.env.hazard_fraction = float(self._hazard_fraction)
            self.env.update_wind_zones(force=True)
            self.env.update_low_signal_zones(force=True)
        if self._battery_initial is not None:
            self.env.batteries = [float(self._battery_initial)] * self.num_drones

    @property
    def hazard_rectangle_counts(self) -> tuple[int, int]:
        """(wind, low_signal) rectangle counts — for provenance/logging."""
        return (
            len(getattr(self.env, "wind_rectangles", []) or []),
            len(getattr(self.env, "low_signal_rectangles", []) or []),
        )

    def observation(self) -> dict:
        """The model's observation — straight from the training env."""
        return self.env.get_state()

    def step(self, actions: list[int]) -> dict:
        _, rewards, done, sd = self.env.step(list(actions))
        n = self.num_drones
        attempts, landed, wrong_land = _landing_outcomes(sd.get("landing_events"), n)
        return {
            "rewards": list(rewards),
            "done": bool(done),
            "sim_positions": [tuple(p) for p in self.env.agents],
            "deliveries": _event_indices(
                sd.get("patient_delivery_events"), ("patient", "patient_idx", "index")
            ),
            "landing_attempts": attempts,
            "landed_this_step": landed,
            "wind_entries": _per_agent_int(sd.get("wind_entries"), n),
            "low_signal_entries": _per_agent_int(sd.get("low_signal_entries"), n),
            "obstacle_collisions": _as_int(sd.get("obstacle_collisions")),
            "agent_collisions": _as_int(sd.get("agent_collisions")),
            "wrong_land_count": wrong_land,
            "raw_step_data": sd,
        }

    # ── state the coordinator syncs/reads ─────────────────────────────────

    @property
    def agent_grids(self) -> list[tuple[int, int]]:
        return [tuple(p) for p in self.env.agents]

    @agent_grids.setter
    def agent_grids(self, grids: list[tuple[int, int]]) -> None:
        g_max = self.grid_size - 1
        self.env.agents = [
            (min(max(int(x), 0), g_max), min(max(int(y), 0), g_max))
            for x, y in grids
        ]

    @property
    def landed(self) -> list[bool]:
        return list(self.env.landed)

    @property
    def depleted(self) -> list[bool]:
        return [
            bool(self.env.battery_depleted[i]) or bool(self.env.drone_died[i])
            for i in range(self.num_drones)
        ]

    @property
    def batteries(self) -> list[float]:
        return [float(b) for b in self.env.batteries]

    @property
    def start_grids(self) -> list[tuple[int, int]]:
        return [tuple(p) for p in self.env.start_positions]

    def landing_grid(self, agent_idx: int) -> tuple[int, int]:
        return tuple(self.env.landing_zones[agent_idx])

    @property
    def obstacles(self) -> set:
        return self.env.obstacles

    # ── mission phase / helpers (baselines + logging) ─────────────────────

    def pending_count(self) -> int:
        return sum(1 for p in self.patients if p.pending)

    def all_spawned(self) -> bool:
        return bool(self.env.all_patients_spawned())

    def landing_phase(self) -> bool:
        return bool(self.env.all_patients_resolved()) and self.all_spawned()

    def return_required(self, agent_idx: int) -> bool:
        return bool(self.env.energy_return_required(agent_idx))

    def action_mask(self, idx: int) -> list[bool]:
        masks = self.env.get_state()["action_masks"]
        return [bool(v) for v in masks[idx]]

    @property
    def patients(self) -> list[PatientView]:
        e = self.env
        views: list[PatientView] = []
        for i in range(self._m.MAX_PATIENTS):
            if not e.patient_active[i]:
                continue
            pending = (
                bool(e.patient_active[i])
                and not bool(e.patients_delivered[i])
                and not bool(e.patients_died[i])
            )
            x, y = e.patient_positions[i]
            views.append(PatientView(
                idx=i, grid_x=float(x), grid_y=float(y),
                weight=int(e.patient_weights[i]),
                initial_weight=int(e.initial_patient_weights[i]),
                timer=int(e.patient_timers[i]),
                active=True,
                pending=pending,
                # patients_delivered doubles as a "resolved" flag upstream;
                # true delivery excludes deaths.
                delivered=bool(e.patients_delivered[i]) and not bool(e.patients_died[i]),
                died=bool(e.patients_died[i]),
            ))
        return views

    def patient_grid(self, patient_idx: int) -> tuple[int, int]:
        x, y = self.env.patient_positions[patient_idx]
        return int(x), int(y)

    def nearest_undelivered_patient(self, pos) -> Optional[int]:
        result = self.env.nearest_undelivered_patient(tuple(pos))
        return None if result is None else int(result)

    def manhattan_distance(self, a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ── geometry ───────────────────────────────────────────────────────────

    def _mpc(self) -> float:
        return float(self.config.get("grid", {}).get("meters_per_cell", 2.0))

    def get_grid_pos(self, north_m: float, east_m: float) -> tuple[int, int]:
        mpc = self._mpc()
        return round(east_m / mpc), round(-north_m / mpc)

    def grid_to_ned(self, grid_x: int, grid_y: int) -> tuple[float, float]:
        mpc = self._mpc()
        return -grid_y * mpc, grid_x * mpc

    # ── outcome metrics ────────────────────────────────────────────────────

    def mission_metrics(self) -> dict[str, Any]:
        return dict(self.env.mission_outcome_metrics())

    def triage_summary(self) -> dict[str, float]:
        metrics = self.mission_metrics()
        return {"triage_efficiency": float(metrics.get("triage_efficiency", 0.0))}


# ── step_data coercion helpers ────────────────────────────────────────────────

def _event_indices(value, keys: tuple[str, ...]) -> list[int]:
    out: list[int] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            for key in keys:
                if key in item:
                    out.append(int(item[key]))
                    break
        elif isinstance(item, (int, float)):
            out.append(int(item))
    return out


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_int(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return int(sum(value))
    return int(value)


def _per_agent_int(value, n: int) -> list[int]:
    if isinstance(value, (list, tuple)) and len(value) == n:
        return [int(v) for v in value]
    return [0] * n


def _landing_outcomes(events, n: int) -> tuple[list[bool], list[bool], int]:
    """Split the env's landing_events into per-agent attempts/successes.

    Every ACTION_LAND emits exactly one event (including post-death land
    actions, which arrive with successful=False), so `land_actions` is just
    len(events) and carries no success information — the events do.
    """
    attempts = [False] * n
    landed = [False] * n
    wrong_land = 0
    for event in _as_list(events):
        if not isinstance(event, dict):
            continue
        successful = bool(event.get("successful"))
        if not successful:
            wrong_land += 1
        idx = event.get("agent")
        if isinstance(idx, int) and 0 <= idx < n:
            attempts[idx] = True
            if successful:
                landed[idx] = True
    return attempts, landed, wrong_land


# ── hazard / energy-discipline accumulation ───────────────────────────────────
#
# The env emits these counters PER STEP only; nothing in `Environment`
# accumulates them. The training script does it in
# `TrainingEpisodeTracker.update()` and turns them into rates in `.finish()`.
# We mirror both so our numbers are directly comparable to the curriculum
# gates (stage 2 demands wind/low-signal avoidance >= 0.98).
#
# This is what makes "0 hazard entries" a *result* rather than an absence of
# evidence: the opportunity counts are the denominator proving hazards were
# encounterable at all.

_HAZARD_SCALAR_KEYS = (
    "wind_avoidance_opportunities",
    "wind_hazard_selections",
    "wind_dominated_hazard_selections",
    "wind_shortcut_hazard_selections",
    "wind_command_attempts",
    "low_signal_avoidance_opportunities",
    "low_signal_hazard_selections",
    "low_signal_dominated_hazard_selections",
    "low_signal_shortcut_hazard_selections",
    "low_signal_command_attempts",
)

_HAZARD_PER_AGENT_KEYS = (
    "wind_exposure_steps",
    "wind_failures",
    "low_signal_exposure_steps",
    "low_signal_failures",
    "reserve_violation_flags",
    "forced_terminal_landing_actions",
)

HAZARD_KEYS = _HAZARD_SCALAR_KEYS + _HAZARD_PER_AGENT_KEYS


def new_hazard_totals() -> dict[str, int]:
    """A zeroed accumulator. Plain dict so it stays JSON-serialisable."""
    return {key: 0 for key in HAZARD_KEYS}


def accumulate_hazard(totals: dict[str, int], step_data: Optional[dict]) -> None:
    """Fold one step's counters in (mirrors TrainingEpisodeTracker.update)."""
    if not step_data:
        return
    for key in _HAZARD_SCALAR_KEYS:
        value = step_data.get(key)
        if isinstance(value, (int, float)):
            totals[key] = totals.get(key, 0) + int(value)
    for key in _HAZARD_PER_AGENT_KEYS:
        value = step_data.get(key)
        if isinstance(value, (list, tuple)):
            totals[key] = totals.get(key, 0) + int(sum(value))
        elif isinstance(value, (int, float)):
            totals[key] = totals.get(key, 0) + int(value)


def hazard_fields(totals: Optional[dict[str, int]]) -> dict[str, float]:
    """Episode-level hazard columns, using the training script's formulas.

    Rates follow `1 - selections / max(1, opportunities)`, so an episode with
    no opportunities scores a vacuous 1.0 exactly as it does in training —
    always read a rate alongside its `*_opportunities` count.
    """
    t = dict(new_hazard_totals())
    if totals:
        t.update({k: int(v) for k, v in totals.items() if k in t})

    def rate(selections: str, opportunities: str) -> float:
        return round(1.0 - t[selections] / max(1, t[opportunities]), 4)

    return {
        "wind_avoidance_opportunities": t["wind_avoidance_opportunities"],
        "wind_hazard_selections": t["wind_hazard_selections"],
        "wind_avoidance_rate": rate(
            "wind_hazard_selections", "wind_avoidance_opportunities"),
        "wind_dominated_avoidance_rate": rate(
            "wind_dominated_hazard_selections", "wind_avoidance_opportunities"),
        "wind_exposure_steps": t["wind_exposure_steps"],
        "wind_movement_failures": t["wind_failures"],
        "low_signal_avoidance_opportunities": t[
            "low_signal_avoidance_opportunities"],
        "low_signal_hazard_selections": t["low_signal_hazard_selections"],
        "low_signal_avoidance_rate": rate(
            "low_signal_hazard_selections", "low_signal_avoidance_opportunities"),
        "low_signal_dominated_avoidance_rate": rate(
            "low_signal_dominated_hazard_selections",
            "low_signal_avoidance_opportunities"),
        "low_signal_exposure_steps": t["low_signal_exposure_steps"],
        "low_signal_movement_failures": t["low_signal_failures"],
        "reserve_violations": t["reserve_violation_flags"],
        "forced_terminal_landings": t["forced_terminal_landing_actions"],
    }
