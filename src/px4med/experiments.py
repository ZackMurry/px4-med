"""Experiment suites, result schemas, and episode backends for the 5-drone
CEDA-FGCS-PX4 mission.

Rewritten 2026-08-29 for the final model; the previous 2-drone version lives
in git history. Provides:
  - SuiteDef / ScenarioDef and build_default_suites()
  - EpisodeResult / StepResult dataclasses with CSV writers
  - summarize_results() with mean + 95% CI per (suite, scenario, policy)
  - build_policy_controller() for learned + heuristic policies
  - run_offline_episode() — world-model-only backend (no SITL)
  - run_sitl_episode() — drives a live, already-booted SITL fleet
  - InMemoryMetricsCollector for workers
"""
from __future__ import annotations

import csv
import logging
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .baselines import make_baseline
from .coordinator import Coordinator
from .episode_budget import (
    LANDING_GRACE_STEPS,
    MISSION_MAX_STEPS,
)
from .true_world import (
    TrueWorld,
    accumulate_hazard,
    hazard_fields,
    new_hazard_totals,
)
from .fgcs_policy import DEFAULT_WEIGHTS_PATH, FGCSPolicy
from .metrics import StepRecord

logger = logging.getLogger(__name__)

NUM_DRONES = 5



# ── suite definitions ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScenarioDef:
    name: str
    label: str
    world: dict
    episodes: int
    max_steps: int
    x_value: float | None = None
    action_delay_steps: int = 0


@dataclass(frozen=True)
class SuiteDef:
    name: str
    title: str
    policies: tuple[str, ...]
    scenarios: tuple[ScenarioDef, ...]


def build_default_suites(
    episodes_override: int | None = None,
    max_steps_override: int | None = None,
) -> list[SuiteDef]:
    def eps(default: int) -> int:
        return episodes_override if episodes_override is not None else default

    def steps(default: int = MISSION_MAX_STEPS) -> int:
        return max_steps_override if max_steps_override is not None else default

    def loop_cap() -> int:
        """Driver-loop bound: let the env's own termination decide."""
        return steps() + LANDING_GRACE_STEPS

    def mission(**extra) -> dict:
        cfg = {
            "max_steps": steps(),
            "landing_grace_steps": LANDING_GRACE_STEPS,
        }
        cfg.update(extra)
        return cfg

    nominal_world: dict = {"mission": mission()}

    return [
        SuiteDef(
            name="baseline_comparison",
            title="Nominal Baseline Comparison",
            policies=("learned", "priority_path", "nearest_path", "random"),
            scenarios=(
                ScenarioDef(
                    name="nominal", label="Nominal",
                    world=nominal_world, episodes=eps(20), max_steps=loop_cap(),
                ),
            ),
        ),
        SuiteDef(
            name="latency_sweep",
            title="Command Latency Robustness",
            policies=("learned",),
            scenarios=tuple(
                ScenarioDef(
                    name=f"delay_{delay}", label=f"{delay} step",
                    x_value=float(delay),
                    world=nominal_world, episodes=eps(4),
                    max_steps=loop_cap(),
                    # Coordinator queues each drone's action this many steps
                    # before dispatch, i.e. the policy acts on state that is
                    # `delay` steps stale — the shape of a real comms lag.
                    # Training had zero delay, so all of these are
                    # off-distribution robustness checks.
                    action_delay_steps=delay,
                )
                for delay in (1, 2, 4)
            ),
        ),
        SuiteDef(
            name="hazard_sweep",
            title="Hazard Density Sweep",
            policies=("learned", "nearest_path"),
            scenarios=tuple(
                ScenarioDef(
                    name=f"hazard_{int(round(fraction * 100)):03d}",
                    label=f"{fraction:g}x",
                    x_value=fraction,
                    episodes=eps(4), max_steps=loop_cap(),
                    world={
                        "mission": mission(),
                        "hazard": {"fraction": fraction},
                    },
                )
                # 0.5 and 1.0 are in-distribution (training used 0.5 at stage 0,
                # 1.0 at stages 1-2). 1.5 and 2.0 are deliberate off-distribution
                # extrapolation and MUST be labelled as such in the paper.
                for fraction in (0.5, 1.0, 1.5, 2.0)
            ),
        ),
        SuiteDef(
            name="battery_sweep",
            title="Battery Stress Sweep",
            policies=("learned", "priority_path"),
            scenarios=(
                ScenarioDef(
                    name="battery_60", label="60", x_value=60,
                    episodes=eps(10), max_steps=loop_cap(),
                    world={
                        "mission": mission(),
                        "battery": {"initial": 60},
                    },
                ),
            ),
        ),
    ]


def suite_lookup() -> dict[tuple[str, str], tuple[SuiteDef, ScenarioDef]]:
    lookup: dict[tuple[str, str], tuple[SuiteDef, ScenarioDef]] = {}
    for suite in build_default_suites():
        for scenario in suite.scenarios:
            lookup[(suite.name, scenario.name)] = (suite, scenario)
    return lookup


# ── result schemas ────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    backend: str
    suite: str
    scenario: str
    policy: str
    episode: int
    seed: int
    steps: int
    duration_s: float
    patients_spawned: int
    patients_delivered: int
    patients_died: int
    patients_unresolved: int
    delivery_rate: float
    mortality_rate: float
    triage_efficiency: float
    mission_success: float          # all patients resolved AND all drones landed
    drones_landed: int
    drones_depleted: int
    all_landed: float
    wrong_land_attempts: int
    total_reward: float
    sim_battery_min: float
    sim_battery_mean: float
    wind_entries: int
    low_signal_entries: int
    obstacle_collisions: int
    agent_collisions: int
    min_inter_drone_distance: float
    mean_tracking_error_m: float
    max_tracking_error_m: float
    time_to_first_delivery: float   # steps; -1 if none
    delivered_w1: int
    delivered_w2: int
    delivered_w3: int
    died_w1: int
    died_w2: int
    died_w3: int

    # ── hazard / energy discipline (see true_world.hazard_fields) ──────────
    # Defaults keep older result JSON loadable on resume.
    wind_avoidance_opportunities: int = 0
    wind_hazard_selections: int = 0
    wind_avoidance_rate: float = 1.0
    wind_dominated_avoidance_rate: float = 1.0
    wind_exposure_steps: int = 0
    wind_movement_failures: int = 0
    low_signal_avoidance_opportunities: int = 0
    low_signal_hazard_selections: int = 0
    low_signal_avoidance_rate: float = 1.0
    low_signal_dominated_avoidance_rate: float = 1.0
    low_signal_exposure_steps: int = 0
    low_signal_movement_failures: int = 0
    reserve_violations: int = 0
    forced_terminal_landings: int = 0

    # ── promoted from the env's own mission_outcome_metrics() ──────────────
    rescue_quality: float = 0.0
    acuity_priority_score: float = 0.0
    priority_fairness_attainment: float = 0.0
    priority_target_attainment: float = 0.0
    class_delivery_jain_fairness: float = 0.0
    delivery_workload_jain_fairness: float = 0.0
    minimum_class_delivery_rate: float = 0.0
    class_delivery_rate_gap: float = 0.0
    triage_delivery_ordering_score: float = 0.0
    triage_response_time_ordering_score: float = 0.0
    mean_delivered_response_time: float = 0.0
    mean_response_time_w1: float = 0.0
    mean_response_time_w2: float = 0.0
    mean_response_time_w3: float = 0.0
    mean_response_ratio_w1: float = 0.0
    mean_response_ratio_w2: float = 0.0
    mean_response_ratio_w3: float = 0.0
    p90_response_time_w1: float = 0.0
    p90_response_time_w2: float = 0.0
    p90_response_time_w3: float = 0.0
    delivery_rate_w1: float = 0.0
    delivery_rate_w2: float = 0.0
    delivery_rate_w3: float = 0.0
    high_vs_low_response_advantage: float = 0.0


@dataclass
class StepResult:
    backend: str
    suite: str
    scenario: str
    policy: str
    episode: int
    step: int
    actions: str                    # semicolon-joined
    positions: str                  # "x:y" semicolon-joined (world grid)
    pending_patients: int
    deliveries: str                 # patient idx semicolon-joined
    reward_sum: float
    landed_count: int
    depleted_count: int
    tracking_error_m: float         # mean over airborne drones (SITL only)


# ── CSV / summary helpers ─────────────────────────────────────────────────────

def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_episode_csv(path: Path, results: Sequence[EpisodeResult]) -> None:
    write_csv(path, [asdict(r) for r in results])


def write_step_csv(path: Path, results: Sequence[StepResult]) -> None:
    write_csv(path, [asdict(r) for r in results])


_SUMMARY_FIELDS = [
    "delivery_rate", "mortality_rate", "triage_efficiency", "mission_success",
    "all_landed", "drones_landed", "drones_depleted", "wrong_land_attempts",
    "total_reward", "sim_battery_min", "obstacle_collisions", "agent_collisions",
    "min_inter_drone_distance", "mean_tracking_error_m", "time_to_first_delivery",
    "steps", "duration_s",
    # hazard / energy discipline
    "wind_avoidance_opportunities", "wind_hazard_selections",
    "wind_avoidance_rate", "wind_exposure_steps", "wind_movement_failures",
    "low_signal_avoidance_opportunities", "low_signal_hazard_selections",
    "low_signal_avoidance_rate", "low_signal_exposure_steps",
    "low_signal_movement_failures", "reserve_violations",
    "forced_terminal_landings",
    # triage quality / fairness / response times
    "rescue_quality", "acuity_priority_score", "priority_fairness_attainment",
    "priority_target_attainment", "class_delivery_jain_fairness",
    "delivery_workload_jain_fairness", "minimum_class_delivery_rate",
    "class_delivery_rate_gap", "triage_delivery_ordering_score",
    "triage_response_time_ordering_score", "mean_delivered_response_time",
    "mean_response_time_w1", "mean_response_time_w2", "mean_response_time_w3",
    "mean_response_ratio_w1", "mean_response_ratio_w2", "mean_response_ratio_w3",
    "p90_response_time_w1", "p90_response_time_w2", "p90_response_time_w3",
    "delivery_rate_w1", "delivery_rate_w2", "delivery_rate_w3",
    "high_vs_low_response_advantage",
]


def summarize_results(results: Sequence[EpisodeResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[EpisodeResult]] = {}
    for r in results:
        groups.setdefault((r.suite, r.scenario, r.policy), []).append(r)

    rows: list[dict[str, Any]] = []
    for (suite, scenario, policy), members in sorted(groups.items()):
        row: dict[str, Any] = {
            "suite": suite, "scenario": scenario, "policy": policy,
            "episodes": len(members),
        }
        for field_name in _SUMMARY_FIELDS:
            values = [float(getattr(m, field_name)) for m in members]
            mean = statistics.fmean(values)
            if len(values) > 1:
                ci = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
            else:
                ci = 0.0
            row[f"{field_name}_mean"] = round(mean, 4)
            row[f"{field_name}_ci95"] = round(ci, 4)
        rows.append(row)
    return rows


# ── policy factory ────────────────────────────────────────────────────────────

def build_policy_controller(
    policy_name: str,
    world,
    seed: int,
    model_path: Optional[Path] = None,
    learned_policy: Optional[FGCSPolicy] = None,
    device: str = "cpu",
):
    if policy_name == "learned":
        if learned_policy is not None:
            return learned_policy
        return FGCSPolicy(
            weights_path=model_path or DEFAULT_WEIGHTS_PATH, device=device
        )
    return make_baseline(policy_name, seed, world)


# ── shared tally helpers ──────────────────────────────────────────────────────

# Scalars lifted out of the env's mission_outcome_metrics() into EpisodeResult
# so they aggregate with mean+CI95. The full 65-key dict is dumped per job by
# the runner for anything not promoted here.
_PROMOTED_MISSION_METRICS = (
    "rescue_quality",
    "acuity_priority_score",
    "priority_fairness_attainment",
    "priority_target_attainment",
    "class_delivery_jain_fairness",
    "delivery_workload_jain_fairness",
    "minimum_class_delivery_rate",
    "class_delivery_rate_gap",
    "triage_delivery_ordering_score",
    "triage_response_time_ordering_score",
    "mean_delivered_response_time",
    "mean_response_time_w1",
    "mean_response_time_w2",
    "mean_response_time_w3",
    "mean_response_ratio_w1",
    "mean_response_ratio_w2",
    "mean_response_ratio_w3",
    "p90_response_time_w1",
    "p90_response_time_w2",
    "p90_response_time_w3",
    "delivery_rate_w1",
    "delivery_rate_w2",
    "delivery_rate_w3",
    "high_vs_low_response_advantage",
)


def _triage_counts(world) -> dict[str, int]:
    counts = {f"{kind}_w{w}": 0 for kind in ("delivered", "died") for w in (1, 2, 3)}
    for p in world.patients:
        if not p.active:
            continue
        w = int(p.initial_weight)
        if p.delivered:
            counts[f"delivered_w{w}"] += 1
        elif p.died:
            counts[f"died_w{w}"] += 1
    return counts


def _episode_result_from_world(
    *,
    backend: str,
    suite: str,
    scenario: str,
    policy: str,
    episode: int,
    seed: int,
    world,
    steps: int,
    duration_s: float,
    total_reward: float,
    wrong_land_attempts: int,
    wind_entries: int,
    low_signal_entries: int,
    obstacle_collisions: int,
    agent_collisions: int,
    min_inter_drone_distance: float,
    mean_tracking_error_m: float,
    max_tracking_error_m: float,
    time_to_first_delivery: float,
    hazard_totals: Optional[dict[str, int]] = None,
) -> EpisodeResult:
    spawned = sum(1 for p in world.patients if p.active)
    delivered = sum(1 for p in world.patients if p.delivered)
    died = sum(1 for p in world.patients if p.died)
    unresolved = world.pending_count()
    triage = world.triage_summary()
    counts = _triage_counts(world)
    all_landed = all(world.landed)
    mission_success = 1.0 if (unresolved == 0 and world.all_spawned() and all_landed) else 0.0

    # The env's own outcome dict is authoritative for triage/fairness/response
    # metrics; promote the paper-relevant scalars so they land in summary.csv
    # with CI95 alongside everything else.
    mission = world.mission_metrics()
    promoted = {
        key: float(mission.get(key, 0.0)) for key in _PROMOTED_MISSION_METRICS
    }

    return EpisodeResult(
        backend=backend, suite=suite, scenario=scenario, policy=policy,
        episode=episode, seed=seed, steps=steps, duration_s=round(duration_s, 1),
        patients_spawned=spawned, patients_delivered=delivered,
        patients_died=died, patients_unresolved=unresolved,
        delivery_rate=delivered / max(1, spawned),
        mortality_rate=died / max(1, spawned),
        triage_efficiency=float(triage["triage_efficiency"]),
        mission_success=mission_success,
        drones_landed=sum(world.landed),
        drones_depleted=sum(world.depleted),
        all_landed=1.0 if all_landed else 0.0,
        wrong_land_attempts=wrong_land_attempts,
        total_reward=round(total_reward, 2),
        sim_battery_min=round(min(world.batteries), 2),
        sim_battery_mean=round(sum(world.batteries) / len(world.batteries), 2),
        wind_entries=wind_entries,
        low_signal_entries=low_signal_entries,
        obstacle_collisions=obstacle_collisions,
        agent_collisions=agent_collisions,
        min_inter_drone_distance=min_inter_drone_distance,
        mean_tracking_error_m=round(mean_tracking_error_m, 3),
        max_tracking_error_m=round(max_tracking_error_m, 3),
        time_to_first_delivery=time_to_first_delivery,
        **counts,
        **hazard_fields(hazard_totals),
        **promoted,
    )


# ── offline backend ───────────────────────────────────────────────────────────

def run_offline_episode(
    suite: SuiteDef,
    scenario: ScenarioDef,
    policy_name: str,
    seed: int,
    episode_idx: int,
    learned_policy: Optional[FGCSPolicy] = None,
    model_path: Optional[Path] = None,
) -> tuple[EpisodeResult, list[StepResult]]:
    random.seed(seed)
    world = TrueWorld(dict(scenario.world))
    world.reset()
    controller = build_policy_controller(
        policy_name, world, seed,
        model_path=model_path, learned_policy=learned_policy,
    )

    start = time.time()
    total_reward = 0.0
    wrong_land = 0
    wind = ls = obs_col = agent_col = 0
    min_dist = math.inf
    first_delivery = -1.0
    step_results: list[StepResult] = []
    hazard_totals = new_hazard_totals()
    step = 0

    for step in range(scenario.max_steps):
        if policy_name == "learned":
            actions = controller.select_actions(world.observation())
        else:
            actions = controller.select_actions(None)
        data = world.step(actions)
        total_reward += sum(data["rewards"])
        wrong_land += sum(
            1 for i in range(world.num_drones)
            if data["landing_attempts"][i] and not data["landed_this_step"][i]
        )
        wind += sum(data["wind_entries"])
        ls += sum(data["low_signal_entries"])
        obs_col += data["obstacle_collisions"]
        agent_col += data["agent_collisions"]
        accumulate_hazard(hazard_totals, data.get("raw_step_data"))
        if data["deliveries"] and first_delivery < 0:
            first_delivery = float(step)
        airborne = [
            i for i in range(world.num_drones)
            if not world.landed[i] and not world.depleted[i]
        ]
        for ai in range(len(airborne)):
            for bi in range(ai + 1, len(airborne)):
                d = world.manhattan_distance(
                    world.agent_grids[airborne[ai]], world.agent_grids[airborne[bi]]
                )
                min_dist = min(min_dist, float(d))
        step_results.append(StepResult(
            backend="offline", suite=suite.name, scenario=scenario.name,
            policy=policy_name, episode=episode_idx, step=step,
            actions=";".join(str(a) for a in actions),
            positions=";".join(f"{x}:{y}" for x, y in world.agent_grids),
            pending_patients=world.pending_count(),
            deliveries=";".join(str(i) for i in data["deliveries"]),
            reward_sum=round(sum(data["rewards"]), 2),
            landed_count=sum(world.landed),
            depleted_count=sum(world.depleted),
            tracking_error_m=0.0,
        ))
        if data["done"]:
            break

    result = _episode_result_from_world(
        backend="offline", suite=suite.name, scenario=scenario.name,
        policy=policy_name, episode=episode_idx, seed=seed, world=world,
        steps=step + 1, duration_s=time.time() - start,
        total_reward=total_reward, wrong_land_attempts=wrong_land,
        wind_entries=wind, low_signal_entries=ls,
        obstacle_collisions=obs_col, agent_collisions=agent_col,
        min_inter_drone_distance=(0.0 if math.isinf(min_dist) else min_dist),
        mean_tracking_error_m=0.0, max_tracking_error_m=0.0,
        time_to_first_delivery=first_delivery,
        hazard_totals=hazard_totals,
    )
    return result, step_results


# ── SITL backend ──────────────────────────────────────────────────────────────

class InMemoryMetricsCollector:
    """Metrics sink capturing step records in memory (worker-side)."""

    def __init__(self) -> None:
        self.step_records: list[StepRecord] = []

    def log_step(self, record: StepRecord) -> None:
        self.step_records.append(record)

    def log_episode(self, record: Any) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass


async def run_sitl_episode(
    *,
    drones: list,
    suite: SuiteDef,
    scenario: ScenarioDef,
    policy_name: str,
    seed: int,
    episode_idx: int,
    metrics: InMemoryMetricsCollector,
    step_hz: float = 2.0,
    model_path: Optional[Path] = None,
    learned_policy: Optional[FGCSPolicy] = None,
) -> tuple[EpisodeResult, list[StepResult]]:
    """Run one episode against an already-connected SITL fleet."""
    random.seed(seed)
    world = TrueWorld(dict(scenario.world))
    controller = build_policy_controller(
        policy_name, world, seed,
        model_path=model_path, learned_policy=learned_policy,
    )
    coordinator = Coordinator(
        drones=drones, policy=controller, world=world, metrics=metrics,
        step_hz=step_hz, action_delay_steps=scenario.action_delay_steps,
    )
    start = time.time()
    summary = await coordinator.run_episode(
        episode=episode_idx, max_steps=scenario.max_steps
    )
    duration = time.time() - start

    # Derive step results + tracking errors from recorded telemetry.
    step_results: list[StepResult] = []
    tracking_means: list[float] = []
    tracking_max = 0.0
    min_dist = math.inf
    first_delivery = -1.0
    for rec in metrics.step_records:
        errors = []
        positions = rec.simulated_positions
        for i, (gx, gy) in enumerate(positions):
            north_t, east_t = world.grid_to_ned(gx, gy)
            err = math.hypot(
                rec.drone_north[i] - north_t, rec.drone_east[i] - east_t
            )
            errors.append(err)
        mean_err = sum(errors) / len(errors) if errors else 0.0
        tracking_means.append(mean_err)
        tracking_max = max(tracking_max, max(errors) if errors else 0.0)
        for ai in range(len(positions)):
            for bi in range(ai + 1, len(positions)):
                d = abs(positions[ai][0] - positions[bi][0]) + abs(
                    positions[ai][1] - positions[bi][1]
                )
                min_dist = min(min_dist, float(d))
        if rec.deliveries and first_delivery < 0:
            first_delivery = float(rec.step)
        step_results.append(StepResult(
            backend="sitl", suite=suite.name, scenario=scenario.name,
            policy=policy_name, episode=episode_idx, step=rec.step,
            actions=";".join(str(a) for a in rec.actions),
            positions=";".join(f"{x}:{y}" for x, y in positions),
            pending_patients=rec.remaining_patients,
            deliveries=";".join(str(i) for i in rec.deliveries),
            reward_sum=round(sum(rec.rewards), 2),
            landed_count=sum(rec.landed_this_step) + 0,
            depleted_count=0,
            tracking_error_m=round(mean_err, 3),
        ))

    result = _episode_result_from_world(
        backend="sitl", suite=suite.name, scenario=scenario.name,
        policy=policy_name, episode=episode_idx, seed=seed, world=world,
        steps=summary["steps"], duration_s=duration,
        total_reward=summary["total_reward"],
        wrong_land_attempts=summary["wrong_land_attempts"],
        wind_entries=sum(summary["wind_entries"]),
        low_signal_entries=sum(summary["low_signal_entries"]),
        obstacle_collisions=summary["obstacle_collisions"],
        agent_collisions=summary["agent_collisions"],
        min_inter_drone_distance=(0.0 if math.isinf(min_dist) else min_dist),
        mean_tracking_error_m=(
            sum(tracking_means) / len(tracking_means) if tracking_means else 0.0
        ),
        max_tracking_error_m=tracking_max,
        time_to_first_delivery=first_delivery,
        hazard_totals=summary.get("hazard_totals"),
    )
    return result, step_results
