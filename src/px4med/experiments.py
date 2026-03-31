"""Validation experiment harness for offline and PX4 SITL sweeps."""
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
import logging
import math
import os
from pathlib import Path
import random
import statistics
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import torch

from .baselines import make_baseline
from .coordinator import Coordinator
from .docker_manager import DockerManager
from .drone import Drone
from .drone import Telemetry
from .environment import WorldEnvironment
from .metrics import StepRecord
from .policy import PolicyNet
from .state import METERS_PER_CELL, build_state

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

logger = logging.getLogger("px4med.experiments")


REPORTED_METRICS = [
    "mission_success_rate",
    "delivery_rate",
    "mortality_rate",
    "triage_efficiency",
    "total_reward",
    "steps",
    "battery_margin_min",
    "obstacle_collisions",
    "agent_collisions",
    "mean_tracking_error_m",
    "max_tracking_error_m",
]

PLOT_METRICS = [
    ("delivery_rate", "Delivery Rate"),
    ("mortality_rate", "Mortality Rate"),
    ("triage_efficiency", "Triage Efficiency"),
    ("mission_success_rate", "Mission Success"),
]

METRIC_LABELS = {
    "delivery_rate": "Delivery rate",
    "mortality_rate": "Mortality rate",
    "triage_efficiency": "Triage efficiency",
    "mission_success_rate": "Mission success rate",
    "total_reward": "Total reward",
    "steps": "Episode steps",
    "battery_margin_min": "Min battery margin",
    "obstacle_collisions": "Obstacle collisions",
    "agent_collisions": "Agent collisions",
    "mean_tracking_error_m": "Mean tracking error (m)",
    "max_tracking_error_m": "Max tracking error (m)",
}

OPPOSITE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2}
COLORS = {
    "learned": "#1b4965",
    "priority_path": "#d17b0f",
    "nearest_path": "#4c956c",
    "random": "#9c6644",
}


@dataclass(frozen=True)
class ScenarioDef:
    name: str
    label: str
    world: dict[str, Any]
    episodes: int
    max_steps: int = 800
    action_delay_steps: int = 0
    x_value: float | int | None = None


@dataclass(frozen=True)
class SuiteDef:
    name: str
    title: str
    plot_kind: str
    x_label: str
    policies: list[str]
    scenarios: list[ScenarioDef]


@dataclass
class EpisodeResult:
    backend: str
    suite: str
    scenario: str
    scenario_label: str
    policy: str
    episode: int
    seed: int
    action_delay_steps: int
    steps: int
    patients_spawned: int
    patients_delivered: int
    patients_died: int
    delivery_rate: float
    mortality_rate: float
    triage_efficiency: float
    mission_success_rate: float
    both_landed: float
    battery_margin_min: float
    battery_agent0: float
    battery_agent1: float
    total_reward: float
    wind_entries: int
    low_signal_entries: int
    obstacle_collisions: int
    agent_collisions: int
    wrong_land_attempts: int
    min_inter_drone_distance: float
    travel_distance_agent0: float
    travel_distance_agent1: float
    mean_tracking_error_m_agent0: float
    mean_tracking_error_m_agent1: float
    max_tracking_error_m_agent0: float
    max_tracking_error_m_agent1: float
    mean_tracking_error_m: float
    max_tracking_error_m: float
    action_reversals_agent0: int
    action_reversals_agent1: int
    time_to_first_delivery: float
    time_to_first_high_acuity_delivery: float
    high_acuity_service_rate: float
    delivered_w1: int
    delivered_w2: int
    delivered_w3: int
    spawned_w1: int
    spawned_w2: int
    spawned_w3: int


class InMemoryMetricsCollector:
    """Capture coordinator step logs in memory for experiment summarization."""

    def __init__(self) -> None:
        self.step_records: list[StepRecord] = []

    def log_step(self, record: StepRecord) -> None:
        self.step_records.append(record)

    def log_episode(self, record: Any) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass
class StepResult:
    backend: str
    suite: str
    scenario: str
    policy: str
    episode: int
    step: int
    actions: list[int]
    deliveries: list[int]
    rewards: list[float]
    remaining_patients: int
    target_distance_0: int
    target_distance_1: int
    drone0_north: float
    drone0_east: float
    drone0_battery: float
    tracking_error_m_0: float
    drone1_north: float
    drone1_east: float
    drone1_battery: float
    tracking_error_m_1: float
    sim_pos0_x: int
    sim_pos0_y: int
    sim_pos1_x: int
    sim_pos1_y: int
    wind_entries_0: int
    wind_entries_1: int
    low_signal_entries_0: int
    low_signal_entries_1: int
    obstacle_collisions: int
    agent_collisions: int
    landing_attempt_0: bool
    landing_attempt_1: bool
    landed_this_step_0: bool
    landed_this_step_1: bool


def main() -> None:
    parser = argparse.ArgumentParser(description="Run px4med validation experiments")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/agent_marl9.pth"),
        help="Path to trained policy weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Directory to write figures and CSV tables",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Suite name to run; repeat to select multiple suites. Defaults to all.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help="Policy name to run; repeat to select multiple. Defaults to each suite's full policy set.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episodes per scenario for a faster or larger run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=800,
        help="Override max steps per episode. Defaults to the 800-step training horizon.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for all experiments. If omitted, generate one at runtime.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--backend",
        default="offline",
        choices=["offline", "sitl"],
        help="Execution backend: offline world-model rollouts or real PX4 SITL episodes",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="For --backend sitl, skip Docker startup and assume SITL is already running",
    )
    parser.add_argument(
        "--step-hz",
        type=float,
        default=4.0,
        help="For --backend sitl, coordinator control frequency",
    )
    parser.add_argument(
        "--grpc-base-port",
        type=int,
        default=50051,
        help="For --backend sitl, base gRPC port for MAVSDK backends",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=3.0,
        help="For --backend sitl, scale PX4 cruise/max horizontal and vertical speeds",
    )
    parser.add_argument(
        "--battery-drain-rate",
        type=float,
        default=180.0,
        help="For --backend sitl, PX4 SIM_BAT_DRAIN value. Higher lasts longer in this setup.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "tables").mkdir(exist_ok=True)
    logger.info("Output directory: %s", args.output_dir)

    suites = build_default_suites(
        episodes_override=args.episodes,
        max_steps_override=args.max_steps,
    )
    if args.suite:
        selected = {name for name in args.suite}
        suites = [suite for suite in suites if suite.name in selected]
        missing = sorted(selected - {suite.name for suite in suites})
        if missing:
            raise SystemExit(f"Unknown suite(s): {', '.join(missing)}")
    if args.policy:
        selected_policies = set(args.policy)
        filtered_suites: list[SuiteDef] = []
        for suite in suites:
            policies = [policy for policy in suite.policies if policy in selected_policies]
            if policies:
                filtered_suites.append(
                    SuiteDef(
                        name=suite.name,
                        title=suite.title,
                        plot_kind=suite.plot_kind,
                        x_label=suite.x_label,
                        policies=policies,
                        scenarios=suite.scenarios,
                    )
                )
        suites = filtered_suites
        if not suites:
            raise SystemExit(f"No suites contain requested policy filter(s): {', '.join(args.policy)}")
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(0, 2**32)

    logger.info("Backend: %s", args.backend)
    logger.info("Selected suites: %s", ", ".join(suite.name for suite in suites))
    if args.policy:
        logger.info("Selected policies: %s", ", ".join(args.policy))
    logger.info("Base seed: %d", args.seed)

    requires_learned = any("learned" in suite.policies for suite in suites)
    learned_policy = PolicyNet(args.model) if requires_learned else None
    if learned_policy is not None:
        logger.info("Loaded learned policy from %s on %s", args.model, learned_policy.device)

    if args.backend == "offline":
        all_results, step_results = run_offline_experiments(suites, learned_policy, args.seed)
    else:
        all_results, step_results = asyncio.run(
            run_sitl_experiments(
                suites=suites,
                learned_policy=learned_policy,
                seed_base=args.seed,
                no_docker=args.no_docker,
                step_hz=args.step_hz,
                grpc_base_port=args.grpc_base_port,
                speed_factor=args.speed_factor,
                battery_drain_rate=args.battery_drain_rate,
                log_dir=args.output_dir / "sitl_logs",
            )
        )

    episode_csv = args.output_dir / "tables" / "episodes.csv"
    step_csv = args.output_dir / "tables" / "steps.csv"
    summary_csv = args.output_dir / "tables" / "summary.csv"
    write_episode_csv(episode_csv, all_results)
    write_step_csv(step_csv, step_results)
    summaries = summarize_results(all_results)
    write_summary_csv(summary_csv, summaries)
    logger.info("Wrote episode table: %s", episode_csv)
    logger.info("Wrote step table: %s", step_csv)
    logger.info("Wrote summary table: %s", summary_csv)

    for suite in suites:
        suite_rows = [row for row in summaries if row["suite"] == suite.name]
        suite_summary_csv = args.output_dir / "tables" / f"{suite.name}_summary.csv"
        suite_figure_dir = args.output_dir / "figures" / suite.name
        write_summary_csv(suite_summary_csv, suite_rows)
        plot_suite(
            suite,
            suite_rows,
            suite_figure_dir,
        )
        logger.info("Wrote suite summary: %s", suite_summary_csv)
        logger.info("Wrote suite figures: %s", suite_figure_dir)

    plot_episode_details(
        step_results=step_results,
        episode_results=all_results,
        output_dir=args.output_dir / "figures" / "episodes",
    )

    logger.info("All experiment suites complete.")


def build_default_suites(
    episodes_override: int | None = None,
    max_steps_override: int | None = None,
) -> list[SuiteDef]:
    def eps(default: int) -> int:
        return episodes_override if episodes_override is not None else default

    def max_steps(default: int = 800) -> int:
        return max_steps_override if max_steps_override is not None else default

    nominal_world = {
        "grid": {"size": 50, "meters_per_cell": 2.0},
    }

    triage_world = {
        "grid": {"size": 50, "meters_per_cell": 2.0},
        "obstacles": [],
        "patients": [
            {"grid": [4, 1], "weight": 1},
            {"grid": [14, 1], "weight": 3},
            {"grid": [10, 10], "weight": 1},
            {"grid": [16, 12], "weight": 2},
            {"grid": [35, 10], "weight": 1},
            {"grid": [10, 35], "weight": 1},
            {"grid": [40, 30], "weight": 1},
            {"grid": [30, 40], "weight": 1},
        ],
        "landing_zones": [{"grid": [48, 48]}, {"grid": [48, 45]}],
    }

    return [
        SuiteDef(
            name="baseline_comparison",
            title="Nominal Baseline Comparison",
            plot_kind="policy_bar",
            x_label="Policy",
            policies=["learned", "priority_path", "nearest_path", "random"],
            scenarios=[
                ScenarioDef(
                    name="nominal",
                    label="Nominal",
                    world=nominal_world,
                    episodes=eps(40),
                    max_steps=max_steps(),
                ),
            ],
        ),
        SuiteDef(
            name="hazard_sweep",
            title="Hazard Robustness Sweep",
            plot_kind="scenario_line",
            x_label="Hazard severity",
            policies=["learned", "priority_path"],
            scenarios=[
                ScenarioDef(
                    name="hazard_low",
                    label="Low",
                    x_value=1,
                    episodes=eps(30),
                    max_steps=max_steps(),
                    world={
                        **nominal_world,
                        "hazards": {
                            "num_wind_zones": 5,
                            "num_low_signal_zones": 4,
                            "low_signal_failure_prob": 0.1,
                        },
                    },
                ),
                ScenarioDef(
                    name="hazard_med",
                    label="Medium",
                    x_value=2,
                    episodes=eps(30),
                    max_steps=max_steps(),
                    world=nominal_world,
                ),
                ScenarioDef(
                    name="hazard_high",
                    label="High",
                    x_value=3,
                    episodes=eps(30),
                    max_steps=max_steps(),
                    world={
                        **nominal_world,
                        "hazards": {
                            "num_wind_zones": 25,
                            "num_low_signal_zones": 18,
                            "low_signal_failure_prob": 0.5,
                        },
                    },
                ),
            ],
        ),
        SuiteDef(
            name="battery_sweep",
            title="Battery Stress Sweep",
            plot_kind="scenario_line",
            x_label="Initial battery",
            policies=["learned", "priority_path"],
            scenarios=[
                ScenarioDef(
                    name="battery_100",
                    label="100",
                    x_value=100,
                    episodes=eps(30),
                    max_steps=max_steps(),
                    world=nominal_world,
                ),
                ScenarioDef(
                    name="battery_75",
                    label="75",
                    x_value=75,
                    episodes=eps(30),
                    max_steps=max_steps(),
                    world={**nominal_world, "battery": {"initial": 75}},
                ),
                ScenarioDef(
                    name="battery_50",
                    label="50",
                    x_value=50,
                    episodes=eps(30),
                    max_steps=max_steps(),
                    world={**nominal_world, "battery": {"initial": 50}},
                ),
                ScenarioDef(
                    name="battery_35",
                    label="35",
                    x_value=35,
                    episodes=eps(30),
                    max_steps=max_steps(),
                    world={**nominal_world, "battery": {"initial": 35}},
                ),
            ],
        ),
        SuiteDef(
            name="delay_sweep",
            title="Action Delay Sweep",
            plot_kind="scenario_line",
            x_label="Action delay (steps)",
            policies=["learned", "priority_path"],
            scenarios=[
                ScenarioDef(
                    name="delay_0",
                    label="0",
                    x_value=0,
                    action_delay_steps=0,
                    episodes=eps(30),
                    max_steps=max_steps(),
                    world=nominal_world,
                ),
                ScenarioDef(
                    name="delay_1",
                    label="1",
                    x_value=1,
                    action_delay_steps=1,
                    episodes=eps(30),
                    max_steps=max_steps(),
                    world=nominal_world,
                ),
                ScenarioDef(
                    name="delay_2",
                    label="2",
                    x_value=2,
                    action_delay_steps=2,
                    episodes=eps(30),
                    max_steps=max_steps(),
                    world=nominal_world,
                ),
                ScenarioDef(
                    name="delay_3",
                    label="3",
                    x_value=3,
                    action_delay_steps=3,
                    episodes=eps(30),
                    max_steps=max_steps(),
                    world=nominal_world,
                ),
            ],
        ),
        SuiteDef(
            name="triage_priority",
            title="Triage Conflict Scenario",
            plot_kind="policy_bar",
            x_label="Policy",
            policies=["learned", "priority_path", "nearest_path", "random"],
            scenarios=[
                ScenarioDef(
                    name="triage_conflict",
                    label="Conflict",
                    episodes=eps(40),
                    max_steps=max_steps(),
                    world=triage_world,
                ),
            ],
        ),
    ]


def run_offline_experiments(
    suites: list[SuiteDef],
    learned_policy: PolicyNet | None,
    seed_base: int,
) -> tuple[list[EpisodeResult], list[StepResult]]:
    all_results: list[EpisodeResult] = []
    all_step_results: list[StepResult] = []
    for suite_idx, suite in enumerate(suites):
        total_suite_runs = len(suite.scenarios) * len(suite.policies)
        logger.info(
            "Starting suite %s (%s): %d scenarios, %d policies, %d scenario-policy runs",
            suite.name,
            suite.title,
            len(suite.scenarios),
            len(suite.policies),
            total_suite_runs,
        )
        for scenario_idx, scenario in enumerate(suite.scenarios):
            logger.info(
                "Scenario %s [%d/%d]: label=%s episodes=%d max_steps=%d delay=%d config=%s",
                scenario.name,
                scenario_idx + 1,
                len(suite.scenarios),
                scenario.label,
                scenario.episodes,
                scenario.max_steps,
                scenario.action_delay_steps,
                world_summary(scenario.world),
            )
            for policy_idx, policy_name in enumerate(suite.policies):
                logger.info(
                    "Policy %s [%d/%d] for scenario %s",
                    policy_name,
                    policy_idx + 1,
                    len(suite.policies),
                    scenario.name,
                )
                for episode_idx in range(scenario.episodes):
                    seed = (
                        seed_base
                        + suite_idx * 100_000
                        + scenario_idx * 1_000
                        + episode_idx
                    )
                    result, step_results = run_offline_episode(
                        suite=suite,
                        scenario=scenario,
                        policy_name=policy_name,
                        policy=learned_policy,
                        seed=seed,
                        episode_idx=episode_idx,
                    )
                    all_results.append(result)
                    all_step_results.extend(step_results)
                    log_episode_completion(result, scenario.episodes)
    return all_results, all_step_results


async def run_sitl_experiments(
    suites: list[SuiteDef],
    learned_policy: PolicyNet | None,
    seed_base: int,
    no_docker: bool,
    step_hz: float,
    grpc_base_port: int,
    speed_factor: float,
    battery_drain_rate: float,
    log_dir: Path,
) -> tuple[list[EpisodeResult], list[StepResult]]:
    dm: DockerManager | None = None
    drones: list[Drone] = []
    all_results: list[EpisodeResult] = []
    all_step_results: list[StepResult] = []
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not no_docker:
            dm = DockerManager(log_dir=log_dir)
            logger.info("Starting PX4 SITL container for experiment backend ...")
            dm.start()
            logger.info("Waiting for PX4 SITL to become healthy ...")
            await dm.wait_healthy()
            logger.info("PX4 SITL healthy.")

        addresses = dm.mavsdk_addresses if dm else [
            f"udpin://0.0.0.0:{14540 + i}" for i in range(2)
        ]
        drones = [
            Drone(i, addresses[i], grpc_port=grpc_base_port + i)
            for i in range(len(addresses))
        ]
        logger.info("Connecting to %d drone(s) for SITL backend ...", len(drones))
        for drone in drones:
            await drone.connect()
        logger.info(
            "Applying PX4 battery drain rate %.2f for SITL experiments ...",
            battery_drain_rate,
        )
        for drone in drones:
            await drone.configure_battery_profile(battery_drain_rate)
        if speed_factor != 1.0:
            logger.info("Applying PX4 speed factor %.2f for SITL experiments ...", speed_factor)
        for drone in drones:
            await drone.configure_speed_profile(speed_factor)
        logger.info("All SITL drones connected.")

        for suite_idx, suite in enumerate(suites):
            total_suite_runs = len(suite.scenarios) * len(suite.policies)
            logger.info(
                "Starting SITL suite %s (%s): %d scenarios, %d policies, %d scenario-policy runs",
                suite.name,
                suite.title,
                len(suite.scenarios),
                len(suite.policies),
                total_suite_runs,
            )
            for scenario_idx, scenario in enumerate(suite.scenarios):
                logger.info(
                    "Scenario %s [%d/%d]: label=%s episodes=%d max_steps=%d delay=%d config=%s",
                    scenario.name,
                    scenario_idx + 1,
                    len(suite.scenarios),
                    scenario.label,
                    scenario.episodes,
                    scenario.max_steps,
                    scenario.action_delay_steps,
                    world_summary(scenario.world),
                )
                for policy_idx, policy_name in enumerate(suite.policies):
                    logger.info(
                        "Policy %s [%d/%d] for scenario %s",
                        policy_name,
                        policy_idx + 1,
                        len(suite.policies),
                        scenario.name,
                    )
                    controller = build_policy_controller(
                        policy_name=policy_name,
                        learned_policy=learned_policy,
                        seed=seed_base + suite_idx * 100_000 + scenario_idx * 1_000,
                    )
                    for episode_idx in range(scenario.episodes):
                        seed = (
                            seed_base
                            + suite_idx * 100_000
                            + scenario_idx * 1_000
                            + episode_idx
                        )
                        result, step_results = await run_sitl_episode(
                            suite=suite,
                            scenario=scenario,
                            policy_name=policy_name,
                            controller=controller,
                            drones=drones,
                            step_hz=step_hz,
                            seed=seed,
                            episode_idx=episode_idx,
                        )
                        all_results.append(result)
                        all_step_results.extend(step_results)
                        log_episode_completion(result, scenario.episodes)
        return all_results, all_step_results
    finally:
        for i, drone in enumerate(drones):
            try:
                await drone.land()
            except Exception as exc:
                logger.warning("Could not land drone %d during experiment shutdown: %s", i, exc)
        if dm is not None:
            logger.info("Stopping PX4 SITL container ...")
            dm.stop()


def run_episode(
    suite: SuiteDef,
    scenario: ScenarioDef,
    policy_name: str,
    policy: PolicyNet | None,
    seed: int,
    episode_idx: int,
) -> tuple[EpisodeResult, list[StepResult]]:
    return run_offline_episode(suite, scenario, policy_name, policy, seed, episode_idx)


def run_offline_episode(
    suite: SuiteDef,
    scenario: ScenarioDef,
    policy_name: str,
    policy: PolicyNet | None,
    seed: int,
    episode_idx: int,
) -> tuple[EpisodeResult, list[StepResult]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    world = WorldEnvironment(copy.deepcopy(scenario.world))
    world.reset()

    if policy_name != "learned":
        controller = make_baseline(policy_name, seed)
    else:
        if policy is None:
            raise ValueError("Learned policy requested but no model was loaded")
        controller = None

    action_queues = [deque([-1] * scenario.action_delay_steps) for _ in range(2)]
    prev_active = [p.active for p in world.patients]
    spawn_weights: dict[int, int] = {
        p.idx: int(p.weight) for p in world.patients if p.active
    }
    delivery_steps: dict[int, int] = {}
    prev_actions = [-1, -1]
    action_reversals = [0, 0]
    travel_distance = [0.0, 0.0]
    wrong_land_attempts = 0
    min_inter_drone_distance = math.inf
    total_reward = 0.0
    wind_entries = [0, 0]
    low_signal_entries = [0, 0]
    obstacle_collisions = 0
    agent_collisions = 0
    step_results: list[StepResult] = []

    for step in range(scenario.max_steps):
        selected_actions = select_actions(policy_name, policy, controller, world)
        actions_to_apply = []
        for agent_idx, action in enumerate(selected_actions):
            if scenario.action_delay_steps > 0:
                action_queues[agent_idx].append(action)
                executed_action = action_queues[agent_idx].popleft()
            else:
                executed_action = action
            actions_to_apply.append(executed_action)

            if (
                prev_actions[agent_idx] in OPPOSITE_ACTION
                and executed_action == OPPOSITE_ACTION[prev_actions[agent_idx]]
            ):
                action_reversals[agent_idx] += 1
            if executed_action in OPPOSITE_ACTION:
                prev_actions[agent_idx] = executed_action

        old_positions = list(world.agent_grids)
        step_data = world.step(actions_to_apply)
        total_reward += sum(step_data["rewards"])
        obstacle_collisions += int(step_data["obstacle_collisions"])
        agent_collisions += int(step_data["agent_collisions"])
        for idx in range(2):
            wind_entries[idx] += int(step_data["wind_entries"][idx])
            low_signal_entries[idx] += int(step_data["low_signal_entries"][idx])
            travel_distance[idx] += manhattan(old_positions[idx], world.agent_grids[idx])
            if step_data["landing_attempts"][idx] and not step_data["landed_this_step"][idx]:
                wrong_land_attempts += 1

        separation = manhattan(world.agent_grids[0], world.agent_grids[1])
        min_inter_drone_distance = min(min_inter_drone_distance, float(separation))

        for patient_idx in step_data["deliveries"]:
            delivery_steps.setdefault(patient_idx, step + 1)

        for patient in world.patients:
            if patient.active and not prev_active[patient.idx]:
                spawn_weights[patient.idx] = int(patient.weight)
        prev_active = [p.active for p in world.patients]

        step_results.append(
            StepResult(
                backend="offline",
                suite=suite.name,
                scenario=scenario.name,
                policy=policy_name,
                episode=episode_idx,
                step=step,
                actions=list(actions_to_apply),
                deliveries=list(step_data["deliveries"]),
                rewards=list(step_data["rewards"]),
                remaining_patients=sum(
                    1 for patient in world.patients if patient.active and not patient.delivered
                ),
                target_distance_0=_target_distance(world, 0),
                target_distance_1=_target_distance(world, 1),
                drone0_north=-world.agent_grids[0][1] * METERS_PER_CELL,
                drone0_east=world.agent_grids[0][0] * METERS_PER_CELL,
                drone0_battery=float(world.batteries[0]),
                tracking_error_m_0=0.0,
                drone1_north=-world.agent_grids[1][1] * METERS_PER_CELL,
                drone1_east=world.agent_grids[1][0] * METERS_PER_CELL,
                drone1_battery=float(world.batteries[1]),
                tracking_error_m_1=0.0,
                sim_pos0_x=world.agent_grids[0][0],
                sim_pos0_y=world.agent_grids[0][1],
                sim_pos1_x=world.agent_grids[1][0],
                sim_pos1_y=world.agent_grids[1][1],
                wind_entries_0=int(step_data["wind_entries"][0]),
                wind_entries_1=int(step_data["wind_entries"][1]),
                low_signal_entries_0=int(step_data["low_signal_entries"][0]),
                low_signal_entries_1=int(step_data["low_signal_entries"][1]),
                obstacle_collisions=int(step_data["obstacle_collisions"]),
                agent_collisions=int(step_data["agent_collisions"]),
                landing_attempt_0=bool(step_data["landing_attempts"][0]),
                landing_attempt_1=bool(step_data["landing_attempts"][1]),
                landed_this_step_0=bool(step_data["landed_this_step"][0]),
                landed_this_step_1=bool(step_data["landed_this_step"][1]),
            )
        )

        if step_data["done"]:
            break

    triage = world.triage_summary()
    patients_spawned = sum(1 for p in world.patients if p.active)
    patients_delivered = sum(1 for p in world.patients if p.actually_delivered)
    patients_died = sum(1 for p in world.patients if p.delivered and not p.actually_delivered)
    spawn_weight_counts = Counter(spawn_weights.values())
    delivered_weight_counts = Counter(
        spawn_weights[pid] for pid in delivery_steps if pid in spawn_weights
    )
    high_acuity_delivered = delivered_weight_counts.get(3, 0)
    high_acuity_spawned = spawn_weight_counts.get(3, 0)

    time_to_first_delivery = min(delivery_steps.values()) if delivery_steps else math.nan
    high_acuity_delivery_steps = [
        delivery_steps[pid]
        for pid in delivery_steps
        if spawn_weights.get(pid) == 3
    ]
    time_to_first_high_acuity = (
        min(high_acuity_delivery_steps) if high_acuity_delivery_steps else math.nan
    )

    both_landed = float(all(world.landed))
    battery_margin_min = float(min(world.batteries))
    mission_success = float(
        all(world.landed) and battery_margin_min > 0.0 and patients_died == 0
    )
    mean_tracking_error_m_0 = 0.0
    mean_tracking_error_m_1 = 0.0
    max_tracking_error_m_0 = 0.0
    max_tracking_error_m_1 = 0.0

    return EpisodeResult(
        backend="offline",
        suite=suite.name,
        scenario=scenario.name,
        scenario_label=scenario.label,
        policy=policy_name,
        episode=episode_idx,
        seed=seed,
        action_delay_steps=scenario.action_delay_steps,
        steps=step + 1,
        patients_spawned=patients_spawned,
        patients_delivered=patients_delivered,
        patients_died=patients_died,
        delivery_rate=safe_div(patients_delivered, patients_spawned),
        mortality_rate=safe_div(patients_died, patients_spawned),
        triage_efficiency=float(triage["triage_efficiency"]),
        mission_success_rate=mission_success,
        both_landed=both_landed,
        battery_margin_min=battery_margin_min,
        battery_agent0=float(world.batteries[0]),
        battery_agent1=float(world.batteries[1]),
        total_reward=total_reward,
        wind_entries=sum(wind_entries),
        low_signal_entries=sum(low_signal_entries),
        obstacle_collisions=obstacle_collisions,
        agent_collisions=agent_collisions,
        wrong_land_attempts=wrong_land_attempts,
        min_inter_drone_distance=min_inter_drone_distance,
        travel_distance_agent0=float(travel_distance[0]),
        travel_distance_agent1=float(travel_distance[1]),
        mean_tracking_error_m_agent0=mean_tracking_error_m_0,
        mean_tracking_error_m_agent1=mean_tracking_error_m_1,
        max_tracking_error_m_agent0=max_tracking_error_m_0,
        max_tracking_error_m_agent1=max_tracking_error_m_1,
        mean_tracking_error_m=0.0,
        max_tracking_error_m=0.0,
        action_reversals_agent0=action_reversals[0],
        action_reversals_agent1=action_reversals[1],
        time_to_first_delivery=time_to_first_delivery,
        time_to_first_high_acuity_delivery=time_to_first_high_acuity,
        high_acuity_service_rate=safe_div(high_acuity_delivered, high_acuity_spawned),
        delivered_w1=delivered_weight_counts.get(1, 0),
        delivered_w2=delivered_weight_counts.get(2, 0),
        delivered_w3=delivered_weight_counts.get(3, 0),
        spawned_w1=spawn_weight_counts.get(1, 0),
        spawned_w2=spawn_weight_counts.get(2, 0),
        spawned_w3=spawn_weight_counts.get(3, 0),
    ), step_results


def select_actions(
    policy_name: str,
    policy: PolicyNet | None,
    controller: Any,
    world: WorldEnvironment,
) -> list[int]:
    if policy_name == "learned":
        if policy is None:
            raise ValueError("Missing learned policy")
        telems = [_telem_from_world(world, idx) for idx in range(2)]
        states = [
            build_state(idx, telems[idx], telems[1 - idx], world)
            for idx in range(2)
        ]
        return [
            policy.select_action(states[idx], states[1 - idx])
            for idx in range(2)
        ]
    return controller.select_actions(world)


async def run_sitl_episode(
    suite: SuiteDef,
    scenario: ScenarioDef,
    policy_name: str,
    controller: Any,
    drones: list[Drone],
    step_hz: float,
    seed: int,
    episode_idx: int,
) -> tuple[EpisodeResult, list[StepResult]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    world = WorldEnvironment(copy.deepcopy(scenario.world))
    metrics = InMemoryMetricsCollector()
    coordinator = Coordinator(
        drones=drones,
        policy=controller,
        world=world,
        metrics=metrics,
        step_hz=step_hz,
        action_delay_steps=scenario.action_delay_steps,
    )
    summary = await coordinator.run_episode(episode=episode_idx, max_steps=scenario.max_steps)
    step_results = build_step_results(
        backend="sitl",
        suite=suite.name,
        scenario=scenario.name,
        policy=policy_name,
        episode=episode_idx,
        step_records=metrics.step_records,
    )
    return build_sitl_result(
        suite=suite,
        scenario=scenario,
        policy_name=policy_name,
        seed=seed,
        episode_idx=episode_idx,
        world=world,
        summary=summary,
        step_records=metrics.step_records,
    ), step_results


def build_sitl_result(
    suite: SuiteDef,
    scenario: ScenarioDef,
    policy_name: str,
    seed: int,
    episode_idx: int,
    world: WorldEnvironment,
    summary: dict[str, Any],
    step_records: list[StepRecord],
) -> EpisodeResult:
    spawn_weights = {
        p.idx: int(p.weight)
        for p in world.patients
        if p.active
    }
    delivered_weight_counts = Counter(
        spawn_weights[pid]
        for pid in {
            delivery
            for record in step_records
            for delivery in record.deliveries
        }
        if pid in spawn_weights
    )
    spawn_weight_counts = Counter(spawn_weights.values())
    high_acuity_delivered = delivered_weight_counts.get(3, 0)
    high_acuity_spawned = spawn_weight_counts.get(3, 0)

    wrong_land_attempts = sum(
        sum(1 for attempt, landed in zip(record.landing_attempts, record.landed_this_step) if attempt and not landed)
        for record in step_records
    )
    min_inter_drone_distance = min(
        (
            abs(record.drone0_east / METERS_PER_CELL - record.drone1_east / METERS_PER_CELL)
            + abs(-record.drone0_north / METERS_PER_CELL + record.drone1_north / METERS_PER_CELL)
            for record in step_records
        ),
        default=math.nan,
    )
    travel_distance_agent0 = 0.0
    travel_distance_agent1 = 0.0
    prev_sim_positions: list[list[int]] | None = None
    prev_actions = [-1, -1]
    action_reversals = [0, 0]
    delivery_steps: dict[int, int] = {}
    tracking_errors_0: list[float] = []
    tracking_errors_1: list[float] = []
    for record in step_records:
        if prev_sim_positions is not None:
            travel_distance_agent0 += manhattan(
                tuple(prev_sim_positions[0]),
                tuple(record.simulated_positions[0]),
            )
            travel_distance_agent1 += manhattan(
                tuple(prev_sim_positions[1]),
                tuple(record.simulated_positions[1]),
            )
        prev_sim_positions = record.simulated_positions
        for i, action in enumerate(record.actions):
            if prev_actions[i] in OPPOSITE_ACTION and action == OPPOSITE_ACTION[prev_actions[i]]:
                action_reversals[i] += 1
            if action in OPPOSITE_ACTION:
                prev_actions[i] = action
        for delivery in record.deliveries:
            delivery_steps.setdefault(delivery, record.step + 1)
        tracking_errors_0.append(
            math.dist(
                (
                    record.drone0_north,
                    record.drone0_east,
                ),
                (
                    -record.simulated_positions[0][1] * METERS_PER_CELL,
                    record.simulated_positions[0][0] * METERS_PER_CELL,
                ),
            )
        )
        tracking_errors_1.append(
            math.dist(
                (
                    record.drone1_north,
                    record.drone1_east,
                ),
                (
                    -record.simulated_positions[1][1] * METERS_PER_CELL,
                    record.simulated_positions[1][0] * METERS_PER_CELL,
                ),
            )
        )

    time_to_first_delivery = min(delivery_steps.values()) if delivery_steps else math.nan
    high_acuity_delivery_steps = [
        delivery_steps[pid] for pid in delivery_steps if spawn_weights.get(pid) == 3
    ]
    time_to_first_high_acuity = (
        min(high_acuity_delivery_steps) if high_acuity_delivery_steps else math.nan
    )
    mean_tracking_error_m_0 = statistics.fmean(tracking_errors_0) if tracking_errors_0 else math.nan
    mean_tracking_error_m_1 = statistics.fmean(tracking_errors_1) if tracking_errors_1 else math.nan
    max_tracking_error_m_0 = max(tracking_errors_0, default=math.nan)
    max_tracking_error_m_1 = max(tracking_errors_1, default=math.nan)

    return EpisodeResult(
        backend="sitl",
        suite=suite.name,
        scenario=scenario.name,
        scenario_label=scenario.label,
        policy=policy_name,
        episode=episode_idx,
        seed=seed,
        action_delay_steps=scenario.action_delay_steps,
        steps=int(summary["steps"]),
        patients_spawned=int(summary["patients_spawned"]),
        patients_delivered=int(summary["patients_delivered"]),
        patients_died=int(summary["patients_died"]),
        delivery_rate=safe_div(summary["patients_delivered"], summary["patients_spawned"]),
        mortality_rate=safe_div(summary["patients_died"], summary["patients_spawned"]),
        triage_efficiency=float(summary["triage_efficiency"]),
        mission_success_rate=float(
            summary["both_landed"]
            and min(summary["simulated_battery_remaining"]) > 0.0
            and summary["patients_died"] == 0
        ),
        both_landed=float(summary["both_landed"]),
        battery_margin_min=float(min(summary["simulated_battery_remaining"])),
        battery_agent0=float(summary["simulated_battery_remaining"][0]),
        battery_agent1=float(summary["simulated_battery_remaining"][1]),
        total_reward=float(summary["total_reward"]),
        wind_entries=int(sum(summary["wind_entries"])),
        low_signal_entries=int(sum(summary["low_signal_entries"])),
        obstacle_collisions=int(summary["obstacle_collisions"]),
        agent_collisions=int(summary["agent_collisions"]),
        wrong_land_attempts=wrong_land_attempts,
        min_inter_drone_distance=float(min_inter_drone_distance),
        travel_distance_agent0=float(travel_distance_agent0),
        travel_distance_agent1=float(travel_distance_agent1),
        mean_tracking_error_m_agent0=float(mean_tracking_error_m_0),
        mean_tracking_error_m_agent1=float(mean_tracking_error_m_1),
        max_tracking_error_m_agent0=float(max_tracking_error_m_0),
        max_tracking_error_m_agent1=float(max_tracking_error_m_1),
        mean_tracking_error_m=float(statistics.fmean([mean_tracking_error_m_0, mean_tracking_error_m_1])),
        max_tracking_error_m=float(max(max_tracking_error_m_0, max_tracking_error_m_1)),
        action_reversals_agent0=action_reversals[0],
        action_reversals_agent1=action_reversals[1],
        time_to_first_delivery=time_to_first_delivery,
        time_to_first_high_acuity_delivery=time_to_first_high_acuity,
        high_acuity_service_rate=safe_div(high_acuity_delivered, high_acuity_spawned),
        delivered_w1=delivered_weight_counts.get(1, 0),
        delivered_w2=delivered_weight_counts.get(2, 0),
        delivered_w3=delivered_weight_counts.get(3, 0),
        spawned_w1=spawn_weight_counts.get(1, 0),
        spawned_w2=spawn_weight_counts.get(2, 0),
        spawned_w3=spawn_weight_counts.get(3, 0),
    )


def _telem_from_world(world: WorldEnvironment, agent_idx: int) -> Telemetry:
    grid_x, grid_y = world.agent_grids[agent_idx]
    return Telemetry(
        north_m=-grid_y * METERS_PER_CELL,
        east_m=grid_x * METERS_PER_CELL,
        down_m=-5.0,
        battery_pct=world.batteries[agent_idx],
        is_landed=world.landed[agent_idx],
    )


def _target_distance(world: WorldEnvironment, agent_idx: int) -> int:
    if world.landed[agent_idx]:
        return 0

    pending_patients = [patient for patient in world.patients if patient.active and not patient.delivered]
    if pending_patients:
        agent_pos = world.agent_grids[agent_idx]
        return min(manhattan(agent_pos, patient.grid) for patient in pending_patients)

    return manhattan(world.agent_grids[agent_idx], world.landing_zone(agent_idx))


def build_policy_controller(
    policy_name: str,
    learned_policy: PolicyNet | None,
    seed: int,
) -> Any:
    if policy_name == "learned":
        if learned_policy is None:
            raise ValueError("Learned policy requested but no model was loaded")
        return learned_policy
    return make_baseline(policy_name, seed)


def build_step_results(
    backend: str,
    suite: str,
    scenario: str,
    policy: str,
    episode: int,
    step_records: list[StepRecord],
) -> list[StepResult]:
    rows: list[StepResult] = []
    for record in step_records:
        rows.append(
            StepResult(
                backend=backend,
                suite=suite,
                scenario=scenario,
                policy=policy,
                episode=episode,
                step=record.step,
                actions=list(record.actions),
                deliveries=list(record.deliveries),
                rewards=list(record.rewards),
                remaining_patients=int(record.remaining_patients),
                target_distance_0=int(record.target_distances[0]),
                target_distance_1=int(record.target_distances[1]),
                drone0_north=float(record.drone0_north),
                drone0_east=float(record.drone0_east),
                drone0_battery=float(record.drone0_battery),
                tracking_error_m_0=float(
                    math.dist(
                        (record.drone0_north, record.drone0_east),
                        (
                            -record.simulated_positions[0][1] * METERS_PER_CELL,
                            record.simulated_positions[0][0] * METERS_PER_CELL,
                        ),
                    )
                ),
                drone1_north=float(record.drone1_north),
                drone1_east=float(record.drone1_east),
                drone1_battery=float(record.drone1_battery),
                tracking_error_m_1=float(
                    math.dist(
                        (record.drone1_north, record.drone1_east),
                        (
                            -record.simulated_positions[1][1] * METERS_PER_CELL,
                            record.simulated_positions[1][0] * METERS_PER_CELL,
                        ),
                    )
                ),
                sim_pos0_x=int(record.simulated_positions[0][0]),
                sim_pos0_y=int(record.simulated_positions[0][1]),
                sim_pos1_x=int(record.simulated_positions[1][0]),
                sim_pos1_y=int(record.simulated_positions[1][1]),
                wind_entries_0=int(record.wind_entries[0]),
                wind_entries_1=int(record.wind_entries[1]),
                low_signal_entries_0=int(record.low_signal_entries[0]),
                low_signal_entries_1=int(record.low_signal_entries[1]),
                obstacle_collisions=int(record.obstacle_collisions),
                agent_collisions=int(record.agent_collisions),
                landing_attempt_0=bool(record.landing_attempts[0]),
                landing_attempt_1=bool(record.landing_attempts[1]),
                landed_this_step_0=bool(record.landed_this_step[0]),
                landed_this_step_1=bool(record.landed_this_step[1]),
            )
        )
    return rows


def summarize_results(results: list[EpisodeResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[(result.backend, result.suite, result.scenario, result.policy)].append(result)

    summaries: list[dict[str, Any]] = []
    for (backend, suite, scenario, policy), rows in sorted(grouped.items()):
        first = rows[0]
        summary: dict[str, Any] = {
            "backend": backend,
            "suite": suite,
            "scenario": scenario,
            "scenario_label": first.scenario_label,
            "policy": policy,
            "episodes": len(rows),
            "action_delay_steps": first.action_delay_steps,
        }
        for metric in set(REPORTED_METRICS + ["high_acuity_service_rate", "wrong_land_attempts"]):
            values = [float(getattr(row, metric)) for row in rows]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_ci95"] = ci95(values)
        summaries.append(summary)
    return summaries


def plot_suite(suite: SuiteDef, summaries: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if suite.plot_kind == "policy_bar":
        scenario_name = suite.scenarios[0].name
        suite_rows = [row for row in summaries if row["scenario"] == scenario_name]
        order = {policy: idx for idx, policy in enumerate(suite.policies)}
        suite_rows.sort(key=lambda row: order[row["policy"]])

        for metric, title in PLOT_METRICS:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            xs = range(len(suite_rows))
            means = [row[f"{metric}_mean"] for row in suite_rows]
            cis = [row[f"{metric}_ci95"] for row in suite_rows]
            colors = [COLORS.get(row["policy"], "#4c4c4c") for row in suite_rows]
            ax.bar(xs, means, yerr=cis, color=colors, capsize=4)
            ax.set_xticks(list(xs), [row["policy"] for row in suite_rows], rotation=20)
            ax.set_title(f"{suite.title} | {title}")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.grid(axis="y", alpha=0.25)
            figure_path = output_dir / f"{suite.name}_{_metric_slug(metric)}.png"
            _save_figure(fig, figure_path)
            _save_figure_csv(
                figure_path,
                [
                    {
                        "policy": row["policy"],
                        "mean": row[f"{metric}_mean"],
                        "ci95": row[f"{metric}_ci95"],
                    }
                    for row in suite_rows
                ],
            )

    elif suite.plot_kind == "scenario_line":
        x_order = {scenario.name: scenario.x_value for scenario in suite.scenarios}
        label_order = {scenario.name: scenario.label for scenario in suite.scenarios}
        for metric, title in PLOT_METRICS:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            csv_rows: list[dict[str, Any]] = []
            for policy in suite.policies:
                rows = [row for row in summaries if row["policy"] == policy]
                rows.sort(key=lambda row: x_order[row["scenario"]])
                xs = [x_order[row["scenario"]] for row in rows]
                means = [row[f"{metric}_mean"] for row in rows]
                cis = [row[f"{metric}_ci95"] for row in rows]
                ax.plot(xs, means, marker="o", label=policy, color=COLORS.get(policy))
                lower = [mean - ci for mean, ci in zip(means, cis)]
                upper = [mean + ci for mean, ci in zip(means, cis)]
                ax.fill_between(xs, lower, upper, alpha=0.15, color=COLORS.get(policy))
                csv_rows.extend(
                    {
                        "policy": policy,
                        "x_value": x_order[row["scenario"]],
                        "x_label": label_order[row["scenario"]],
                        "mean": row[f"{metric}_mean"],
                        "ci95": row[f"{metric}_ci95"],
                    }
                    for row in rows
                )
            ax.set_title(f"{suite.title} | {title}")
            ax.set_xlabel(suite.x_label)
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.grid(alpha=0.25)
            ax.set_xticks(
                [scenario.x_value for scenario in suite.scenarios],
                [label_order[scenario.name] for scenario in suite.scenarios],
            )
            ax.legend()
            figure_path = output_dir / f"{suite.name}_{_metric_slug(metric)}.png"
            _save_figure(fig, figure_path)
            _save_figure_csv(figure_path, csv_rows)
    else:
        raise ValueError(f"Unknown plot kind: {suite.plot_kind}")


def plot_episode_details(
    step_results: list[StepResult],
    episode_results: list[EpisodeResult],
    output_dir: Path,
) -> None:
    if not step_results or not episode_results:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_steps: dict[tuple[str, str, str, str, int], list[StepResult]] = defaultdict(list)
    for row in step_results:
        grouped_steps[(row.backend, row.suite, row.scenario, row.policy, row.episode)].append(row)

    episode_lookup = {
        (row.backend, row.suite, row.scenario, row.policy, row.episode): row
        for row in episode_results
    }

    for key, rows in grouped_steps.items():
        rows.sort(key=lambda row: row.step)
        episode = episode_lookup.get(key)
        if episode is None:
            continue

        backend, suite, scenario, policy, episode_idx = key
        stem = f"{backend}_{suite}_{scenario}_{policy}_ep{episode_idx}"
        _plot_episode_progress(
            rows,
            episode,
            output_dir / stem,
        )
        _plot_episode_trajectories(
            rows,
            output_dir / f"{stem}_trajectory.png",
        )
        _plot_episode_tracking(
            rows,
            episode,
            output_dir / stem,
        )


def _plot_episode_progress(
    rows: list[StepResult],
    episode: EpisodeResult,
    output_base: Path,
) -> None:
    steps = [row.step + 1 for row in rows]
    cumulative_deliveries = []
    delivered_so_far = 0
    remaining_patients = []
    cumulative_reward = []
    reward_so_far = 0.0
    cumulative_hazard_entries = []
    hazard_so_far = 0
    cumulative_collisions = []
    collision_so_far = 0
    for row in rows:
        delivered_so_far += len(row.deliveries)
        reward_so_far += sum(row.rewards)
        hazard_so_far += (
            row.wind_entries_0
            + row.wind_entries_1
            + row.low_signal_entries_0
            + row.low_signal_entries_1
        )
        collision_so_far += row.obstacle_collisions + row.agent_collisions
        cumulative_deliveries.append(delivered_so_far)
        remaining_patients.append(row.remaining_patients)
        cumulative_reward.append(reward_so_far)
        cumulative_hazard_entries.append(hazard_so_far)
        cumulative_collisions.append(collision_so_far)

    delivery_steps = [row.step + 1 for row in rows if row.deliveries]
    title_prefix = (
        f"{episode.backend} | {episode.suite} | {episode.scenario} | {episode.policy} | ep {episode.episode}"
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(steps, cumulative_deliveries, color="#1b4965")
    for delivery_step in delivery_steps:
        ax.axvline(delivery_step, color="#1b4965", alpha=0.15, linewidth=1)
    ax.set_title(f"{title_prefix} | Cumulative Deliveries")
    ax.set_xlabel("Step")
    ax.set_ylabel("Deliveries")
    ax.grid(alpha=0.25)
    delivery_path = output_base.parent / f"{output_base.name}_cumulative_deliveries.png"
    _save_figure(fig, delivery_path)
    _save_figure_csv(
        delivery_path,
        [
            {
                "step": step,
                "cumulative_deliveries": total,
                "delivery_event": step in delivery_steps,
            }
            for step, total in zip(steps, cumulative_deliveries)
        ],
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(steps, remaining_patients, color="#9c6644")
    ax.set_title(f"{title_prefix} | Remaining Patients")
    ax.set_xlabel("Step")
    ax.set_ylabel("Remaining")
    ax.grid(alpha=0.25)
    remaining_path = output_base.parent / f"{output_base.name}_remaining_patients.png"
    _save_figure(fig, remaining_path)
    _save_figure_csv(
        remaining_path,
        [
            {"step": step, "remaining_patients": remaining}
            for step, remaining in zip(steps, remaining_patients)
        ],
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(steps, cumulative_reward, color="#4c956c")
    ax.set_title(f"{title_prefix} | Cumulative Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    reward_path = output_base.parent / f"{output_base.name}_cumulative_reward.png"
    _save_figure(fig, reward_path)
    _save_figure_csv(
        reward_path,
        [
            {"step": step, "cumulative_reward": reward}
            for step, reward in zip(steps, cumulative_reward)
        ],
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    drone0_battery = [row.drone0_battery for row in rows]
    drone1_battery = [row.drone1_battery for row in rows]
    ax.plot(steps, drone0_battery, label="drone0", color="#c1121f")
    ax.plot(steps, drone1_battery, label="drone1", color="#1d4ed8")
    ax.set_title(f"{title_prefix} | Battery Margin")
    ax.set_xlabel("Step")
    ax.set_ylabel("Battery")
    ax.grid(alpha=0.25)
    ax.legend()
    battery_path = output_base.parent / f"{output_base.name}_battery_margin.png"
    _save_figure(fig, battery_path)
    _save_figure_csv(
        battery_path,
        [
            {
                "step": step,
                "drone0_battery": battery0,
                "drone1_battery": battery1,
            }
            for step, battery0, battery1 in zip(steps, drone0_battery, drone1_battery)
        ],
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    target_distance_0 = [row.target_distance_0 for row in rows]
    target_distance_1 = [row.target_distance_1 for row in rows]
    ax.plot(steps, target_distance_0, label="drone0", color="#c1121f")
    ax.plot(steps, target_distance_1, label="drone1", color="#1d4ed8")
    ax.set_title(f"{title_prefix} | Distance to Current Goal")
    ax.set_xlabel("Step")
    ax.set_ylabel("Grid distance")
    ax.grid(alpha=0.25)
    ax.legend()
    distance_path = output_base.parent / f"{output_base.name}_distance_to_goal.png"
    _save_figure(fig, distance_path)
    _save_figure_csv(
        distance_path,
        [
            {
                "step": step,
                "drone0_distance": distance0,
                "drone1_distance": distance1,
            }
            for step, distance0, distance1 in zip(steps, target_distance_0, target_distance_1)
        ],
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(steps, cumulative_hazard_entries, label="hazard_entries", color="#d17b0f")
    ax.plot(steps, cumulative_collisions, label="collisions", color="#6b7280")
    ax.set_title(f"{title_prefix} | Risk Events")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative count")
    ax.grid(alpha=0.25)
    ax.legend()
    risk_path = output_base.parent / f"{output_base.name}_risk_events.png"
    _save_figure(fig, risk_path)
    _save_figure_csv(
        risk_path,
        [
            {
                "step": step,
                "hazard_entries": hazards,
                "collisions": collisions,
            }
            for step, hazards, collisions in zip(
                steps,
                cumulative_hazard_entries,
                cumulative_collisions,
            )
        ],
    )


def _plot_episode_trajectories(rows: list[StepResult], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    x0 = [row.sim_pos0_x for row in rows]
    y0 = [row.sim_pos0_y for row in rows]
    x1 = [row.sim_pos1_x for row in rows]
    y1 = [row.sim_pos1_y for row in rows]

    ax.plot(x0, y0, color="#c1121f", marker="o", markersize=2, linewidth=1.5, label="drone0")
    ax.plot(x1, y1, color="#1d4ed8", marker="o", markersize=2, linewidth=1.5, label="drone1")
    ax.scatter([x0[0], x1[0]], [y0[0], y1[0]], color=["#ff8fa3", "#93c5fd"], s=50, label="start")
    ax.scatter([x0[-1], x1[-1]], [y0[-1], y1[-1]], color=["#7f1d1d", "#1e3a8a"], s=50, label="end")
    ax.set_title("Simulated Grid Trajectory")
    ax.set_xlabel("Grid x")
    ax.set_ylabel("Grid y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.invert_yaxis()
    ax.legend()

    _save_figure(fig, output_path)
    _save_figure_csv(
        output_path,
        [
            {
                "step": row.step + 1,
                "drone0_x": row.sim_pos0_x,
                "drone0_y": row.sim_pos0_y,
                "drone1_x": row.sim_pos1_x,
                "drone1_y": row.sim_pos1_y,
            }
            for row in rows
        ],
    )


def _plot_episode_tracking(
    rows: list[StepResult],
    episode: EpisodeResult,
    output_base: Path,
) -> None:
    steps = [row.step + 1 for row in rows]
    sim0_east = [row.sim_pos0_x * METERS_PER_CELL for row in rows]
    sim0_north = [-row.sim_pos0_y * METERS_PER_CELL for row in rows]
    sim1_east = [row.sim_pos1_x * METERS_PER_CELL for row in rows]
    sim1_north = [-row.sim_pos1_y * METERS_PER_CELL for row in rows]
    title_prefix = (
        f"PX4 Tracking Detail | {episode.suite} | {episode.scenario} | {episode.policy} | ep {episode.episode}"
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(sim0_east, sim0_north, color="#fca5a5", linewidth=2, label="drone0 simulated")
    ax.plot(
        [row.drone0_east for row in rows],
        [row.drone0_north for row in rows],
        color="#c1121f",
        linewidth=1.5,
        label="drone0 actual",
    )
    ax.plot(sim1_east, sim1_north, color="#93c5fd", linewidth=2, label="drone1 simulated")
    ax.plot(
        [row.drone1_east for row in rows],
        [row.drone1_north for row in rows],
        color="#1d4ed8",
        linewidth=1.5,
        label="drone1 actual",
    )
    ax.set_title(f"{title_prefix} | Actual vs Sim Trajectory")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    tracking_path = output_base.parent / f"{output_base.name}_actual_vs_sim_trajectory.png"
    _save_figure(fig, tracking_path)
    _save_figure_csv(
        tracking_path,
        [
            {
                "step": step,
                "drone0_sim_east": d0_sim_e,
                "drone0_sim_north": d0_sim_n,
                "drone0_actual_east": row.drone0_east,
                "drone0_actual_north": row.drone0_north,
                "drone1_sim_east": d1_sim_e,
                "drone1_sim_north": d1_sim_n,
                "drone1_actual_east": row.drone1_east,
                "drone1_actual_north": row.drone1_north,
            }
            for step, row, d0_sim_e, d0_sim_n, d1_sim_e, d1_sim_n in zip(
                steps,
                rows,
                sim0_east,
                sim0_north,
                sim1_east,
                sim1_north,
            )
        ],
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    tracking_error_0 = [row.tracking_error_m_0 for row in rows]
    tracking_error_1 = [row.tracking_error_m_1 for row in rows]
    ax.plot(steps, tracking_error_0, color="#c1121f", label="drone0")
    ax.plot(steps, tracking_error_1, color="#1d4ed8", label="drone1")
    ax.set_title(
        f"{title_prefix} | Tracking Error to Simulated Grid State "
        f"(mean={episode.mean_tracking_error_m:.2f}m, max={episode.max_tracking_error_m:.2f}m)"
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Position error (m)")
    ax.grid(alpha=0.25)
    ax.legend()
    error_path = output_base.parent / f"{output_base.name}_tracking_error.png"
    _save_figure(fig, error_path)
    _save_figure_csv(
        error_path,
        [
            {
                "step": step,
                "drone0_tracking_error_m": error0,
                "drone1_tracking_error_m": error1,
            }
            for step, error0, error1 in zip(steps, tracking_error_0, tracking_error_1)
        ],
    )


def write_episode_csv(path: Path, results: list[EpisodeResult]) -> None:
    rows = [asdict(result) for result in results]
    write_csv(path, rows)


def write_step_csv(path: Path, results: list[StepResult]) -> None:
    rows = [asdict(result) for result in results]
    write_csv(path, rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric_slug(metric: str) -> str:
    return metric.replace("_rate", "").replace("_", "-")


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_figure_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(output_path.with_suffix(".csv"), rows)


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def log_episode_completion(result: EpisodeResult, total_episodes: int) -> None:
    logger.info(
        "Completed [%s] %s/%s/%s episode %d/%d seed=%d delivered=%d/%d died=%d "
        "eff=%.2f reward=%.2f success=%.0f steps=%d",
        result.backend,
        result.suite,
        result.scenario,
        result.policy,
        result.episode + 1,
        total_episodes,
        result.seed,
        result.patients_delivered,
        result.patients_spawned,
        result.patients_died,
        result.triage_efficiency,
        result.total_reward,
        result.mission_success_rate,
        result.steps,
    )


def world_summary(world: dict[str, Any]) -> str:
    hazards = world.get("hazards", {})
    battery = world.get("battery", {})
    return (
        f"hazards(wind={hazards.get('num_wind_zones', 'default')},"
        f" ls={hazards.get('num_low_signal_zones', 'default')},"
        f" fail={hazards.get('low_signal_failure_prob', 'default')}) "
        f"battery(initial={battery.get('initial', 'default')})"
    )


if __name__ == "__main__":
    main()
