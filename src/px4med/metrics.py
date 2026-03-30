"""Per-step and per-episode telemetry collection. Outputs jsonlines files.

Mirrors Data_Collection from AneeshMARL5.py but writes structured records
incrementally rather than accumulating lists in memory.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jsonlines


@dataclass
class StepRecord:
    episode: int
    step: int
    timestamp: float
    drone0_north: float
    drone0_east: float
    drone0_battery: float
    drone1_north: float
    drone1_east: float
    drone1_battery: float
    actions: list[int]
    deliveries: list[int]  # patient indices delivered this step
    rewards: list[float]
    simulated_positions: list[list[int]]
    wind_entries: list[int]
    low_signal_entries: list[int]
    obstacle_collisions: int
    agent_collisions: int
    landing_attempts: list[bool]
    landed_this_step: list[bool]


@dataclass
class EpisodeRecord:
    episode: int
    steps: int
    patients_delivered: int
    patients_died: int
    patients_spawned: int
    both_landed: bool
    battery_remaining: list[float]
    simulated_battery_remaining: list[float]
    total_reward: float
    triage_efficiency: float
    wind_entries: list[int]
    low_signal_entries: list[int]
    obstacle_collisions: int
    agent_collisions: int


class MetricsCollector:
    """Writes step and episode records to jsonlines files."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._step_writer: Any = None
        self._episode_writer: Any = None

    def open(self, run_id: str) -> None:
        self._step_writer = jsonlines.open(
            self.log_dir / f"{run_id}_steps.jsonl", mode="w"
        )
        self._episode_writer = jsonlines.open(
            self.log_dir / f"{run_id}_episodes.jsonl", mode="w"
        )

    def log_step(self, record: StepRecord) -> None:
        if self._step_writer:
            self._step_writer.write(asdict(record))

    def log_episode(self, record: EpisodeRecord) -> None:
        if self._episode_writer:
            self._episode_writer.write(asdict(record))

    def close(self) -> None:
        if self._step_writer:
            self._step_writer.close()
        if self._episode_writer:
            self._episode_writer.close()
