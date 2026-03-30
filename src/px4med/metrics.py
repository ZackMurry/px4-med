"""Per-step and per-episode telemetry collection. Outputs jsonlines files.

Mirrors Data_Collection from AneeshMARL5.py but writes structured records
incrementally rather than accumulating lists in memory.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

try:
    import jsonlines  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    jsonlines = None


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._handle = path.open("w", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        self._handle.write(json.dumps(obj) + "\n")

    def close(self) -> None:
        self._handle.close()


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
    remaining_patients: int
    target_distances: list[int]
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
        step_path = self.log_dir / f"{run_id}_steps.jsonl"
        episode_path = self.log_dir / f"{run_id}_episodes.jsonl"
        if jsonlines is not None:
            self._step_writer = jsonlines.open(step_path, mode="w")
            self._episode_writer = jsonlines.open(episode_path, mode="w")
        else:
            self._step_writer = _JsonlWriter(step_path)
            self._episode_writer = _JsonlWriter(episode_path)

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
