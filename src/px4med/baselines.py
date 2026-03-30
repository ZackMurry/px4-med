"""Heuristic baseline policies for experiment comparisons."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random

from .environment import WorldEnvironment


@dataclass
class BaselinePolicy:
    """Simple deterministic or seeded heuristic controller."""

    name: str
    rng: random.Random

    def select_actions(self, world: WorldEnvironment) -> list[int]:
        return [
            _select_action(self.name, self.rng, world, agent_idx)
            for agent_idx in range(2)
        ]


def make_baseline(name: str, seed: int) -> BaselinePolicy:
    return BaselinePolicy(name=name, rng=random.Random(seed))


def _select_action(
    name: str,
    rng: random.Random,
    world: WorldEnvironment,
    agent_idx: int,
) -> int:
    pos = world.agent_grids[agent_idx]
    landing = world.landing_grid(agent_idx)

    if world.landed[agent_idx]:
        return 4

    candidates = [p for p in world.patients if p.active and not p.delivered]
    if not candidates:
        return _step_toward(world, pos, landing)

    if name == "random":
        return rng.randrange(5)

    if name == "nearest_path":
        target = min(
            candidates,
            key=lambda p: (_path_distance(world, pos, world.patient_grid(p.idx)), p.idx),
        )
        return _step_toward(world, pos, world.patient_grid(target.idx))

    if name == "priority_path":
        target = max(
            candidates,
            key=lambda p: (
                int(p.weight),
                -_path_distance(world, pos, world.patient_grid(p.idx)),
                float(p.timer),
                -p.idx,
            ),
        )
        return _step_toward(world, pos, world.patient_grid(target.idx))

    raise ValueError(f"Unknown baseline policy: {name}")


def _step_toward(
    world: WorldEnvironment,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> int:
    if start == goal:
        return 4 if goal in [world.landing_grid(0), world.landing_grid(1)] else -1

    path = _shortest_path(world, start, goal)
    if len(path) < 2:
        return -1

    next_x, next_y = path[1]
    dx = next_x - start[0]
    dy = next_y - start[1]
    if dx == 1:
        return 3
    if dx == -1:
        return 2
    if dy == 1:
        return 1
    if dy == -1:
        return 0
    return -1


def _path_distance(
    world: WorldEnvironment,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> int:
    path = _shortest_path(world, start, goal)
    return max(0, len(path) - 1) if path else math.inf


def _shortest_path(
    world: WorldEnvironment,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    if start == goal:
        return [start]

    grid_size = int(world.config.get("grid", {}).get("size", 50))
    queue: deque[tuple[int, int]] = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (x + dx, y + dy)
            if nxt in came_from:
                continue
            if nxt[0] < 0 or nxt[0] >= grid_size or nxt[1] < 0 or nxt[1] >= grid_size:
                continue
            if nxt in world.obstacles:
                continue
            came_from[nxt] = current
            queue.append(nxt)

    if goal not in came_from:
        return []

    path = [goal]
    current = goal
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
