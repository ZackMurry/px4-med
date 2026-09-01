"""Heuristic baseline policies for experiment comparisons (5-drone mission).

Baselines act on the world state directly (not the model observation dict) but
expose the same `select_actions(observation)` interface as FGCSPolicy so the
Coordinator can drive either. Actions: 0=N 1=S 2=W 3=E 4=hover 5=land.

Compared to the 2-drone paper baselines, these add two behaviors the 5-drone
mission requires: distinct target claiming (so the fleet doesn't converge on
one patient) and safe-return-to-pad using the same energy model the learned
policy observes (otherwise every baseline dies at battery exhaustion).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import random
from typing import Mapping, Optional

from .environment import (
    ACTION_HOVER,
    ACTION_LAND,
    WorldEnvironment,
)

_MOVE_TOWARD_FAILED = ACTION_HOVER


@dataclass
class BaselinePolicy:
    """Deterministic or seeded heuristic controller bound to a world."""

    name: str
    rng: random.Random
    world: WorldEnvironment
    device: str = "heuristic"
    action_names: tuple = ("north", "south", "west", "east", "hover", "land")
    _path_cache: dict = field(default_factory=dict)

    def select_actions(self, observation: Optional[Mapping] = None) -> list[int]:
        # observation is accepted for interface parity and ignored.
        self._path_cache.clear()
        world = self.world
        n = world.num_drones

        if self.name == "random":
            actions = []
            for i in range(n):
                mask = world.action_mask(i)
                valid = [a for a, ok in enumerate(mask) if ok]
                actions.append(self.rng.choice(valid))
            return actions

        assignments = self._assign_targets()
        actions: list[int] = []
        for i in range(n):
            actions.append(self._drone_action(i, assignments.get(i)))
        return actions

    # ------------------------------------------------------------------

    def _returning(self, agent_idx: int) -> bool:
        world = self.world
        return world.landing_phase() or world.return_required(agent_idx)

    def _assign_targets(self) -> dict[int, int]:
        """Assign distinct pending patients to non-returning drones."""
        world = self.world
        available_drones = [
            i for i in range(world.num_drones)
            if not world.landed[i] and not world.depleted[i] and not self._returning(i)
        ]
        pending = [p for p in world.patients if p.pending]
        assignments: dict[int, int] = {}
        if not pending or not available_drones:
            return assignments

        if self.name == "nearest_path":
            # Greedy globally-nearest (drone, patient) pairs.
            pairs = []
            for i in available_drones:
                pos = world.agent_grids[i]
                for p in pending:
                    d = self._path_distance(pos, world.patient_grid(p.idx))
                    pairs.append((d, i, p.idx))
            pairs.sort()
            used_drones: set[int] = set()
            used_patients: set[int] = set()
            for d, i, pidx in pairs:
                if math.isinf(d) or i in used_drones or pidx in used_patients:
                    continue
                assignments[i] = pidx
                used_drones.add(i)
                used_patients.add(pidx)
            # Leftover drones (more drones than patients) chase nearest anyway.
            for i in available_drones:
                if i not in assignments:
                    best = min(
                        pending,
                        key=lambda p: (
                            self._path_distance(world.agent_grids[i], world.patient_grid(p.idx)),
                            p.idx,
                        ),
                    )
                    assignments[i] = best.idx
            return assignments

        if self.name == "priority_path":
            # Patients by (weight desc, timer asc); each claims nearest free drone.
            ordered = sorted(
                pending,
                key=lambda p: (-int(p.weight), float(p.timer), p.idx),
            )
            free = set(available_drones)
            for p in ordered:
                if not free:
                    break
                target_grid = world.patient_grid(p.idx)
                best = min(
                    free,
                    key=lambda i: (
                        self._path_distance(world.agent_grids[i], target_grid),
                        i,
                    ),
                )
                if math.isinf(self._path_distance(world.agent_grids[best], target_grid)):
                    continue
                assignments[best] = p.idx
                free.discard(best)
            # Leftover drones chase the highest-priority patient reachable.
            for i in available_drones:
                if i not in assignments and ordered:
                    assignments[i] = ordered[0].idx
            return assignments

        raise ValueError(f"Unknown baseline policy: {self.name}")

    def _drone_action(self, agent_idx: int, patient_idx: Optional[int]) -> int:
        world = self.world
        mask = world.action_mask(agent_idx)
        if world.landed[agent_idx] or world.depleted[agent_idx]:
            return ACTION_HOVER
        if mask[ACTION_LAND] and not any(mask[:ACTION_LAND]):
            return ACTION_LAND

        pos = world.agent_grids[agent_idx]
        if self._returning(agent_idx) or patient_idx is None:
            goal = world.landing_grid(agent_idx)
            if pos == goal:
                # Waiting on pad (e.g. rescue phase but return required not
                # yet triggering land-only mask): hover in place.
                return ACTION_LAND if mask[ACTION_LAND] else ACTION_HOVER
            return self._step_toward(pos, goal)

        return self._step_toward(pos, world.patient_grid(patient_idx))

    # ------------------------------------------------------------------

    def _step_toward(self, start: tuple[int, int], goal: tuple[int, int]) -> int:
        path = self._shortest_path(start, goal)
        if len(path) < 2:
            return _MOVE_TOWARD_FAILED
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
        return _MOVE_TOWARD_FAILED

    def _path_distance(self, start: tuple[int, int], goal: tuple[int, int]) -> float:
        path = self._shortest_path(start, goal)
        return float(max(0, len(path) - 1)) if path else math.inf

    def _shortest_path(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        s = (int(start[0]), int(start[1]))
        g = (int(goal[0]), int(goal[1]))
        cached = self._path_cache.get((s, g))
        if cached is not None:
            return cached
        path = _bfs_path(self.world, s, g)
        self._path_cache[(s, g)] = path
        return path


def make_baseline(name: str, seed: int, world: WorldEnvironment) -> BaselinePolicy:
    return BaselinePolicy(name=name, rng=random.Random(seed), world=world)


def _bfs_path(
    world: WorldEnvironment,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    if start == goal:
        return [start]

    grid_size = world.grid_size
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
    current: tuple[int, int] = goal
    while True:
        parent = came_from[current]
        if parent is None:
            break
        current = parent
        path.append(current)
    path.reverse()
    return path
