from __future__ import annotations

from collections import deque

from px4med.coordinator import Coordinator
from px4med.environment import WorldEnvironment


def _coordinator() -> Coordinator:
    world = WorldEnvironment({"grid": {"size": 50, "meters_per_cell": 2.0}})
    world.reset()
    return Coordinator(
        drones=[],
        policy=object(),
        world=world,
        metrics=object(),
    )


def test_detects_square_loop():
    coordinator = _coordinator()
    coordinator._position_history[0] = deque(
        [(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)],
        maxlen=8,
    )
    assert coordinator._is_square_loop(0) is True


def test_detects_two_point_loop():
    coordinator = _coordinator()
    coordinator._position_history[0] = deque(
        [(10, 10), (11, 10), (10, 10), (11, 10)],
        maxlen=8,
    )
    assert coordinator._is_two_point_loop(0) is True
    assert coordinator._is_square_loop(0) is True


def test_escape_action_changes_looping_move():
    coordinator = _coordinator()
    coordinator.world.agent_grids[0] = (10, 10)
    coordinator.world.patient_grid = lambda idx: (14, 10)  # type: ignore[method-assign]
    coordinator.world.nearest_undelivered_patient = lambda pos: 0  # type: ignore[method-assign]
    coordinator._position_history[0] = deque(
        [(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)],
        maxlen=8,
    )

    override = coordinator._choose_escape_action(0, current_action=1)

    assert override in {0, 2, 3}
    assert override != 1


def test_escape_action_changes_two_point_looping_move():
    coordinator = _coordinator()
    coordinator.world.agent_grids[0] = (10, 10)
    coordinator.world.patient_grid = lambda idx: (14, 10)  # type: ignore[method-assign]
    coordinator.world.nearest_undelivered_patient = lambda pos: 0  # type: ignore[method-assign]
    coordinator._position_history[0] = deque(
        [(10, 10), (11, 10), (10, 10), (11, 10)],
        maxlen=8,
    )

    override = coordinator._choose_escape_action(0, current_action=3)

    assert override in {0, 1, 2}
    assert override != 3
