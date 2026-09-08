# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import pytest

from motifshift.rl.gridworld import (
    ACTIONS,
    GridWorld,
    action_from_to,
    classify_situation,
    generate_episodes,
    generate_gridworlds,
    in_bounds,
    is_free,
    local_wall_pattern,
    neighbors4,
    shortest_path,
    step_position,
)

EXPECTED_SITUATIONS = {
    "goal",
    "dead_end",
    "corridor",
    "corner",
    "junction",
    "open_area",
    "other",
}


def test_generate_gridworlds_is_deterministic() -> None:
    worlds_a = generate_gridworlds(n_layouts=3, width=8, height=8, seed=123)
    worlds_b = generate_gridworlds(n_layouts=3, width=8, height=8, seed=123)

    assert worlds_a == worlds_b


def test_different_seeds_change_some_layout() -> None:
    worlds_a = generate_gridworlds(n_layouts=3, width=8, height=8, seed=123)
    worlds_b = generate_gridworlds(n_layouts=3, width=8, height=8, seed=456)

    assert any(a.walls != b.walls for a, b in zip(worlds_a, worlds_b))


def test_generated_layouts_have_valid_paths() -> None:
    worlds = generate_gridworlds(n_layouts=4, width=10, height=10, seed=7)

    assert len(worlds) == 4
    for index, world in enumerate(worlds):
        assert world.layout_id == f"layout_{index}"
        assert is_free(world, world.start)
        assert is_free(world, world.goal)
        path = shortest_path(world)
        assert path
        assert path[0] == world.start
        assert path[-1] == world.goal


def test_in_bounds_is_free_and_neighbors() -> None:
    world = GridWorld(
        layout_id="toy",
        width=3,
        height=3,
        walls=frozenset({(1, 0)}),
        start=(0, 0),
        goal=(2, 2),
    )

    assert in_bounds(world, (0, 0))
    assert not in_bounds(world, (-1, 0))
    assert is_free(world, (0, 0))
    assert not is_free(world, (1, 0))
    assert set(neighbors4(world, (0, 0))) == {(0, 1)}


def test_step_position_respects_walls_and_boundaries() -> None:
    world = GridWorld(
        layout_id="toy",
        width=3,
        height=3,
        walls=frozenset({(1, 0)}),
        start=(0, 0),
        goal=(2, 2),
    )

    assert step_position(world, (0, 0), "left") == (0, 0)
    assert step_position(world, (0, 0), "right") == (0, 0)
    assert step_position(world, (0, 0), "down") == (0, 1)


def test_local_wall_pattern_and_situation_classification() -> None:
    world = generate_gridworlds(n_layouts=1, width=8, height=8, seed=123)[0]
    x, y = shortest_path(world)[1]

    assert len(local_wall_pattern(world, x, y, radius=1)) == 9
    assert classify_situation(world, x, y) in EXPECTED_SITUATIONS
    assert classify_situation(world, *world.goal) == "goal"


def test_action_from_to() -> None:
    assert action_from_to((1, 1), (1, 0)) == "up"
    assert action_from_to((1, 1), (1, 2)) == "down"
    assert action_from_to((1, 1), (0, 1)) == "left"
    assert action_from_to((1, 1), (2, 1)) == "right"
    assert action_from_to((1, 1), (1, 1)) == "stay"
    with pytest.raises(ValueError):
        action_from_to((1, 1), (3, 1))


def test_generate_episodes_returns_valid_deterministic_steps() -> None:
    worlds = generate_gridworlds(n_layouts=2, width=8, height=8, seed=123)
    steps_a = generate_episodes(worlds, episodes_per_world=3, max_steps=25, seed=11, noise_prob=0.1)
    steps_b = generate_episodes(worlds, episodes_per_world=3, max_steps=25, seed=11, noise_prob=0.1)

    assert steps_a
    assert steps_a == steps_b

    worlds_by_id = {world.layout_id: world for world in worlds}
    for step in steps_a:
        world = worlds_by_id[step.layout_id]
        assert step.action in ACTIONS
        assert is_free(world, (step.x, step.y))
        assert is_free(world, (step.next_x, step.next_y))
        assert step_position(world, (step.x, step.y), step.action) == (
            step.next_x,
            step.next_y,
        )
        assert step.done == ((step.next_x, step.next_y) == world.goal)


def test_generate_episodes_raises_for_world_without_path() -> None:
    world = GridWorld(
        layout_id="blocked",
        width=3,
        height=3,
        walls=frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
        start=(0, 0),
        goal=(2, 2),
    )

    with pytest.raises(ValueError):
        generate_episodes([world])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_layouts": 0},
        {"width": 3},
        {"height": 3},
    ],
)
def test_generate_gridworlds_invalid_args_raise_value_error(kwargs) -> None:
    with pytest.raises(ValueError):
        generate_gridworlds(**kwargs)
