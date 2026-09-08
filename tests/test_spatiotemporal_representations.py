# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import pytest

from motifshift.representations.spatiotemporal import (
    absolute_grid_tokens,
    motion_tokens,
    relative_displacement_grid_tokens,
    situation_motion_tokens,
)
from motifshift.spatiotemporal.data import TrajectoryPoint, generate_scene


def test_absolute_grid_tokens_quantize_positions() -> None:
    points = [
        TrajectoryPoint("s", 0, 0, 0.2, 1.9),
        TrajectoryPoint("s", 0, 1, 1.1, 2.0),
    ]

    assert absolute_grid_tokens(points, cell_size=1.0) == [
        ("abs", 0, 1),
        ("abs", 1, 2),
    ]


def test_relative_displacement_grid_tokens_are_translation_invariant() -> None:
    scene_a = generate_scene("a", n_agents=1, n_steps=3, origin=(0.0, 0.0))
    scene_b = generate_scene("b", n_agents=1, n_steps=3, origin=(10.0, -10.0))

    assert relative_displacement_grid_tokens(scene_a.points) == [
        ("rel", 1, 0),
        ("rel", 1, 0),
    ]
    assert relative_displacement_grid_tokens(scene_a.points) == (
        relative_displacement_grid_tokens(scene_b.points)
    )


def test_motion_tokens_include_speed_and_direction_bins() -> None:
    scene = generate_scene("toy", n_agents=2, n_steps=2, origin=(0.0, 0.0))

    tokens = motion_tokens(scene.points, speed_bins=(0.5, 1.5), direction_bins=8)

    assert tokens == [
        ("motion", 1, 0),
        ("motion", 1, 2),
    ]


def test_situation_motion_tokens_include_context_bins() -> None:
    points = [
        TrajectoryPoint("s", 0, 0, 0.0, 0.0),
        TrajectoryPoint("s", 1, 0, 1.0, 0.0),
        TrajectoryPoint("s", 0, 1, 1.0, 0.0),
        TrajectoryPoint("s", 1, 1, 1.0, 1.0),
    ]

    tokens = situation_motion_tokens(
        points,
        speed_bins=(0.5, 1.5),
        direction_bins=8,
        distance_bins=(0.5, 1.5),
        density_bins=(0.0, 1.0, 2.0),
        density_radius=1.5,
    )

    assert tokens == [
        ("situation_motion", 1, 0, 1, 1),
        ("situation_motion", 1, 2, 1, 1),
    ]


def test_tokenizers_return_empty_for_no_transitions() -> None:
    points = [TrajectoryPoint("s", 0, 0, 0.0, 0.0)]

    assert absolute_grid_tokens([]) == []
    assert relative_displacement_grid_tokens(points) == []
    assert motion_tokens(points) == []
    assert situation_motion_tokens(points) == []


@pytest.mark.parametrize(
    "call",
    [
        lambda: absolute_grid_tokens([], cell_size=0.0),
        lambda: relative_displacement_grid_tokens([], cell_size=-1.0),
        lambda: motion_tokens([], direction_bins=0),
        lambda: motion_tokens(
            [
                TrajectoryPoint("s", 0, 0, 0.0, 0.0),
                TrajectoryPoint("s", 0, 1, 1.0, 0.0),
            ],
            speed_bins=(2.0, 1.0),
        ),
        lambda: situation_motion_tokens([], density_radius=-1.0),
    ],
)
def test_invalid_tokenizer_arguments_raise_value_error(call) -> None:
    with pytest.raises(ValueError):
        call()
