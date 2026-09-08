# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import pytest

from motifshift.spatiotemporal.data import (
    TrajectoryPoint,
    flatten_scenes,
    generate_scene,
    generate_synthetic_scenes,
    predictive_trajectory_steps,
)


def test_generate_scene_has_expected_shape_and_dataclass_points() -> None:
    scene = generate_scene("toy", n_agents=2, n_steps=3, origin=(10.0, -5.0))

    assert scene.scene_id == "toy"
    assert len(scene.points) == 6
    assert isinstance(scene.points[0], TrajectoryPoint)
    assert scene.points[0] == TrajectoryPoint(scene_id="toy", agent_id=0, t=0, x=10.0, y=-5.0)


def test_generate_scene_reuses_motion_patterns_by_agent() -> None:
    scene = generate_scene("toy", n_agents=3, n_steps=3, origin=(0.0, 0.0))
    by_agent = {
        agent_id: [point for point in scene.points if point.agent_id == agent_id]
        for agent_id in range(3)
    }

    assert [(point.x, point.y) for point in by_agent[0]] == [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
    ]
    assert [(point.x, point.y) for point in by_agent[1]] == [
        (2.0, 2.0),
        (2.0, 3.0),
        (2.0, 4.0),
    ]
    assert [(point.x, point.y) for point in by_agent[2]] == [
        (4.0, 4.0),
        (5.0, 5.0),
        (6.0, 6.0),
    ]


def test_generate_synthetic_scenes_is_deterministic_and_uses_distinct_layouts() -> None:
    scenes_a = generate_synthetic_scenes(n_scenes=2, n_agents=2, n_steps=3, seed=7)
    scenes_b = generate_synthetic_scenes(n_scenes=2, n_agents=2, n_steps=3, seed=7)

    assert scenes_a == scenes_b
    assert len(scenes_a) == 2
    assert scenes_a[0].points[0].scene_id == "scene_0"
    assert scenes_a[1].points[0].scene_id == "scene_1"
    assert (scenes_a[0].points[0].x, scenes_a[0].points[0].y) != (
        scenes_a[1].points[0].x,
        scenes_a[1].points[0].y,
    )


def test_flatten_scenes_preserves_total_points() -> None:
    scenes = generate_synthetic_scenes(n_scenes=2, n_agents=3, n_steps=4, seed=123)

    assert len(flatten_scenes(scenes)) == 2 * 3 * 4


def test_predictive_trajectory_steps_use_next_displacement_target() -> None:
    scene = generate_scene("toy", n_agents=1, n_steps=4, origin=(0.0, 0.0))

    steps = predictive_trajectory_steps(scene)

    assert len(steps) == 2
    assert steps[0]["prev_dx"] == 1.0
    assert steps[0]["prev_dy"] == 0.0
    assert steps[0]["dx"] == steps[0]["prev_dx"]
    assert steps[0]["target_dx"] == 1.0
    assert steps[0]["target_dy"] == 0.0
    assert steps[0]["t"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_agents": 0},
        {"n_steps": 0},
        {"agent_spacing": 0.0},
        {"motion_patterns": ()},
    ],
)
def test_generate_scene_invalid_arguments_raise_value_error(kwargs) -> None:
    with pytest.raises(ValueError):
        generate_scene("bad", **kwargs)


def test_generate_synthetic_scenes_requires_positive_scene_count() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_scenes(n_scenes=0)
