# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import math

import pytest

from motifshift.spatiotemporal.predictors import (
    ConstantVelocityPredictor,
    KNNDisplacementPredictor,
    MeanDisplacementPredictor,
    MotionTokenMarkovPredictor,
    SituationMotionTokenPredictor,
    aggregate_error_results,
    displacement_errors,
    evaluate_leave_one_scene_prediction,
    summarize_errors,
)


def _step(
    scene_id: str = "scene_0",
    agent_id: int = 0,
    t: int = 0,
    x: float = 0.0,
    y: float = 0.0,
    dx: float = 1.0,
    dy: float = 0.0,
) -> dict:
    return {
        "scene_id": scene_id,
        "agent_id": agent_id,
        "t": t,
        "x": x,
        "y": y,
        "next_x": x + dx,
        "next_y": y + dy,
        "dx": dx,
        "dy": dy,
    }


def test_constant_velocity_predictor_returns_step_displacement() -> None:
    step = _step(dx=2.0, dy=-1.0)

    predictor = ConstantVelocityPredictor().fit([step])

    assert predictor.predict(step) == (2.0, -1.0)
    assert predictor.predict_batch([step]) == [(2.0, -1.0)]


def test_constant_velocity_uses_previous_motion_for_predictive_records() -> None:
    step = {
        **_step(dx=2.0, dy=-1.0),
        "prev_dx": 0.5,
        "prev_dy": 0.25,
        "target_dx": 1.5,
        "target_dy": 0.0,
    }

    predictor = ConstantVelocityPredictor().fit([step])

    assert predictor.predict(step) == (0.5, 0.25)
    assert displacement_errors([predictor.predict(step)], [step])[0] > 0


def test_mean_displacement_predictor_returns_training_mean() -> None:
    steps = [_step(dx=1.0, dy=0.0), _step(dx=3.0, dy=2.0)]

    predictor = MeanDisplacementPredictor().fit(steps)

    assert predictor.predict(_step(dx=9.0, dy=9.0)) == pytest.approx((2.0, 1.0))


def test_mean_displacement_predictor_fits_target_displacements() -> None:
    steps = [
        {**_step(dx=1.0, dy=0.0), "target_dx": 2.0, "target_dy": 0.0},
        {**_step(dx=3.0, dy=2.0), "target_dx": 4.0, "target_dy": 2.0},
    ]

    predictor = MeanDisplacementPredictor().fit(steps)

    assert predictor.predict(_step(dx=9.0, dy=9.0)) == pytest.approx((3.0, 1.0))


def test_knn_displacement_predictor_works_on_toy_steps() -> None:
    train_steps = [
        _step(x=0.0, y=0.0, dx=1.0, dy=0.0),
        _step(x=10.0, y=10.0, dx=0.0, dy=1.0),
    ]

    predictor = KNNDisplacementPredictor(k=1).fit(train_steps)

    assert predictor.predict(_step(x=0.1, y=0.0, dx=1.0, dy=0.0)) == pytest.approx((1.0, 0.0))


def test_knn_displacement_predictor_rejects_empty_fit() -> None:
    with pytest.raises(ValueError):
        KNNDisplacementPredictor().fit([])


def test_token_predictors_return_one_prediction_per_step() -> None:
    steps = [
        _step("scene_0", 0, 0, 0.0, 0.0, 1.0, 0.0),
        _step("scene_0", 1, 0, 0.0, 1.0, 1.0, 0.0),
        _step("scene_0", 0, 1, 1.0, 0.0, 1.0, 0.0),
        _step("scene_0", 1, 1, 1.0, 1.0, 1.0, 0.0),
    ]

    for predictor in [
        MotionTokenMarkovPredictor().fit(steps),
        SituationMotionTokenPredictor().fit(steps),
    ]:
        predictions = predictor.predict_batch(steps)
        assert len(predictions) == len(steps)
        assert all(len(prediction) == 2 for prediction in predictions)


def test_displacement_errors_and_summary() -> None:
    steps = [_step(dx=1.0, dy=0.0), _step(dx=0.0, dy=1.0)]
    predictions = [(1.0, 0.0), (0.0, 3.0)]

    errors = displacement_errors(predictions, steps)
    summary = summarize_errors(errors)

    assert errors == pytest.approx([0.0, 2.0])
    assert summary["mean_error"] == pytest.approx(1.0)
    assert summary["median_error"] == pytest.approx(1.0)
    assert summary["n_steps"] == pytest.approx(2.0)


def test_displacement_errors_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        displacement_errors([], [_step()])


def test_summarize_errors_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        summarize_errors([])


def test_evaluate_leave_one_scene_prediction() -> None:
    steps_by_scene = {
        "scene_0": [_step("scene_0", dx=1.0), _step("scene_0", t=1, dx=1.0)],
        "scene_1": [_step("scene_1", dx=1.0), _step("scene_1", t=1, dx=1.0)],
    }

    results = evaluate_leave_one_scene_prediction(steps_by_scene, ConstantVelocityPredictor)

    assert set(results) == {"scene_0", "scene_1"}
    assert all(math.isfinite(row["mean_error"]) for row in results.values())


def test_aggregate_error_results() -> None:
    mean, std = aggregate_error_results(
        {
            "scene_0": {"mean_error": 0.0, "median_error": 0.0, "n_steps": 2.0},
            "scene_1": {"mean_error": 2.0, "median_error": 2.0, "n_steps": 4.0},
        }
    )

    assert mean["mean_error"] == pytest.approx(1.0)
    assert std["mean_error"] == pytest.approx(1.0)
    assert mean["n_steps"] == pytest.approx(3.0)


def test_aggregate_error_results_handles_empty_input() -> None:
    assert aggregate_error_results({}) == ({}, {})
