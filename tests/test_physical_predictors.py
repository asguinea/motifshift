# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import pytest

from motifshift.physical.field import (
    field_records,
    generate_physical_field_environments,
)
from motifshift.physical.predictors import (
    LocalPatchTokenPredictor,
    MeanValuePredictor,
    RawCellLookupPredictor,
    RegimeTokenPredictor,
    ValueChangeTokenPredictor,
    absolute_errors,
    aggregate_prediction_results,
    evaluate_leave_one_environment_prediction,
    summarize_absolute_errors,
)


def _toy_records():
    return [
        {
            "env_id": "field_0",
            "t": 0,
            "x": 0,
            "y": 0,
            "value": 0.2,
            "next_value": 0.3,
            "delta": 0.1,
            "north": 0.2,
            "south": 0.3,
            "east": 0.25,
            "west": 0.2,
            "grad_x": 0.025,
            "grad_y": 0.05,
            "grad_mag": 0.056,
        },
        {
            "env_id": "field_0",
            "t": 0,
            "x": 1,
            "y": 0,
            "value": 0.4,
            "next_value": 0.5,
            "delta": 0.1,
            "north": 0.4,
            "south": 0.45,
            "east": 0.5,
            "west": 0.3,
            "grad_x": 0.1,
            "grad_y": 0.025,
            "grad_mag": 0.103,
        },
    ]


def test_physical_predictors_fit_and_predict() -> None:
    records = _toy_records()

    for predictor in [
        MeanValuePredictor(),
        RawCellLookupPredictor(),
        ValueChangeTokenPredictor(),
        LocalPatchTokenPredictor(),
        RegimeTokenPredictor(),
    ]:
        fitted = predictor.fit(records)
        predictions = fitted.predict_batch(records)
        assert len(predictions) == len(records)
        assert all(0 <= prediction <= 1 for prediction in predictions)


def test_physical_predictors_empty_fit_raises_value_error() -> None:
    with pytest.raises(ValueError):
        MeanValuePredictor().fit([])
    with pytest.raises(ValueError):
        RawCellLookupPredictor().fit([])


def test_absolute_error_helpers() -> None:
    records = _toy_records()
    errors = absolute_errors([0.2, 0.7], records)
    summary = summarize_absolute_errors(errors)

    assert errors == pytest.approx([0.1, 0.2])
    assert summary["mean_abs_error"] == pytest.approx(0.15)
    assert summary["median_abs_error"] == pytest.approx(0.15)
    assert summary["n_records"] == pytest.approx(2.0)


def test_leave_one_environment_prediction_and_aggregation() -> None:
    envs = generate_physical_field_environments(
        n_environments=3, grid_size=6, timesteps=4, seed=123
    )
    records_by_env = {env.env_id: field_records(env) for env in envs}

    results = evaluate_leave_one_environment_prediction(records_by_env, MeanValuePredictor)
    mean, std = aggregate_prediction_results(results)

    assert set(results) == {"field_0", "field_1", "field_2"}
    assert "mean_abs_error" in mean
    assert "mean_abs_error" in std


def test_error_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        absolute_errors([], _toy_records())
    with pytest.raises(ValueError):
        summarize_absolute_errors([])
