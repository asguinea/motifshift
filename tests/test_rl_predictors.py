# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import pytest

from motifshift.rl.predictors import (
    MostFrequentNextTokenPredictor,
    aggregate_prediction_results,
    evaluate_leave_one_layout_prediction,
)


def test_most_frequent_next_token_predictor_fit_and_predict() -> None:
    predictor = MostFrequentNextTokenPredictor().fit(["a", "b", "a", "b", "c"])

    assert predictor.predict_next("a") == "b"
    assert predictor.predict_next("b") == "a"


def test_unseen_current_token_uses_global_next_token_fallback() -> None:
    predictor = MostFrequentNextTokenPredictor().fit(["a", "b", "a", "b", "c"])

    assert predictor.predict_next("unseen") == "b"


def test_predictor_uses_global_token_fallback_when_no_bigrams() -> None:
    predictor = MostFrequentNextTokenPredictor().fit(["only"])

    assert predictor.predict_next("unseen") == "only"


def test_score_next_token_accuracy() -> None:
    predictor = MostFrequentNextTokenPredictor().fit(["a", "b", "a", "b", "c"])

    assert predictor.score_next_token_accuracy(["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert predictor.score_next_token_accuracy(["short"]) == pytest.approx(0.0)


def test_empty_fit_raises_value_error() -> None:
    with pytest.raises(ValueError):
        MostFrequentNextTokenPredictor().fit([])


def test_predict_before_fit_raises_value_error() -> None:
    with pytest.raises(ValueError):
        MostFrequentNextTokenPredictor().predict_next("a")


def test_evaluate_leave_one_layout_prediction() -> None:
    tokens_by_layout = {
        "layout_0": ["a", "b", "a"],
        "layout_1": ["a", "b", "c"],
        "layout_2": ["b", "c", "b"],
    }

    results = evaluate_leave_one_layout_prediction(tokens_by_layout)

    assert set(results) == {"layout_0", "layout_1", "layout_2"}
    for row in results.values():
        assert "next_token_accuracy" in row
        assert "train_length" in row
        assert "test_length" in row
        assert 0 <= row["next_token_accuracy"] <= 1
        assert row["train_length"] == 6
        assert row["test_length"] == 3


def test_evaluate_leave_one_layout_prediction_requires_two_layouts() -> None:
    with pytest.raises(ValueError):
        evaluate_leave_one_layout_prediction({"layout_0": ["a", "b"]})


def test_aggregate_prediction_results() -> None:
    mean, std = aggregate_prediction_results(
        {
            "layout_0": {"next_token_accuracy": 0.0, "train_length": 4.0},
            "layout_1": {"next_token_accuracy": 1.0, "train_length": 6.0},
        }
    )

    assert mean["next_token_accuracy"] == pytest.approx(0.5)
    assert std["next_token_accuracy"] == pytest.approx(0.5)
    assert mean["train_length"] == pytest.approx(5.0)


def test_aggregate_prediction_results_handles_empty_input() -> None:
    assert aggregate_prediction_results({}) == ({}, {})
