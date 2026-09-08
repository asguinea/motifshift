# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Lightweight transfer predictors for synthetic physical fields."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from motifshift.physical.representations import (
    dynamical_regime_tokens,
    local_patch_tokens,
    raw_cell_tokens,
    value_change_tokens,
)

Record = Mapping[str, Any]


def prediction_features(records: Sequence[Record]) -> list[dict[str, Any]]:
    """Expose current observations and lagged change, excluding future labels.

    The first frame has no previous change and uses zero. Diagnostic tokens may
    summarize complete transitions; predictive tokens must only use the past.
    """
    return [
        {
            **{key: value for key, value in record.items() if key not in {"next_value", "delta"}},
            "delta": float(record.get("prev_delta", 0.0)),
        }
        for record in records
    ]


class MeanValuePredictor:
    """Predict the mean next value from training records."""

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, records: Sequence[Record]) -> "MeanValuePredictor":
        if not records:
            raise ValueError("records must be non-empty")
        self._mean = float(np.mean([float(record["next_value"]) for record in records]))
        return self

    def predict_batch(self, records: Sequence[Record]) -> list[float]:
        if self._mean is None:
            raise ValueError("predictor must be fit before prediction")
        return [self._mean for _ in records]


class TokenMeanValuePredictor:
    """Predict mean next value conditioned on a token."""

    def __init__(self, token_fn: Callable[[Sequence[Record]], list[Hashable]]) -> None:
        self.token_fn = token_fn
        self._global_mean: float | None = None
        self._means: dict[Hashable, float] = {}

    def fit(self, records: Sequence[Record]) -> "TokenMeanValuePredictor":
        if not records:
            raise ValueError("records must be non-empty")
        tokens = self.token_fn(prediction_features(records))
        values = np.asarray([float(record["next_value"]) for record in records], dtype=float)
        self._global_mean = float(np.mean(values))
        sums: dict[Hashable, float] = defaultdict(float)
        counts: dict[Hashable, int] = defaultdict(int)
        for token, value in zip(tokens, values):
            sums[token] += float(value)
            counts[token] += 1
        self._means = {token: sums[token] / counts[token] for token in sums}
        return self

    def predict_batch(self, records: Sequence[Record]) -> list[float]:
        if self._global_mean is None:
            raise ValueError("predictor must be fit before prediction")
        return [
            self._means.get(token, self._global_mean)
            for token in self.token_fn(prediction_features(records))
        ]


class RawCellLookupPredictor(TokenMeanValuePredictor):
    def __init__(self) -> None:
        super().__init__(raw_cell_tokens)


class ValueChangeTokenPredictor(TokenMeanValuePredictor):
    def __init__(self) -> None:
        super().__init__(value_change_tokens)


class LocalPatchTokenPredictor(TokenMeanValuePredictor):
    def __init__(self) -> None:
        super().__init__(local_patch_tokens)


class RegimeTokenPredictor(TokenMeanValuePredictor):
    def __init__(self) -> None:
        super().__init__(dynamical_regime_tokens)


def absolute_errors(predictions: Sequence[float], records: Sequence[Record]) -> list[float]:
    """Compute absolute next-value prediction errors."""
    if len(predictions) != len(records):
        raise ValueError("predictions and records must have the same length")
    return [
        abs(float(prediction) - float(record["next_value"]))
        for prediction, record in zip(predictions, records)
    ]


def summarize_absolute_errors(errors: Sequence[float]) -> dict[str, float]:
    """Summarize absolute errors."""
    if not errors:
        raise ValueError("errors must be non-empty")
    values = np.asarray(errors, dtype=float)
    return {
        "mean_abs_error": float(np.mean(values)),
        "median_abs_error": float(np.median(values)),
        "n_records": float(len(values)),
    }


def evaluate_leave_one_environment_prediction(
    records_by_env: Mapping[str, Sequence[Record]],
    predictor_factory: Callable[[], Any],
) -> dict[str, dict[str, float]]:
    """Evaluate next-value prediction by holding out one environment."""
    if len(records_by_env) < 2:
        raise ValueError("at least two environments are required")
    results: dict[str, dict[str, float]] = {}
    for held_out in sorted(records_by_env):
        train_records = [
            record
            for env_id in sorted(records_by_env)
            if env_id != held_out
            for record in records_by_env[env_id]
        ]
        test_records = list(records_by_env[held_out])
        predictor = predictor_factory().fit(train_records)
        predictions = predictor.predict_batch(test_records)
        results[held_out] = summarize_absolute_errors(absolute_errors(predictions, test_records))
    return results


def aggregate_prediction_results(
    results: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate prediction summaries by mean and standard deviation."""
    if not results:
        return {}, {}
    keys = list(next(iter(results.values())).keys())
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in results.values()], dtype=float)
        means[key] = float(np.mean(values))
        stds[key] = float(np.std(values))
    return means, stds
