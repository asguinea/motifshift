# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Lightweight spatio-temporal transfer predictors."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np

Step = Mapping[str, Any]


def _target(step: Step) -> tuple[float, float]:
    return (
        float(step.get("target_dx", step["dx"])),
        float(step.get("target_dy", step["dy"])),
    )


def _observed_displacement(step: Step) -> tuple[float, float]:
    return (
        float(step.get("prev_dx", step["dx"])),
        float(step.get("prev_dy", step["dy"])),
    )


def _feature(step: Step) -> list[float]:
    return [
        float(step["x"]),
        float(step["y"]),
        float(step.get("prev_dx", step["dx"])),
        float(step.get("prev_dy", step["dy"])),
    ]


def _validate_bins(upper_bounds: Sequence[float]) -> None:
    if any(bound < 0 for bound in upper_bounds):
        raise ValueError("bin upper bounds must be non-negative")
    if list(upper_bounds) != sorted(upper_bounds):
        raise ValueError("bin upper bounds must be sorted")


def _bin_value(value: float, upper_bounds: Sequence[float]) -> int:
    _validate_bins(upper_bounds)
    return int(np.searchsorted(np.asarray(upper_bounds, dtype=float), value, side="left"))


def _motion_token(
    step: Step,
    speed_bins: Sequence[float] = (0.4, 0.8, 1.2, 1.6, 2.2, 3.0),
    direction_bins: int = 16,
) -> tuple[str, int, int]:
    if direction_bins <= 0:
        raise ValueError("direction_bins must be positive")

    dx, dy = _observed_displacement(step)
    speed = math.hypot(dx, dy)
    speed_bin = _bin_value(speed, speed_bins)
    if speed == 0:
        direction_bin = -1
    else:
        angle = math.atan2(dy, dx) % (2 * math.pi)
        direction_bin = int(math.floor(angle / (2 * math.pi) * direction_bins))
        direction_bin = min(direction_bin, direction_bins - 1)
    return ("motion", speed_bin, direction_bin)


def _motion_tokens(steps: Sequence[Step]) -> list[Hashable]:
    return [_motion_token(step) for step in steps]


def _situation_motion_tokens(
    steps: Sequence[Step],
    distance_bins: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    density_bins: Sequence[float] = (0.0, 1.0, 2.0, 4.0),
    density_radius: float = 3.0,
) -> list[Hashable]:
    if density_radius < 0:
        raise ValueError("density_radius must be non-negative")
    _validate_bins(distance_bins)
    _validate_bins(density_bins)

    steps_by_scene_time: dict[tuple[str, int], list[Step]] = defaultdict(list)
    for step in steps:
        steps_by_scene_time[(str(step["scene_id"]), int(step["t"]))].append(step)

    tokens: list[Hashable] = []
    for step in steps:
        motion = _motion_token(step)
        neighbors = [
            other
            for other in steps_by_scene_time[(str(step["scene_id"]), int(step["t"]))]
            if int(other["agent_id"]) != int(step["agent_id"])
        ]
        distances = [
            math.hypot(float(other["x"]) - float(step["x"]), float(other["y"]) - float(step["y"]))
            for other in neighbors
        ]
        nearest_distance = min(distances) if distances else math.inf
        local_density = sum(1 for distance in distances if distance <= density_radius)
        tokens.append(
            (
                "situation_motion",
                motion[1],
                motion[2],
                _bin_value(nearest_distance, distance_bins),
                _bin_value(float(local_density), density_bins),
            )
        )
    return tokens


class ConstantVelocityPredictor:
    """Predict that the next displacement equals the observed displacement."""

    def fit(self, train_steps: Sequence[Step]) -> "ConstantVelocityPredictor":
        return self

    def predict(self, step: Step) -> tuple[float, float]:
        return _observed_displacement(step)

    def predict_batch(self, steps: Sequence[Step]) -> list[tuple[float, float]]:
        return [self.predict(step) for step in steps]


class MeanDisplacementPredictor:
    """Predict the mean training displacement."""

    def __init__(self) -> None:
        self._mean: tuple[float, float] | None = None

    def fit(self, train_steps: Sequence[Step]) -> "MeanDisplacementPredictor":
        if not train_steps:
            raise ValueError("train_steps must be non-empty")
        targets = np.asarray([_target(step) for step in train_steps], dtype=float)
        mean = targets.mean(axis=0)
        self._mean = (float(mean[0]), float(mean[1]))
        return self

    def predict(self, step: Step) -> tuple[float, float]:
        if self._mean is None:
            raise ValueError("predictor must be fit before prediction")
        return self._mean

    def predict_batch(self, steps: Sequence[Step]) -> list[tuple[float, float]]:
        return [self.predict(step) for step in steps]


class KNNDisplacementPredictor:
    """Predict displacement by averaging nearest training transition features."""

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self._features: np.ndarray | None = None
        self._targets: np.ndarray | None = None

    def fit(self, train_steps: Sequence[Step]) -> "KNNDisplacementPredictor":
        if not train_steps:
            raise ValueError("train_steps must be non-empty")
        self._features = np.asarray([_feature(step) for step in train_steps], dtype=float)
        self._targets = np.asarray([_target(step) for step in train_steps], dtype=float)
        return self

    def predict(self, step: Step) -> tuple[float, float]:
        if self._features is None or self._targets is None:
            raise ValueError("predictor must be fit before prediction")
        query = np.asarray(_feature(step), dtype=float)
        distances = np.linalg.norm(self._features - query, axis=1)
        k = min(self.k, len(distances))
        neighbor_indices = np.argpartition(distances, k - 1)[:k]
        prediction = self._targets[neighbor_indices].mean(axis=0)
        return float(prediction[0]), float(prediction[1])

    def predict_batch(self, steps: Sequence[Step]) -> list[tuple[float, float]]:
        return [self.predict(step) for step in steps]


class TokenMeanDisplacementPredictor:
    """Predict mean displacement conditioned on a representation token."""

    def __init__(self, token_fn: Callable[[Sequence[Step]], list[Hashable]]) -> None:
        self.token_fn = token_fn
        self._token_means: dict[Hashable, tuple[float, float]] = {}
        self._global_mean: tuple[float, float] | None = None

    def fit(self, train_steps: Sequence[Step]) -> "TokenMeanDisplacementPredictor":
        if not train_steps:
            raise ValueError("train_steps must be non-empty")

        tokens = self.token_fn(train_steps)
        targets = np.asarray([_target(step) for step in train_steps], dtype=float)
        global_mean = targets.mean(axis=0)
        self._global_mean = (float(global_mean[0]), float(global_mean[1]))

        sums: dict[Hashable, np.ndarray] = {}
        counts: dict[Hashable, int] = defaultdict(int)
        for token, target in zip(tokens, targets):
            if token not in sums:
                sums[token] = np.zeros(2, dtype=float)
            sums[token] += target
            counts[token] += 1

        self._token_means = {
            token: (float(total[0] / counts[token]), float(total[1] / counts[token]))
            for token, total in sums.items()
        }
        return self

    def predict_batch(self, steps: Sequence[Step]) -> list[tuple[float, float]]:
        if self._global_mean is None:
            raise ValueError("predictor must be fit before prediction")
        return [self._token_means.get(token, self._global_mean) for token in self.token_fn(steps)]


class MotionTokenMarkovPredictor(TokenMeanDisplacementPredictor):
    """Token-conditioned predictor using motion tokens."""

    def __init__(self) -> None:
        super().__init__(_motion_tokens)


class SituationMotionTokenPredictor(TokenMeanDisplacementPredictor):
    """Token-conditioned predictor using situation-motion tokens."""

    def __init__(self) -> None:
        super().__init__(_situation_motion_tokens)


def displacement_errors(
    predictions: Sequence[tuple[float, float]], steps: Sequence[Step]
) -> list[float]:
    """Compute Euclidean displacement errors."""
    if len(predictions) != len(steps):
        raise ValueError("predictions and steps must have the same length")
    errors: list[float] = []
    for prediction, step in zip(predictions, steps):
        true_dx, true_dy = _target(step)
        errors.append(math.hypot(prediction[0] - true_dx, prediction[1] - true_dy))
    return errors


def summarize_errors(errors: Sequence[float]) -> dict[str, float]:
    """Summarize displacement errors."""
    if not errors:
        raise ValueError("errors must be non-empty")
    values = np.asarray(errors, dtype=float)
    return {
        "mean_error": float(np.mean(values)),
        "median_error": float(np.median(values)),
        "n_steps": float(len(values)),
    }


def evaluate_leave_one_scene_prediction(
    steps_by_scene: Mapping[str, Sequence[Step]],
    predictor_factory: Callable[[], Any],
) -> dict[str, dict[str, float]]:
    """Evaluate a predictor with leave-one-scene-out transfer."""
    if len(steps_by_scene) < 2:
        raise ValueError("at least two scenes are required")

    results: dict[str, dict[str, float]] = {}
    for held_out_scene in sorted(steps_by_scene):
        train_steps = [
            step
            for scene_id in sorted(steps_by_scene)
            if scene_id != held_out_scene
            for step in steps_by_scene[scene_id]
        ]
        test_steps = list(steps_by_scene[held_out_scene])
        predictor = predictor_factory().fit(train_steps)
        predictions = predictor.predict_batch(test_steps)
        results[held_out_scene] = summarize_errors(displacement_errors(predictions, test_steps))
    return results


def aggregate_error_results(
    results: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate error summaries by mean and standard deviation."""
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
