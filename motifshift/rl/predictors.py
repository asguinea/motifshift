# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Simple token-level transfer predictors for RL diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence

import numpy as np


class MostFrequentNextTokenPredictor:
    """Predict the most frequent next token conditioned on the current token."""

    def __init__(self) -> None:
        self._next_counts: dict[Hashable, Counter[Hashable]] = defaultdict(Counter)
        self._global_next_counts: Counter[Hashable] = Counter()
        self._global_token_counts: Counter[Hashable] = Counter()
        self._is_fit = False

    def fit(self, tokens: Sequence[Hashable]) -> "MostFrequentNextTokenPredictor":
        """Fit next-token counts from a token sequence."""
        return self.fit_sequences([tokens])

    def fit_sequences(
        self, sequences: Sequence[Sequence[Hashable]]
    ) -> "MostFrequentNextTokenPredictor":
        """Fit independent episodes without inventing transitions between them."""
        tokens = [token for sequence in sequences for token in sequence]
        if not tokens:
            raise ValueError("tokens must be non-empty")
        self._next_counts = defaultdict(Counter)
        self._global_next_counts = Counter()
        self._global_token_counts = Counter(tokens)
        for sequence in sequences:
            for current, next_token in zip(sequence[:-1], sequence[1:]):
                self._next_counts[current][next_token] += 1
                self._global_next_counts[next_token] += 1
        self._is_fit = True
        return self

    def predict_next(self, token: Hashable) -> Hashable:
        """Predict the next token after a current token."""
        if not self._is_fit:
            raise ValueError("predictor must be fit before prediction")

        if token in self._next_counts and self._next_counts[token]:
            return self._next_counts[token].most_common(1)[0][0]
        if self._global_next_counts:
            return self._global_next_counts.most_common(1)[0][0]
        return self._global_token_counts.most_common(1)[0][0]

    def score_next_token_accuracy(self, tokens: Sequence[Hashable]) -> float:
        """Score next-token prediction accuracy on a held-out token sequence."""
        if len(tokens) < 2:
            return 0.0

        correct = 0
        total = 0
        for current, expected in zip(tokens[:-1], tokens[1:]):
            correct += int(self.predict_next(current) == expected)
            total += 1
        return float(correct / total)


def evaluate_leave_one_layout_prediction(
    tokens_by_layout: Mapping[str, Sequence[Hashable]],
    *,
    sequences_by_layout: Mapping[str, Sequence[Sequence[Hashable]]] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate next-token prediction by holding out each layout in turn."""
    if len(tokens_by_layout) < 2:
        raise ValueError("at least two layouts are required")

    results: dict[str, dict[str, float]] = {}
    for held_out_layout in sorted(tokens_by_layout):
        train_tokens = [
            token
            for layout_id in sorted(tokens_by_layout)
            if layout_id != held_out_layout
            for token in tokens_by_layout[layout_id]
        ]
        test_tokens = list(tokens_by_layout[held_out_layout])
        train_sequences = [
            sequence
            for layout_id in sorted(tokens_by_layout)
            if layout_id != held_out_layout
            for sequence in (
                sequences_by_layout[layout_id]
                if sequences_by_layout is not None
                else [tokens_by_layout[layout_id]]
            )
        ]
        test_sequences = (
            sequences_by_layout[held_out_layout]
            if sequences_by_layout is not None
            else [test_tokens]
        )
        predictor = MostFrequentNextTokenPredictor().fit_sequences(train_sequences)
        pairs = [pair for sequence in test_sequences for pair in zip(sequence[:-1], sequence[1:])]
        accuracy = (
            sum(predictor.predict_next(current) == expected for current, expected in pairs)
            / len(pairs)
            if pairs
            else 0.0
        )
        results[held_out_layout] = {
            "next_token_accuracy": accuracy,
            "test_length": float(len(test_tokens)),
            "train_length": float(len(train_tokens)),
        }

    return results


def aggregate_prediction_results(
    results: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate numeric prediction results by mean and standard deviation."""
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
