# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import math

import pytest

from motifshift.metrics.distributions import (
    effective_vocabulary_size,
    entropy,
    normalized_effective_vocabulary_size,
    token_counts,
    token_distribution,
)


def test_token_counts_and_distribution() -> None:
    tokens = ["a", "a", "a", "b"]

    counts = token_counts(tokens)
    distribution = token_distribution(tokens)

    assert counts["a"] == 3
    assert counts["b"] == 1
    assert distribution["a"] == pytest.approx(0.75)
    assert distribution["b"] == pytest.approx(0.25)


def test_entropy_and_effective_vocabulary_are_in_expected_ranges() -> None:
    tokens = ["a", "a", "a", "b"]
    distribution = token_distribution(tokens)

    assert entropy(list(distribution.values())) > 0

    effective_vocab = effective_vocabulary_size(tokens)
    normalized_effective_vocab = normalized_effective_vocabulary_size(tokens)

    assert 1 < effective_vocab < 2
    assert 0 < normalized_effective_vocab < 1


def test_deterministic_sequence_has_effective_vocabulary_one() -> None:
    assert effective_vocabulary_size(["x", "x", "x"]) == pytest.approx(1.0)


def test_entropy_supports_bits_and_internal_normalization() -> None:
    assert entropy([1, 1], base=2) == pytest.approx(1.0)
    assert entropy([0.5, 0.5], base=2) == pytest.approx(1.0)
    assert math.isfinite(entropy([0.0, 2.0]))


@pytest.mark.parametrize(
    "probs",
    [
        [],
        [-0.1, 1.1],
        [0.0, 0.0],
    ],
)
def test_entropy_invalid_probabilities_raise_value_error(probs: list[float]) -> None:
    with pytest.raises(ValueError):
        entropy(probs)


def test_empty_tokens_raise_value_error() -> None:
    with pytest.raises(ValueError):
        token_counts([])
    with pytest.raises(ValueError):
        token_distribution([])
    with pytest.raises(ValueError):
        effective_vocabulary_size([])
    with pytest.raises(ValueError):
        normalized_effective_vocabulary_size([])
