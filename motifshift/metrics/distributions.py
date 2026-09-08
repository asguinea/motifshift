# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Token distribution metrics."""

from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np


def token_counts(tokens: Sequence[Hashable]) -> Counter[Hashable]:
    """Count token observations."""
    if not tokens:
        raise ValueError("tokens must be non-empty")
    return Counter(tokens)


def token_distribution(tokens: Sequence[Hashable]) -> dict[Hashable, float]:
    """Compute empirical token probabilities."""
    counts = token_counts(tokens)
    total = sum(counts.values())
    return {token: count / total for token, count in counts.items()}


def entropy(probs: Sequence[float], base: float | None = None) -> float:
    """Compute Shannon entropy for a probability vector."""
    if not probs:
        raise ValueError("probabilities must be non-empty")
    if base is not None and (base <= 0 or base == 1):
        raise ValueError("base must be positive and not equal to 1")

    values = np.asarray(probs, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("probabilities must be finite")
    if np.any(values < 0):
        raise ValueError("probabilities must be non-negative")

    total = float(values.sum())
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")

    normalized = values / total
    positive = normalized[normalized > 0]
    log_values = np.log(positive)
    if base is not None:
        log_values = log_values / np.log(base)

    return float(-np.sum(positive * log_values))


def effective_vocabulary_size(tokens: Sequence[Hashable]) -> float:
    """Compute exp(H(p)) from empirical token probabilities."""
    distribution = token_distribution(tokens)
    return float(np.exp(entropy(list(distribution.values()))))


def normalized_effective_vocabulary_size(tokens: Sequence[Hashable]) -> float:
    """Compute effective vocabulary size normalized by observations."""
    return float(effective_vocabulary_size(tokens) / len(tokens))
