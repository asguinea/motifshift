# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Domain-independent diagnostic summaries over token observations."""

from collections.abc import Hashable, Sequence
from typing import Any

from motifshift.metrics.distributions import (
    effective_vocabulary_size,
    normalized_effective_vocabulary_size,
    token_counts,
)


def top_k_coverage(tokens: Sequence[Hashable], k: int) -> float:
    """Return the fraction of observations covered by the top-k tokens."""
    if k <= 0:
        raise ValueError("k must be positive")

    counts = token_counts(tokens)
    if k >= len(counts):
        return 1.0

    covered = sum(count for _, count in counts.most_common(k))
    return float(covered / len(tokens))


def diagnostic_summary(
    tokens: Sequence[Hashable], top_ks: Sequence[int] = (10, 50, 100)
) -> dict[str, float | int]:
    """Return a compact diagnostic summary for token observations."""
    counts = token_counts(tokens)
    summary: dict[str, Any] = {
        "n_observations": int(len(tokens)),
        "vocab_size": int(len(counts)),
        "effective_vocab_size": float(effective_vocabulary_size(tokens)),
        "normalized_effective_vocab_size": float(normalized_effective_vocabulary_size(tokens)),
    }

    for k in top_ks:
        summary[f"top_{k}_coverage"] = float(top_k_coverage(tokens, k))

    return summary
