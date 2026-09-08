# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Reusable metrics for representation diagnostics."""

from motifshift.metrics.diagnostics import diagnostic_summary, top_k_coverage
from motifshift.metrics.distributions import (
    effective_vocabulary_size,
    entropy,
    normalized_effective_vocabulary_size,
    token_counts,
    token_distribution,
)
from motifshift.metrics.transfer import (
    bigram_coverage,
    extract_ngrams,
    jensen_shannon_divergence,
    novel_recombination_rate,
    token_coverage,
    top_k_jaccard,
    transfer_summary,
    unique_token_coverage,
)

__all__ = [
    "token_counts",
    "token_distribution",
    "entropy",
    "effective_vocabulary_size",
    "normalized_effective_vocabulary_size",
    "top_k_coverage",
    "diagnostic_summary",
    "token_coverage",
    "unique_token_coverage",
    "top_k_jaccard",
    "jensen_shannon_divergence",
    "extract_ngrams",
    "bigram_coverage",
    "novel_recombination_rate",
    "transfer_summary",
]
