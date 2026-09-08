# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Transfer metrics that preserve independent sequence boundaries."""

from collections import defaultdict
from collections.abc import Hashable, Sequence

from motifshift.metrics.transfer import extract_ngrams, transfer_summary


def group_token_sequences(
    tokens: Sequence[Hashable], keys: Sequence[Hashable]
) -> list[list[Hashable]]:
    """Group aligned tokens by track/episode/cell, preserving within-group order."""
    if len(tokens) != len(keys):
        raise ValueError("tokens and sequence keys must have equal lengths")
    groups: dict[Hashable, list[Hashable]] = defaultdict(list)
    for token, key in zip(tokens, keys, strict=True):
        groups[key].append(token)
    return list(groups.values())


def sequence_transfer_summary(
    train: Sequence[Sequence[Hashable]], test: Sequence[Sequence[Hashable]], top_k: int = 50
) -> dict[str, float]:
    """Compute unigram metrics and within-sequence bigram diagnostics."""
    train_tokens = [token for sequence in train for token in sequence]
    test_tokens = [token for sequence in test for token in sequence]
    result = transfer_summary(train_tokens, test_tokens, top_k=top_k)
    train_bigrams = {pair for sequence in train for pair in extract_ngrams(sequence)}
    test_bigrams = [pair for sequence in test for pair in extract_ngrams(sequence)]
    result["bigram_coverage"] = (
        sum(pair in train_bigrams for pair in test_bigrams) / len(test_bigrams)
        if test_bigrams
        else 0.0
    )
    train_vocab = set(train_tokens)
    candidates = [pair for pair in test_bigrams if all(token in train_vocab for token in pair)]
    result["novel_recombination_rate"] = (
        sum(pair not in train_bigrams for pair in candidates) / len(candidates)
        if candidates
        else 0.0
    )
    return result
