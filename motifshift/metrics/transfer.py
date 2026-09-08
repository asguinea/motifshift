# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Transfer and recombination metrics over token sequences."""

from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np


def _require_non_empty(tokens: Sequence[Hashable], name: str) -> None:
    if not tokens:
        raise ValueError(f"{name} must be non-empty")


def _top_k_set(tokens: Sequence[Hashable], k: int) -> set[Hashable]:
    return {token for token, _ in Counter(tokens).most_common(k)}


def token_coverage(train_tokens: Sequence[Hashable], test_tokens: Sequence[Hashable]) -> float:
    """Return the fraction of test observations seen at least once in train."""
    _require_non_empty(train_tokens, "train_tokens")
    _require_non_empty(test_tokens, "test_tokens")

    train_vocab = set(train_tokens)
    covered = sum(1 for token in test_tokens if token in train_vocab)
    return float(covered / len(test_tokens))


def unique_token_coverage(
    train_tokens: Sequence[Hashable], test_tokens: Sequence[Hashable]
) -> float:
    """Return the fraction of unique test token types seen in train."""
    _require_non_empty(train_tokens, "train_tokens")
    _require_non_empty(test_tokens, "test_tokens")

    train_vocab = set(train_tokens)
    test_vocab = set(test_tokens)
    return float(len(test_vocab & train_vocab) / len(test_vocab))


def top_k_jaccard(
    train_tokens: Sequence[Hashable], test_tokens: Sequence[Hashable], k: int
) -> float:
    """Compute Jaccard overlap between train and test top-k token sets."""
    if k <= 0:
        raise ValueError("k must be positive")
    _require_non_empty(train_tokens, "train_tokens")
    _require_non_empty(test_tokens, "test_tokens")

    train_top = _top_k_set(train_tokens, k)
    test_top = _top_k_set(test_tokens, k)
    if not train_top and not test_top:
        return 1.0
    if not train_top or not test_top:
        raise ValueError("top-k token sets must be non-empty")

    return float(len(train_top & test_top) / len(train_top | test_top))


def jensen_shannon_divergence(
    tokens_a: Sequence[Hashable], tokens_b: Sequence[Hashable], base: float | None = 2
) -> float:
    """Compute Jensen-Shannon divergence between empirical token distributions."""
    _require_non_empty(tokens_a, "tokens_a")
    _require_non_empty(tokens_b, "tokens_b")
    if base is not None and (base <= 0 or base == 1):
        raise ValueError("base must be positive and not equal to 1")

    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    vocab = list(dict.fromkeys([*counts_a, *counts_b]))

    probs_a = np.asarray([counts_a[token] / len(tokens_a) for token in vocab], dtype=float)
    probs_b = np.asarray([counts_b[token] / len(tokens_b) for token in vocab], dtype=float)
    mixture = 0.5 * (probs_a + probs_b)

    def kl_divergence(probs: np.ndarray, reference: np.ndarray) -> float:
        mask = probs > 0
        logs = np.log(probs[mask] / reference[mask])
        if base is not None:
            logs = logs / np.log(base)
        return float(np.sum(probs[mask] * logs))

    divergence = 0.5 * kl_divergence(probs_a, mixture) + 0.5 * kl_divergence(probs_b, mixture)
    return float(max(0.0, divergence))


def extract_ngrams(tokens: Sequence[Hashable], n: int = 2) -> list[tuple[Hashable, ...]]:
    """Return contiguous n-grams from a token sequence."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if len(tokens) < n:
        return []

    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def bigram_coverage(train_tokens: Sequence[Hashable], test_tokens: Sequence[Hashable]) -> float:
    """Return the fraction of test bigram observations seen in train."""
    _require_non_empty(train_tokens, "train_tokens")
    _require_non_empty(test_tokens, "test_tokens")

    train_bigrams = set(extract_ngrams(train_tokens, 2))
    test_bigrams = extract_ngrams(test_tokens, 2)
    if not test_bigrams or not train_bigrams:
        return 0.0

    covered = sum(1 for bigram in test_bigrams if bigram in train_bigrams)
    return float(covered / len(test_bigrams))


def novel_recombination_rate(
    train_tokens: Sequence[Hashable], test_tokens: Sequence[Hashable]
) -> float:
    """Measure novel test bigrams composed of train-seen unigrams."""
    _require_non_empty(train_tokens, "train_tokens")
    _require_non_empty(test_tokens, "test_tokens")

    train_unigrams = set(train_tokens)
    train_bigrams = set(extract_ngrams(train_tokens, 2))
    candidate_bigrams = [
        bigram
        for bigram in extract_ngrams(test_tokens, 2)
        if bigram[0] in train_unigrams and bigram[1] in train_unigrams
    ]

    if not candidate_bigrams:
        return 0.0

    novel = sum(1 for bigram in candidate_bigrams if bigram not in train_bigrams)
    return float(novel / len(candidate_bigrams))


def transfer_summary(
    train_tokens: Sequence[Hashable], test_tokens: Sequence[Hashable], top_k: int = 50
) -> dict[str, float]:
    """Return a compact transfer diagnostic summary."""
    return {
        "token_coverage": token_coverage(train_tokens, test_tokens),
        "unique_token_coverage": unique_token_coverage(train_tokens, test_tokens),
        f"top_{top_k}_jaccard": top_k_jaccard(train_tokens, test_tokens, top_k),
        "js_divergence": jensen_shannon_divergence(train_tokens, test_tokens),
        "bigram_coverage": bigram_coverage(train_tokens, test_tokens),
        "novel_recombination_rate": novel_recombination_rate(train_tokens, test_tokens),
    }
