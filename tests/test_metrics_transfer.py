# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import math

import pytest

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


def test_token_coverage_metrics() -> None:
    train = ["a", "b", "a", "c"]
    test = ["a", "b", "d", "a"]

    assert token_coverage(train, test) == pytest.approx(0.75)
    assert unique_token_coverage(train, test) == pytest.approx(2 / 3)


def test_top_k_jaccard_and_jsd_are_finite() -> None:
    train = ["a", "b", "a", "c"]
    test = ["a", "b", "d", "a"]

    jaccard = top_k_jaccard(train, test, 2)
    divergence = jensen_shannon_divergence(train, test)

    assert 0 <= jaccard <= 1
    assert divergence >= 0
    assert math.isfinite(divergence)


def test_identical_distributions_have_zero_jsd() -> None:
    divergence = jensen_shannon_divergence(["a", "b", "a"], ["a", "b", "a"])

    assert divergence == pytest.approx(0.0)


def test_disjoint_deterministic_distributions_have_one_bit_jsd() -> None:
    divergence = jensen_shannon_divergence(["a", "a"], ["b", "b"], base=2)

    assert divergence == pytest.approx(1.0)


def test_extract_ngrams() -> None:
    assert extract_ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    assert extract_ngrams(["a", "b", "c"], 1) == [("a",), ("b",), ("c",)]
    assert extract_ngrams(["a"], 2) == []


def test_extract_ngrams_invalid_n_raises_value_error() -> None:
    with pytest.raises(ValueError):
        extract_ngrams(["a", "b"], 0)


def test_bigram_coverage_and_novel_recombination_rate() -> None:
    train = ["a", "b", "c"]
    test = ["a", "c", "b"]

    assert bigram_coverage(train, test) == pytest.approx(0.0)
    assert novel_recombination_rate(train, test) == pytest.approx(1.0)


def test_bigram_edge_cases() -> None:
    assert bigram_coverage(["a"], ["a", "b"]) == pytest.approx(0.0)
    assert bigram_coverage(["a", "b"], ["a"]) == pytest.approx(0.0)
    assert novel_recombination_rate(["a"], ["a", "b"]) == pytest.approx(0.0)


def test_transfer_summary_includes_expected_keys() -> None:
    summary = transfer_summary(["a", "b", "a", "c"], ["a", "b", "d", "a"], top_k=2)

    assert set(summary) == {
        "token_coverage",
        "unique_token_coverage",
        "top_2_jaccard",
        "js_divergence",
        "bigram_coverage",
        "novel_recombination_rate",
    }


@pytest.mark.parametrize(
    "function,args",
    [
        (token_coverage, ([], ["a"])),
        (token_coverage, (["a"], [])),
        (unique_token_coverage, ([], ["a"])),
        (top_k_jaccard, ([], ["a"], 1)),
        (jensen_shannon_divergence, ([], ["a"])),
        (bigram_coverage, ([], ["a"])),
        (novel_recombination_rate, (["a"], [])),
    ],
)
def test_empty_train_or_test_sequences_raise_value_error(function, args) -> None:
    with pytest.raises(ValueError):
        function(*args)


def test_invalid_top_k_jaccard_raises_value_error() -> None:
    with pytest.raises(ValueError):
        top_k_jaccard(["a"], ["a"], 0)
