# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import pytest

from motifshift.metrics.diagnostics import diagnostic_summary, top_k_coverage


def test_top_k_coverage() -> None:
    tokens = ["a", "a", "b", "c"]

    assert top_k_coverage(tokens, 1) == pytest.approx(0.5)
    assert top_k_coverage(tokens, 2) == pytest.approx(0.75)
    assert top_k_coverage(tokens, 10) == pytest.approx(1.0)


def test_diagnostic_summary_includes_requested_keys() -> None:
    summary = diagnostic_summary(["a", "a", "b", "c"], top_ks=(1, 2, 10))

    assert summary["n_observations"] == 4
    assert summary["vocab_size"] == 3
    assert "effective_vocab_size" in summary
    assert "normalized_effective_vocab_size" in summary
    assert summary["top_1_coverage"] == pytest.approx(0.5)
    assert summary["top_2_coverage"] == pytest.approx(0.75)
    assert summary["top_10_coverage"] == pytest.approx(1.0)
    assert isinstance(summary["n_observations"], int)
    assert isinstance(summary["effective_vocab_size"], float)


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_top_k_raises_value_error(k: int) -> None:
    with pytest.raises(ValueError):
        top_k_coverage(["a"], k)


def test_empty_tokens_raise_value_error() -> None:
    with pytest.raises(ValueError):
        top_k_coverage([], 1)
    with pytest.raises(ValueError):
        diagnostic_summary([])
