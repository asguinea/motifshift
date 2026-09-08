# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import pytest

from motifshift.metrics import token_coverage
from motifshift.physical.field import (
    field_records,
    generate_physical_field_environments,
)
from motifshift.physical.representations import (
    dynamical_regime_tokens,
    local_patch_tokens,
    raw_cell_tokens,
    value_change_tokens,
)


def _records():
    env = generate_physical_field_environments(
        n_environments=1, grid_size=8, timesteps=4, seed=123
    )[0]
    return field_records(env)


def test_physical_tokenizers_return_one_tuple_per_record() -> None:
    records = _records()

    for tokenizer in [
        raw_cell_tokens,
        value_change_tokens,
        local_patch_tokens,
        dynamical_regime_tokens,
    ]:
        tokens = tokenizer(records)
        assert len(tokens) == len(records)
        assert isinstance(tokens[0], tuple)
        assert tokens == tokenizer(records)


def test_physical_tokenizers_handle_empty_records() -> None:
    assert raw_cell_tokens([]) == []
    assert value_change_tokens([]) == []
    assert local_patch_tokens([]) == []
    assert dynamical_regime_tokens([]) == []


@pytest.mark.parametrize(
    "call",
    [
        lambda: raw_cell_tokens(_records(), value_bins=0),
        lambda: raw_cell_tokens(_records(), grid_bin_size=0),
        lambda: value_change_tokens(_records(), delta_bins=0),
        lambda: value_change_tokens(_records(), delta_range=0),
        lambda: local_patch_tokens(_records(), value_bins=0),
        lambda: dynamical_regime_tokens(_records(), grad_direction_bins=0),
        lambda: dynamical_regime_tokens(_records(), grad_magnitude_bins=(0.2, 0.1)),
    ],
)
def test_physical_tokenizers_invalid_params_raise_value_error(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_abstract_tokens_reuse_across_environments() -> None:
    envs = generate_physical_field_environments(
        n_environments=2, grid_size=10, timesteps=5, seed=123
    )
    train = field_records(envs[0])
    test = field_records(envs[1])

    raw_coverage = token_coverage(raw_cell_tokens(train), raw_cell_tokens(test))
    value_coverage = token_coverage(value_change_tokens(train), value_change_tokens(test))
    regime_coverage = token_coverage(dynamical_regime_tokens(train), dynamical_regime_tokens(test))

    assert value_coverage >= raw_coverage or regime_coverage >= raw_coverage
