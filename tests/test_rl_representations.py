# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import pytest

from motifshift.metrics import token_coverage
from motifshift.representations.rl import (
    local_observation_action_tokens,
    raw_state_action_tokens,
    situation_action_tokens,
    situation_transition_tokens,
)
from motifshift.rl.gridworld import generate_episodes, generate_gridworlds


def _sample_steps():
    worlds = generate_gridworlds(n_layouts=2, width=8, height=8, seed=123)
    steps = generate_episodes(worlds, episodes_per_world=3, max_steps=25, seed=321, noise_prob=0.1)
    worlds_by_id = {world.layout_id: world for world in worlds}
    return worlds, steps, worlds_by_id


def test_tokenizers_return_one_token_per_step() -> None:
    _, steps, worlds_by_id = _sample_steps()

    assert len(raw_state_action_tokens(steps, worlds_by_id)) == len(steps)
    assert len(local_observation_action_tokens(steps, worlds_by_id)) == len(steps)
    assert len(situation_action_tokens(steps, worlds_by_id)) == len(steps)
    assert len(situation_transition_tokens(steps, worlds_by_id)) == len(steps)


def test_raw_tokens_include_layout_id() -> None:
    _, steps, worlds_by_id = _sample_steps()

    token = raw_state_action_tokens(steps, worlds_by_id)[0]

    assert token[0] == steps[0].layout_id
    assert len(token) == 4


def test_local_observation_tokens_do_not_include_layout_id() -> None:
    _, steps, worlds_by_id = _sample_steps()

    token = local_observation_action_tokens(steps, worlds_by_id)[0]

    assert len(token) == 2
    assert token[1] == steps[0].action
    assert steps[0].layout_id not in token
    assert len(token[0]) == 9


def test_situation_tokens_have_expected_tuple_lengths() -> None:
    _, steps, worlds_by_id = _sample_steps()

    situation_action = situation_action_tokens(steps, worlds_by_id)[0]
    situation_transition = situation_transition_tokens(steps, worlds_by_id)[0]

    assert len(situation_action) == 2
    assert len(situation_transition) == 3


def test_tokenization_is_deterministic() -> None:
    _, steps, worlds_by_id = _sample_steps()

    assert raw_state_action_tokens(steps, worlds_by_id) == raw_state_action_tokens(
        steps, worlds_by_id
    )
    assert situation_transition_tokens(steps, worlds_by_id) == (
        situation_transition_tokens(steps, worlds_by_id)
    )


def test_cross_layout_abstractions_recur_more_than_raw_tokens() -> None:
    worlds = generate_gridworlds(n_layouts=2, width=8, height=8, seed=123)
    steps = generate_episodes(
        worlds, episodes_per_world=10, max_steps=30, seed=123, noise_prob=0.15
    )
    worlds_by_id = {world.layout_id: world for world in worlds}
    train_steps = [step for step in steps if step.layout_id == "layout_0"]
    test_steps = [step for step in steps if step.layout_id == "layout_1"]

    raw_coverage = token_coverage(
        raw_state_action_tokens(train_steps, worlds_by_id),
        raw_state_action_tokens(test_steps, worlds_by_id),
    )
    local_coverage = token_coverage(
        local_observation_action_tokens(train_steps, worlds_by_id),
        local_observation_action_tokens(test_steps, worlds_by_id),
    )
    situation_coverage = token_coverage(
        situation_action_tokens(train_steps, worlds_by_id),
        situation_action_tokens(test_steps, worlds_by_id),
    )

    assert local_coverage >= raw_coverage
    assert situation_coverage >= raw_coverage


def test_empty_steps_return_empty_lists() -> None:
    worlds = generate_gridworlds(n_layouts=1, width=8, height=8, seed=123)
    worlds_by_id = {world.layout_id: world for world in worlds}

    assert raw_state_action_tokens([], worlds_by_id) == []
    assert local_observation_action_tokens([], worlds_by_id) == []
    assert situation_action_tokens([], worlds_by_id) == []
    assert situation_transition_tokens([], worlds_by_id) == []


def test_missing_world_id_raises_key_error() -> None:
    _, steps, _ = _sample_steps()

    with pytest.raises(KeyError, match="Missing world"):
        local_observation_action_tokens(steps[:1], {})
    with pytest.raises(KeyError, match="Missing world"):
        situation_action_tokens(steps[:1], {})
    with pytest.raises(KeyError, match="Missing world"):
        situation_transition_tokens(steps[:1], {})
