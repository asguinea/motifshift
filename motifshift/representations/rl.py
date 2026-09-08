# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""RL gridworld token representations."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

from motifshift.rl.gridworld import (
    GridWorld,
    RLStep,
    classify_situation,
    local_wall_pattern,
)


def _world_for_step(step: RLStep, worlds_by_id: Mapping[str, GridWorld]) -> GridWorld:
    try:
        return worlds_by_id[step.layout_id]
    except KeyError as exc:
        raise KeyError(f"Missing world for layout_id: {step.layout_id}") from exc


def raw_state_action_tokens(
    steps: Sequence[RLStep], worlds_by_id: Mapping[str, GridWorld]
) -> list[Hashable]:
    """Tokenize layout-specific state-action observations."""
    return [(step.layout_id, step.x, step.y, step.action) for step in steps]


def local_observation_action_tokens(
    steps: Sequence[RLStep], worlds_by_id: Mapping[str, GridWorld]
) -> list[Hashable]:
    """Tokenize local wall observations paired with actions."""
    tokens: list[Hashable] = []
    for step in steps:
        world = _world_for_step(step, worlds_by_id)
        tokens.append((local_wall_pattern(world, step.x, step.y, radius=1), step.action))
    return tokens


def situation_action_tokens(
    steps: Sequence[RLStep], worlds_by_id: Mapping[str, GridWorld]
) -> list[Hashable]:
    """Tokenize reusable situation types paired with actions."""
    tokens: list[Hashable] = []
    for step in steps:
        world = _world_for_step(step, worlds_by_id)
        tokens.append((classify_situation(world, step.x, step.y), step.action))
    return tokens


def situation_transition_tokens(
    steps: Sequence[RLStep], worlds_by_id: Mapping[str, GridWorld]
) -> list[Hashable]:
    """Tokenize situation-action-next-situation transitions."""
    tokens: list[Hashable] = []
    for step in steps:
        world = _world_for_step(step, worlds_by_id)
        current = classify_situation(world, step.x, step.y)
        next_situation = classify_situation(world, step.next_x, step.next_y)
        tokens.append((current, step.action, next_situation))
    return tokens
