# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Synthetic gridworld generation and replay utilities."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ACTIONS: tuple[str, ...] = ("up", "down", "left", "right")
ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(frozen=True)
class GridWorld:
    """A small deterministic gridworld layout."""

    layout_id: str
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    start: tuple[int, int]
    goal: tuple[int, int]


@dataclass(frozen=True)
class RLStep:
    """One state-action-next-state transition from a gridworld episode."""

    layout_id: str
    episode_id: int
    t: int
    x: int
    y: int
    action: str
    next_x: int
    next_y: int
    done: bool


def in_bounds(world: GridWorld, pos: tuple[int, int]) -> bool:
    """Return whether a position is inside the grid."""
    x, y = pos
    return 0 <= x < world.width and 0 <= y < world.height


def is_free(world: GridWorld, pos: tuple[int, int]) -> bool:
    """Return whether a position is in bounds and not a wall."""
    return in_bounds(world, pos) and pos not in world.walls


def step_position(world: GridWorld, pos: tuple[int, int], action: str) -> tuple[int, int]:
    """Apply an action, staying in place on wall or boundary collisions."""
    if action not in ACTION_DELTAS:
        raise ValueError(f"Unknown action: {action}")
    dx, dy = ACTION_DELTAS[action]
    next_pos = (pos[0] + dx, pos[1] + dy)
    return next_pos if is_free(world, next_pos) else pos


def neighbors4(world: GridWorld, pos: tuple[int, int]) -> list[tuple[int, int]]:
    """Return valid free four-neighbors."""
    return [
        next_pos for action in ACTIONS if (next_pos := step_position(world, pos, action)) != pos
    ]


def _shortest_path_between(
    world: GridWorld, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    if not is_free(world, start) or not is_free(world, goal):
        return []

    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        pos = queue.popleft()
        if pos == goal:
            break
        for neighbor in neighbors4(world, pos):
            if neighbor not in parent:
                parent[neighbor] = pos
                queue.append(neighbor)

    if goal not in parent:
        return []

    path = [goal]
    while path[-1] != start:
        previous = parent[path[-1]]
        if previous is None:
            break
        path.append(previous)
    path.reverse()
    return path


def shortest_path(world: GridWorld) -> list[tuple[int, int]]:
    """Return a BFS shortest path from start to goal, or an empty list."""
    return _shortest_path_between(world, world.start, world.goal)


def _carved_corridor(width: int, height: int, gap_x: int, gap_y: int) -> set[tuple[int, int]]:
    path: set[tuple[int, int]] = set()
    path.update((x, 0) for x in range(gap_x + 1))
    path.update((gap_x, y) for y in range(gap_y + 1))
    path.update((x, gap_y) for x in range(gap_x, width))
    path.update((width - 1, y) for y in range(gap_y, height))
    return path


def _candidate_walls(
    width: int, height: int, layout_index: int, rng: np.random.Generator
) -> set[tuple[int, int]]:
    walls: set[tuple[int, int]] = set()
    mid_x = width // 2
    mid_y = height // 2
    gap_y = int(rng.integers(1, max(2, height - 1)))
    gap_x = int(rng.integers(1, max(2, width - 1)))

    walls.update((mid_x, y) for y in range(1, height - 1) if y != gap_y)
    walls.update((x, mid_y) for x in range(1, width - 1) if x != gap_x)

    block_w = max(2, min(3, width - 4))
    block_h = max(2, min(3, height - 4))
    if width >= 7 and height >= 7:
        block_x = int(rng.integers(1, width - block_w - 1))
        block_y = int(rng.integers(1, height - block_h - 1))
        walls.update(
            (x, y)
            for x in range(block_x, block_x + block_w)
            for y in range(block_y, block_y + block_h)
        )

    for offset in range(2, width - 2, 4):
        y = 1 + ((offset + layout_index) % max(1, height - 2))
        walls.add((offset, y))

    return walls


def generate_gridworlds(
    n_layouts: int = 5,
    width: int = 12,
    height: int = 12,
    seed: int = 123,
) -> list[GridWorld]:
    """Generate deterministic layouts with varied walls and guaranteed paths."""
    if n_layouts <= 0:
        raise ValueError("n_layouts must be positive")
    if width < 4 or height < 4:
        raise ValueError("width and height must be at least 4")

    rng = np.random.default_rng(seed)
    worlds: list[GridWorld] = []
    start = (0, 0)
    goal = (width - 1, height - 1)

    for layout_index in range(n_layouts):
        gap_x = int(rng.integers(1, width - 1))
        gap_y = int(rng.integers(1, height - 1))
        carved = _carved_corridor(width, height, gap_x, gap_y)
        walls = _candidate_walls(width, height, layout_index, rng)
        walls.difference_update(carved)
        walls.discard(start)
        walls.discard(goal)

        world = GridWorld(
            layout_id=f"layout_{layout_index}",
            width=width,
            height=height,
            walls=frozenset(walls),
            start=start,
            goal=goal,
        )
        if not shortest_path(world):
            world = GridWorld(
                layout_id=world.layout_id,
                width=width,
                height=height,
                walls=frozenset(),
                start=start,
                goal=goal,
            )
        worlds.append(world)

    return worlds


def action_from_to(pos: tuple[int, int], next_pos: tuple[int, int]) -> str:
    """Return the action describing a one-step displacement."""
    dx = next_pos[0] - pos[0]
    dy = next_pos[1] - pos[1]
    if (dx, dy) == (0, 0):
        return "stay"
    for action, delta in ACTION_DELTAS.items():
        if delta == (dx, dy):
            return action
    raise ValueError(f"Invalid one-step displacement: {pos} -> {next_pos}")


def _valid_actions(world: GridWorld, pos: tuple[int, int]) -> list[str]:
    return [action for action in ACTIONS if step_position(world, pos, action) != pos]


def generate_episodes(
    worlds: Sequence[GridWorld],
    episodes_per_world: int = 30,
    max_steps: int = 50,
    seed: int = 123,
    noise_prob: float = 0.15,
) -> list[RLStep]:
    """Generate deterministic goal-directed episodes with light action noise."""
    if episodes_per_world <= 0:
        raise ValueError("episodes_per_world must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0 <= noise_prob <= 1:
        raise ValueError("noise_prob must be between 0 and 1")

    rng = np.random.default_rng(seed)
    steps: list[RLStep] = []
    episode_id = 0

    for world in worlds:
        if not shortest_path(world):
            raise ValueError(f"World has no valid start-goal path: {world.layout_id}")
        for _ in range(episodes_per_world):
            pos = world.start
            for t in range(max_steps):
                path = _shortest_path_between(world, pos, world.goal)
                if len(path) < 2:
                    break
                path_action = action_from_to(pos, path[1])
                valid_actions = _valid_actions(world, pos)
                if valid_actions and rng.random() < noise_prob:
                    action = str(rng.choice(valid_actions))
                else:
                    action = path_action
                next_pos = step_position(world, pos, action)
                done = next_pos == world.goal
                steps.append(
                    RLStep(
                        layout_id=world.layout_id,
                        episode_id=episode_id,
                        t=t,
                        x=pos[0],
                        y=pos[1],
                        action=action,
                        next_x=next_pos[0],
                        next_y=next_pos[1],
                        done=done,
                    )
                )
                pos = next_pos
                if done:
                    break
            episode_id += 1

    return steps


def local_wall_pattern(world: GridWorld, x: int, y: int, radius: int = 1) -> tuple[int, ...]:
    """Return a flattened binary wall/out-of-bounds pattern around a cell."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    pattern: list[int] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            pos = (x + dx, y + dy)
            pattern.append(0 if is_free(world, pos) else 1)
    return tuple(pattern)


def classify_situation(world: GridWorld, x: int, y: int) -> str:
    """Classify a cell by local four-neighbor topology."""
    pos = (x, y)
    if pos == world.goal:
        return "goal"
    if not is_free(world, pos):
        return "other"

    free_neighbors = neighbors4(world, pos)
    degree = len(free_neighbors)
    if degree <= 1:
        return "dead_end"
    if degree == 2:
        first, second = free_neighbors
        if first[0] == second[0] or first[1] == second[1]:
            return "corridor"
        return "corner"
    if degree == 3:
        return "junction"
    if degree == 4:
        return "open_area"
    return "other"
