# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Synthetic spatio-temporal trajectory data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrajectoryPoint:
    """A single agent position at a discrete time."""

    scene_id: str
    agent_id: int
    t: int
    x: float
    y: float


@dataclass
class Scene:
    """A lightweight collection of trajectory points for one scene."""

    scene_id: str
    points: list[TrajectoryPoint]


def _motion_delta(pattern: str, step: int) -> tuple[float, float]:
    if pattern == "east":
        return 1.0, 0.0
    if pattern == "north":
        return 0.0, 1.0
    if pattern == "diagonal":
        return 1.0, 1.0
    if pattern == "zigzag":
        return 1.0, 1.0 if step % 2 == 0 else -1.0
    raise ValueError(f"Unknown motion pattern: {pattern}")


def generate_scene(
    scene_id: str,
    n_agents: int = 4,
    n_steps: int = 8,
    origin: tuple[float, float] = (0.0, 0.0),
    agent_spacing: float = 2.0,
    motion_patterns: tuple[str, ...] = ("east", "north", "diagonal", "zigzag"),
) -> Scene:
    """Generate one deterministic scene with reusable per-agent motion patterns."""
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if agent_spacing <= 0:
        raise ValueError("agent_spacing must be positive")
    if not motion_patterns:
        raise ValueError("motion_patterns must be non-empty")

    origin_x, origin_y = origin
    points: list[TrajectoryPoint] = []
    for agent_id in range(n_agents):
        pattern = motion_patterns[agent_id % len(motion_patterns)]
        x = origin_x + agent_id * agent_spacing
        y = origin_y + agent_id * agent_spacing
        points.append(TrajectoryPoint(scene_id=scene_id, agent_id=agent_id, t=0, x=x, y=y))

        for t in range(1, n_steps):
            dx, dy = _motion_delta(pattern, t - 1)
            x += dx
            y += dy
            points.append(TrajectoryPoint(scene_id=scene_id, agent_id=agent_id, t=t, x=x, y=y))

    return Scene(scene_id=scene_id, points=points)


def generate_synthetic_scenes(
    n_scenes: int = 3,
    n_agents: int = 4,
    n_steps: int = 8,
    seed: int = 123,
    noise_std: float = 0.03,
    speed_variation: float = 0.25,
    scene_variation: float = 0.35,
) -> list[Scene]:
    """Generate related but non-identical synthetic trajectory scenes."""
    if n_scenes <= 0:
        raise ValueError("n_scenes must be positive")
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if speed_variation < 0:
        raise ValueError("speed_variation must be non-negative")
    if scene_variation < 0:
        raise ValueError("scene_variation must be non-negative")

    rng = np.random.default_rng(seed)
    scenes: list[Scene] = []
    motifs = ("straight", "turn", "accelerate", "bottleneck", "curve")
    base_weights = np.asarray([0.28, 0.20, 0.20, 0.17, 0.15], dtype=float)
    for scene_index in range(n_scenes):
        translation = np.asarray(
            [
                float(rng.integers(-20, 21) + scene_index * 10),
                float(rng.integers(-20, 21) - scene_index * 5),
            ],
            dtype=float,
        )
        rotation = float(rng.normal(0.0, 0.75 * scene_variation))
        scale = max(0.55, float(1.0 + rng.normal(0.0, 0.25 * scene_variation)))
        speed_scene = max(0.45, float(1.0 + rng.normal(0.0, 0.30 * scene_variation)))
        bottleneck_center = np.asarray(
            [
                float(rng.normal(2.0, 1.5 * scene_variation)),
                float(rng.normal(1.0, 1.5 * scene_variation)),
            ],
            dtype=float,
        )
        weights = base_weights + rng.normal(0.0, 0.10 * scene_variation, size=len(motifs))
        weights = np.maximum(weights, 0.03)
        weights = weights / weights.sum()
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        transform = scale * np.asarray([[cos_r, -sin_r], [sin_r, cos_r]], dtype=float)

        points: list[TrajectoryPoint] = []
        grid_width = max(1, int(np.ceil(np.sqrt(n_agents))))
        for agent_id in range(n_agents):
            motif = str(rng.choice(motifs, p=weights))
            speed = speed_scene * max(0.25, float(1.0 + rng.normal(0.0, speed_variation)))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            heading = float(
                (agent_id % 8) * (np.pi / 4.0) + rng.normal(0.0, 0.35 * scene_variation)
            )
            start = np.asarray(
                [
                    (agent_id % grid_width) * 2.0,
                    (agent_id // grid_width) * 2.0,
                ],
                dtype=float,
            )
            start += rng.normal(0.0, 0.25, size=2)
            local_positions = _motif_positions(
                motif=motif,
                n_steps=n_steps,
                speed=speed,
                heading=heading,
                phase=phase,
                start=start,
                bottleneck_center=bottleneck_center,
                scene_variation=scene_variation,
            )
            if noise_std:
                local_positions += rng.normal(0.0, noise_std, size=local_positions.shape)

            world_positions = local_positions @ transform.T + translation
            for t, position in enumerate(world_positions):
                points.append(
                    TrajectoryPoint(
                        scene_id=f"scene_{scene_index}",
                        agent_id=agent_id,
                        t=t,
                        x=float(position[0]),
                        y=float(position[1]),
                    )
                )

        scenes.append(Scene(scene_id=f"scene_{scene_index}", points=points))

    return scenes


def _motif_positions(
    motif: str,
    n_steps: int,
    speed: float,
    heading: float,
    phase: float,
    start: np.ndarray,
    bottleneck_center: np.ndarray,
    scene_variation: float,
) -> np.ndarray:
    times = np.arange(n_steps, dtype=float)
    direction = np.asarray([np.cos(heading), np.sin(heading)], dtype=float)
    lateral = np.asarray([-direction[1], direction[0]], dtype=float)

    if motif == "straight":
        positions = start + np.outer(times * speed, direction)
        positions += np.outer(0.15 * np.sin(0.4 * times + phase), lateral)
    elif motif == "turn":
        curvature = 0.09 + 0.10 * scene_variation
        angles = heading + curvature * times
        deltas = speed * np.column_stack([np.cos(angles), np.sin(angles)])
        positions = start + np.vstack([np.zeros(2), np.cumsum(deltas[:-1], axis=0)])
    elif motif == "accelerate":
        acceleration = 0.04 + 0.08 * (0.5 + np.sin(phase) ** 2)
        distances = speed * times + 0.5 * acceleration * times**2
        positions = start + np.outer(distances, direction)
        positions += np.outer(0.10 * np.sin(0.5 * times + phase), lateral)
    elif motif == "bottleneck":
        entry = start - direction * (1.5 + 0.4 * np.sin(phase))
        exit_point = bottleneck_center + direction * (n_steps * speed * 0.65)
        alpha = times / max(1.0, n_steps - 1)
        approach = (1.0 - alpha)[:, None] * entry + alpha[:, None] * exit_point
        squeeze = np.exp(-((alpha - 0.5) ** 2) / 0.035)
        positions = approach + np.outer(0.45 * (1.0 - squeeze) * np.sin(phase), lateral)
    elif motif == "curve":
        radius = 4.0 + 2.0 * np.sin(phase) ** 2
        angular_speed = speed / radius
        angles = heading + angular_speed * times
        center = start + radius * lateral
        positions = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        raise ValueError(f"Unknown synthetic motif: {motif}")

    return positions


def trajectory_steps(scene: Scene) -> list[dict[str, Any]]:
    """Return one-step transition records for a scene."""
    by_agent: dict[int, list[TrajectoryPoint]] = {}
    for point in scene.points:
        by_agent.setdefault(point.agent_id, []).append(point)

    steps: list[dict[str, Any]] = []
    for agent_id, points in by_agent.items():
        ordered = sorted(points, key=lambda point: point.t)
        for current, next_point in zip(ordered[:-1], ordered[1:]):
            dx = next_point.x - current.x
            dy = next_point.y - current.y
            steps.append(
                {
                    "scene_id": scene.scene_id,
                    "agent_id": agent_id,
                    "t": current.t,
                    "x": current.x,
                    "y": current.y,
                    "next_x": next_point.x,
                    "next_y": next_point.y,
                    "dx": dx,
                    "dy": dy,
                    "speed": float(np.hypot(dx, dy)),
                }
            )
    return sorted(steps, key=lambda step: (step["scene_id"], step["agent_id"], step["t"]))


def predictive_trajectory_steps(scene: Scene) -> list[dict[str, Any]]:
    """Return records for predicting the next displacement from current motion."""
    by_agent: dict[int, list[TrajectoryPoint]] = {}
    for point in scene.points:
        by_agent.setdefault(point.agent_id, []).append(point)

    steps: list[dict[str, Any]] = []
    for agent_id, points in by_agent.items():
        ordered = sorted(points, key=lambda point: point.t)
        for previous, current, next_point in zip(ordered[:-2], ordered[1:-1], ordered[2:]):
            prev_dx = current.x - previous.x
            prev_dy = current.y - previous.y
            target_dx = next_point.x - current.x
            target_dy = next_point.y - current.y
            steps.append(
                {
                    "scene_id": scene.scene_id,
                    "agent_id": agent_id,
                    "t": current.t,
                    "prev_x": previous.x,
                    "prev_y": previous.y,
                    "x": current.x,
                    "y": current.y,
                    "next_x": next_point.x,
                    "next_y": next_point.y,
                    "prev_dx": prev_dx,
                    "prev_dy": prev_dy,
                    "dx": prev_dx,
                    "dy": prev_dy,
                    "target_dx": target_dx,
                    "target_dy": target_dy,
                    "speed": float(np.hypot(prev_dx, prev_dy)),
                    "target_speed": float(np.hypot(target_dx, target_dy)),
                }
            )
    return sorted(steps, key=lambda step: (step["scene_id"], step["agent_id"], step["t"]))


def flatten_scenes(scenes: list[Scene]) -> list[TrajectoryPoint]:
    """Flatten scenes into one point list, preserving scene order."""
    return [point for scene in scenes for point in scene.points]
