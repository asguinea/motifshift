# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Spatio-temporal token representations."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from motifshift.spatiotemporal.data import TrajectoryPoint


def _bin_value(value: float, upper_bounds: Sequence[float]) -> int:
    _validate_bins(upper_bounds)
    return int(np.searchsorted(np.asarray(upper_bounds, dtype=float), value, side="left"))


def _validate_bins(upper_bounds: Sequence[float]) -> None:
    if any(bound < 0 for bound in upper_bounds):
        raise ValueError("bin upper bounds must be non-negative")
    if list(upper_bounds) != sorted(upper_bounds):
        raise ValueError("bin upper bounds must be sorted")


def _validate_cell_size(cell_size: float) -> None:
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")


def _sorted_points(points: Sequence[TrajectoryPoint]) -> list[TrajectoryPoint]:
    return sorted(points, key=lambda point: (point.scene_id, point.agent_id, point.t))


def _transitions(
    points: Sequence[TrajectoryPoint],
) -> list[tuple[TrajectoryPoint, TrajectoryPoint]]:
    by_track: dict[tuple[str, int], list[TrajectoryPoint]] = defaultdict(list)
    for point in points:
        by_track[(point.scene_id, point.agent_id)].append(point)

    transitions: list[tuple[TrajectoryPoint, TrajectoryPoint]] = []
    for track_points in by_track.values():
        ordered = sorted(track_points, key=lambda point: point.t)
        transitions.extend(zip(ordered[:-1], ordered[1:]))

    return sorted(
        transitions,
        key=lambda pair: (pair[1].scene_id, pair[1].agent_id, pair[1].t),
    )


def _motion_components(
    previous: TrajectoryPoint,
    current: TrajectoryPoint,
    speed_bins: Sequence[float],
    direction_bins: int,
) -> tuple[int, int]:
    if direction_bins <= 0:
        raise ValueError("direction_bins must be positive")

    dx = current.x - previous.x
    dy = current.y - previous.y
    speed = math.hypot(dx, dy)
    speed_bin = _bin_value(speed, speed_bins)

    if speed == 0:
        direction_bin = -1
    else:
        angle = math.atan2(dy, dx) % (2 * math.pi)
        direction_bin = int(math.floor(angle / (2 * math.pi) * direction_bins))
        direction_bin = min(direction_bin, direction_bins - 1)

    return speed_bin, direction_bin


def absolute_grid_tokens(
    points: Sequence[TrajectoryPoint],
    cell_size: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[str, int, int]]:
    """Tokenize absolute positions into grid cells."""
    _validate_cell_size(cell_size)
    origin_x, origin_y = origin

    return [
        (
            "abs",
            math.floor((point.x - origin_x) / cell_size),
            math.floor((point.y - origin_y) / cell_size),
        )
        for point in _sorted_points(points)
    ]


def relative_displacement_grid_tokens(
    points: Sequence[TrajectoryPoint], cell_size: float = 1.0
) -> list[tuple[str, int, int]]:
    """Tokenize within-agent displacements into grid cells."""
    _validate_cell_size(cell_size)

    tokens: list[tuple[str, int, int]] = []
    for previous, current in _transitions(points):
        dx = current.x - previous.x
        dy = current.y - previous.y
        tokens.append(("rel", round(dx / cell_size), round(dy / cell_size)))

    return tokens


def motion_tokens(
    points: Sequence[TrajectoryPoint],
    speed_bins: Sequence[float] = (0.4, 0.8, 1.2, 1.6, 2.2, 3.0),
    direction_bins: int = 16,
) -> list[tuple[str, int, int]]:
    """Tokenize transitions as speed-bin and direction-bin pairs."""
    if direction_bins <= 0:
        raise ValueError("direction_bins must be positive")
    _validate_bins(speed_bins)

    return [
        ("motion", *(_motion_components(previous, current, speed_bins, direction_bins)))
        for previous, current in _transitions(points)
    ]


def situation_motion_tokens(
    points: Sequence[TrajectoryPoint],
    speed_bins: Sequence[float] = (0.4, 0.8, 1.2, 1.6, 2.2, 3.0),
    direction_bins: int = 16,
    distance_bins: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    density_bins: Sequence[float] = (0.0, 1.0, 2.0, 4.0),
    density_radius: float = 3.0,
) -> list[tuple[str, int, int, int, int]]:
    """Tokenize motion together with local crowding context at the endpoint."""
    if density_radius < 0:
        raise ValueError("density_radius must be non-negative")
    if direction_bins <= 0:
        raise ValueError("direction_bins must be positive")
    _validate_bins(speed_bins)
    _validate_bins(distance_bins)
    _validate_bins(density_bins)

    points_by_scene_time: dict[tuple[str, int], list[TrajectoryPoint]] = defaultdict(list)
    for point in points:
        points_by_scene_time[(point.scene_id, point.t)].append(point)

    tokens: list[tuple[str, int, int, int, int]] = []
    for previous, current in _transitions(points):
        speed_bin, direction_bin = _motion_components(previous, current, speed_bins, direction_bins)
        neighbors = [
            point
            for point in points_by_scene_time[(current.scene_id, current.t)]
            if point.agent_id != current.agent_id
        ]
        distances = [math.hypot(point.x - current.x, point.y - current.y) for point in neighbors]
        nearest_distance = min(distances) if distances else math.inf
        local_density = sum(1 for distance in distances if distance <= density_radius)

        distance_bin = _bin_value(nearest_distance, distance_bins)
        density_bin = _bin_value(float(local_density), density_bins)
        tokens.append(("situation_motion", speed_bin, direction_bin, distance_bin, density_bin))

    return tokens
