# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Synthetic weather-like scalar field generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FieldEnvironment:
    """A scalar field evolving over time under simple local dynamics."""

    env_id: str
    fields: np.ndarray
    wind: tuple[float, float]
    diffusion: float
    source: tuple[int, int]


def _gaussian_patch(grid_size: int, center: tuple[float, float], sigma: float) -> np.ndarray:
    y, x = np.mgrid[0:grid_size, 0:grid_size]
    cx, cy = center
    return np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2)))


def _average_four_neighbors(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="edge")
    return (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0


def generate_physical_field_environments(
    n_environments: int = 5,
    grid_size: int = 24,
    timesteps: int = 30,
    seed: int = 123,
    noise_std: float = 0.01,
) -> list[FieldEnvironment]:
    """Generate deterministic synthetic advection/diffusion-like fields."""
    if n_environments <= 0:
        raise ValueError("n_environments must be positive")
    if grid_size < 5:
        raise ValueError("grid_size must be at least 5")
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    rng = np.random.default_rng(seed)
    environments: list[FieldEnvironment] = []
    for env_index in range(n_environments):
        angle = float(2.0 * np.pi * env_index / max(1, n_environments))
        angle += float(rng.normal(0.0, 0.25))
        speed = float(rng.uniform(0.45, 1.25))
        wind = (speed * np.cos(angle), speed * np.sin(angle))
        diffusion = float(rng.uniform(0.08, 0.22))
        source = (
            int(rng.integers(2, grid_size - 2)),
            int(rng.integers(2, grid_size - 2)),
        )
        phase_center = (
            float(rng.uniform(0.25 * grid_size, 0.75 * grid_size)),
            float(rng.uniform(0.25 * grid_size, 0.75 * grid_size)),
        )

        field = 0.75 * _gaussian_patch(grid_size, phase_center, sigma=max(2.0, grid_size / 7.0))
        field += 0.15 * _gaussian_patch(
            grid_size,
            (float(source[0]), float(source[1])),
            sigma=max(1.5, grid_size / 10.0),
        )
        field += rng.normal(0.0, noise_std, size=(grid_size, grid_size))
        field = np.clip(field, 0.0, 1.0)

        fields = np.zeros((timesteps, grid_size, grid_size), dtype=float)
        fields[0] = field
        shift_accumulator = np.zeros(2, dtype=float)
        for t in range(1, timesteps):
            shift_accumulator += np.asarray(wind)
            shift_x = int(np.round(shift_accumulator[0]))
            shift_y = int(np.round(shift_accumulator[1]))
            shift_accumulator -= np.asarray([shift_x, shift_y], dtype=float)
            advected = np.roll(field, shift=(shift_y, shift_x), axis=(0, 1))
            smoothed = _average_four_neighbors(advected)
            field = (1.0 - diffusion) * advected + diffusion * smoothed

            source_strength = 0.05 + 0.025 * np.sin(0.4 * t + env_index)
            source_patch = _gaussian_patch(
                grid_size,
                (float(source[0]), float(source[1])),
                sigma=max(1.2, grid_size / 14.0),
            )
            field += source_strength * source_patch
            if noise_std:
                field += rng.normal(0.0, noise_std, size=field.shape)
            field = np.clip(field, 0.0, 1.0)
            fields[t] = field

        environments.append(
            FieldEnvironment(
                env_id=f"field_{env_index}",
                fields=fields,
                wind=(float(wind[0]), float(wind[1])),
                diffusion=diffusion,
                source=source,
            )
        )

    return environments


def field_records(env: FieldEnvironment) -> list[dict[str, Any]]:
    """Return cell transition records from t to t+1 for an environment."""
    fields = env.fields
    if fields.ndim != 3:
        raise ValueError("env.fields must have shape (timesteps, height, width)")

    timesteps, height, width = fields.shape
    records: list[dict[str, Any]] = []
    for t in range(timesteps - 1):
        current = fields[t]
        next_field = fields[t + 1]
        for y in range(height):
            for x in range(width):
                value = float(current[y, x])
                north = float(current[max(0, y - 1), x])
                south = float(current[min(height - 1, y + 1), x])
                west = float(current[y, max(0, x - 1)])
                east = float(current[y, min(width - 1, x + 1)])
                grad_x = (east - west) / 2.0
                grad_y = (south - north) / 2.0
                next_value = float(next_field[y, x])
                records.append(
                    {
                        "env_id": env.env_id,
                        "t": t,
                        "x": x,
                        "y": y,
                        "value": value,
                        "next_value": next_value,
                        "delta": next_value - value,
                        "prev_delta": value - float(fields[t - 1, y, x]) if t > 0 else 0.0,
                        "north": north,
                        "south": south,
                        "east": east,
                        "west": west,
                        "grad_x": grad_x,
                        "grad_y": grad_y,
                        "grad_mag": float(np.hypot(grad_x, grad_y)),
                    }
                )
    return records
