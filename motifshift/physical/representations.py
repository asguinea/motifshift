# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Token representations for synthetic physical fields."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

Record = Mapping[str, Any]


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _value_bin(value: float, bins: int) -> int:
    _require_positive("bins", bins)
    clipped = min(1.0, max(0.0, float(value)))
    return min(bins - 1, int(clipped * bins))


def _delta_bin(delta: float, bins: int, delta_range: float) -> int:
    _require_positive("delta_bins", bins)
    _require_positive("delta_range", delta_range)
    clipped = min(delta_range, max(-delta_range, float(delta)))
    normalized = (clipped + delta_range) / (2.0 * delta_range)
    return min(bins - 1, int(normalized * bins))


def raw_cell_tokens(
    records: Sequence[Record], value_bins: int = 8, grid_bin_size: int = 1
) -> list[tuple[int, int, int]]:
    """Tokenize absolute cell coordinates and value bin."""
    _require_positive("value_bins", value_bins)
    _require_positive("grid_bin_size", grid_bin_size)
    return [
        (
            int(record["x"]) // grid_bin_size,
            int(record["y"]) // grid_bin_size,
            _value_bin(float(record["value"]), value_bins),
        )
        for record in records
    ]


def value_change_tokens(
    records: Sequence[Record],
    value_bins: int = 8,
    delta_bins: int = 9,
    delta_range: float = 0.5,
) -> list[tuple[int, int]]:
    """Tokenize local scalar value and signed change."""
    _require_positive("value_bins", value_bins)
    _require_positive("delta_bins", delta_bins)
    _require_positive("delta_range", delta_range)
    return [
        (
            _value_bin(float(record["value"]), value_bins),
            _delta_bin(float(record["delta"]), delta_bins, delta_range),
        )
        for record in records
    ]


def local_patch_tokens(
    records: Sequence[Record], value_bins: int = 6
) -> list[tuple[int, int, int, int, int]]:
    """Tokenize center and four-neighbor value bins."""
    _require_positive("value_bins", value_bins)
    return [
        (
            _value_bin(float(record["value"]), value_bins),
            _value_bin(float(record["north"]), value_bins),
            _value_bin(float(record["south"]), value_bins),
            _value_bin(float(record["east"]), value_bins),
            _value_bin(float(record["west"]), value_bins),
        )
        for record in records
    ]


def dynamical_regime_tokens(
    records: Sequence[Record],
    value_bins: int = 6,
    grad_direction_bins: int = 8,
    grad_magnitude_bins: Sequence[float] = (0.0, 0.03, 0.08, 0.15, 1e9),
    delta_eps: float = 0.01,
) -> list[tuple[int, int, int, int]]:
    """Tokenize value, change sign, gradient direction, and gradient strength."""
    _require_positive("value_bins", value_bins)
    _require_positive("grad_direction_bins", grad_direction_bins)
    if delta_eps < 0:
        raise ValueError("delta_eps must be non-negative")
    if list(grad_magnitude_bins) != sorted(grad_magnitude_bins):
        raise ValueError("grad_magnitude_bins must be sorted")

    tokens: list[tuple[int, int, int, int]] = []
    for record in records:
        delta = float(record["delta"])
        if delta < -delta_eps:
            delta_sign = -1
        elif delta > delta_eps:
            delta_sign = 1
        else:
            delta_sign = 0

        grad_x = float(record["grad_x"])
        grad_y = float(record["grad_y"])
        grad_mag = float(record["grad_mag"])
        if grad_mag <= 1e-12:
            direction_bin = 0
        else:
            angle = math.atan2(grad_y, grad_x) % (2.0 * math.pi)
            direction_bin = int(math.floor(angle / (2.0 * math.pi) * grad_direction_bins))
            direction_bin = min(direction_bin, grad_direction_bins - 1)
        magnitude_bin = int(
            np.searchsorted(np.asarray(grad_magnitude_bins, dtype=float), grad_mag, side="right")
            - 1
        )
        magnitude_bin = max(0, min(magnitude_bin, len(grad_magnitude_bins) - 1))
        tokens.append(
            (
                _value_bin(float(record["value"]), value_bins),
                delta_sign,
                direction_bin,
                magnitude_bin,
            )
        )
    return tokens
