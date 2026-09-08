# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Deterministic seed handling."""

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed supported random number generators."""
    random.seed(seed)
    np.random.seed(seed)
