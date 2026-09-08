# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Figure reporting helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

matplotlib_cache = Path(tempfile.gettempdir()) / "motifshift_matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_bar_chart(
    path_pdf: Path,
    path_png: Path,
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
) -> tuple[Path, Path]:
    """Save a bar chart as PDF and PNG."""
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    path_png.parent.mkdir(parents=True, exist_ok=True)

    horizontal = len(labels) > 6 or any(len(label) > 22 for label in labels)
    figure, axis = plt.subplots(
        figsize=(10, max(4, len(labels) * 0.4 + 1.2)) if horizontal else (8, 4.5)
    )
    if horizontal:
        axis.barh(labels, values)
        axis.invert_yaxis()
        axis.set_xlabel(ylabel)
        axis.grid(axis="x", alpha=0.2)
    else:
        axis.bar(labels, values)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", labelrotation=25)
        for tick in axis.get_xticklabels():
            tick.set_horizontalalignment("right")
        axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path_pdf)
    figure.savefig(path_png, dpi=200)
    plt.close(figure)
    return path_pdf, path_png


def save_heatmap(
    path_pdf: Path,
    path_png: Path,
    matrix: Any,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    colorbar_label: str,
) -> tuple[Path, Path]:
    """Save a heatmap as PDF and PNG."""
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    path_png.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 4))
    image = axis.imshow(matrix, aspect="auto")
    axis.set_xticks(range(len(x_labels)), labels=x_labels, rotation=25, ha="right")
    axis.set_yticks(range(len(y_labels)), labels=y_labels)
    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    figure.tight_layout()
    figure.savefig(path_pdf)
    figure.savefig(path_png, dpi=200)
    plt.close(figure)
    return path_pdf, path_png
