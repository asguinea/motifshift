# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Table reporting helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from motifshift.io import write_text


def format_float(x: float, digits: int = 3) -> str:
    """Format a float for compact reporting."""
    return f"{float(x):.{digits}f}"


def _escape_latex_cell(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("_", r"\_")
    return str(value)


def write_latex_table(
    path: Path,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[Any]],
) -> Path:
    """Write a simple complete LaTeX table environment."""
    alignment = "l" + "r" * max(0, len(columns) - 1)
    escaped_columns = [_escape_latex_cell(column) for column in columns]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{_escape_latex_cell(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(escaped_columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_escape_latex_cell(cell) for cell in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])

    write_text(path, "\n".join(lines))
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write rows to CSV using keys from the first row as the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    return path
