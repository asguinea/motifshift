# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Filesystem helpers for suite outputs."""

from pathlib import Path
from typing import Any


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Create the configured output directory tree and return its paths."""
    outputs = config.get("outputs", {})
    root = Path(outputs.get("root", "outputs"))

    output_dirs = {
        "root": root,
        "reports": root / outputs.get("reports_dir", "reports"),
        "tables": root / outputs.get("tables_dir", "tables"),
        "figures": root / outputs.get("figures_dir", "figures"),
        "data": root / outputs.get("data_dir", "data"),
    }

    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    return output_dirs


def write_text(path: Path, content: str) -> None:
    """Write UTF-8 text, creating the parent directory if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
