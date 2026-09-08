# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Run summary reporting."""

import platform
import sys
from pathlib import Path
from typing import Any

import motifshift
from motifshift.io import write_text


def _enabled(config: dict[str, Any], module: str) -> bool:
    return bool(config.get(module, {}).get("enabled", False))


def write_run_summary(
    config: dict[str, Any],
    output_dirs: dict[str, Path],
    status: str = "success",
    artifacts: list[str] | None = None,
) -> Path:
    """Write a markdown summary for a diagnostic run."""
    experiment = config.get("experiment", {})
    summary_path = output_dirs["reports"] / "run_summary.md"

    root = Path(output_dirs["root"])
    display = {key: str(Path(path).relative_to(root)) for key, path in output_dirs.items()}
    lines = [
        "# MotifShift — Run Summary",
        "",
        f"- Experiment name: {experiment.get('name', 'unknown')}",
        f"- Seed: {experiment.get('seed', 'unknown')}",
        f"- Smoke mode: {bool(experiment.get('smoke', False))}",
        "- Enabled modules:",
        f"  - spatiotemporal: {_enabled(config, 'spatiotemporal')}",
        f"  - rl: {_enabled(config, 'rl')}",
        f"  - physical_field: {_enabled(config, 'physical_field')}",
        "- Output directories:",
        f"  - root: {display['root']}",
        f"  - reports: {display['reports']}",
        f"  - tables: {display['tables']}",
        f"  - figures: {display['figures']}",
        f"  - data: {display['data']}",
        f"- Status: {status}",
        "- Environment:",
        f"  - Python: {sys.version.split()[0]}",
        f"  - Platform: {platform.platform()}",
        f"  - motifshift: {motifshift.__version__}",
        "",
    ]

    if artifacts:
        lines.extend(["## Artifacts", ""])
        lines.extend(f"- {Path(artifact).relative_to(root)}" for artifact in artifacts)
        lines.append("")

    lines.extend(
        [
            "Note: Synthetic diagnostics are lightweight stress tests, not SOTA benchmarks.",
            "",
        ]
    )

    write_text(summary_path, "\n".join(lines))
    return summary_path
