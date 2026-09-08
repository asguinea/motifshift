# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Portable run provenance without machine-specific paths or Git metadata."""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def capture_provenance(config: dict[str, Any], output_dirs: dict[str, Path]) -> dict[str, Any]:
    snapshot = copy.deepcopy(config)
    snapshot.setdefault("outputs", {})["root"] = "."
    config_path = output_dirs["data"] / "config.json"
    config_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    return {
        "config": snapshot,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_fingerprint(),
        "dependencies": dict(
            sorted((d.metadata["Name"], d.version) for d in distributions() if d.metadata["Name"])
        ),
    }
