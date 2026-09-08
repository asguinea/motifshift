# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Copy a reviewed source tree into a fresh directory without local state."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT_FILES = [
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
]
SOURCE_TYPES = {
    "motifshift": {".py"},
    "tests": {".py"},
    "configs": {".yaml"},
    "docs": {".md", ".png"},
    ".github": {".md", ".yml"},
    "tools": {".py"},
}


def export_source(root: Path, destination: Path) -> int:
    root = root.resolve()
    destination = destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("destination must be outside the source tree")
    if destination.exists():
        raise FileExistsError("destination must be a new directory")
    paths = [root / name for name in ROOT_FILES]
    for directory, suffixes in SOURCE_TYPES.items():
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and path.suffix in suffixes
        )
    for path in paths:
        if not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError(f"missing file or symlink in source export: {path.relative_to(root)}")
    destination.mkdir(parents=True)
    for path in sorted(paths):
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return len(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = export_source(Path(__file__).resolve().parents[1], args.destination)
    print(f"Exported {count} source files to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
