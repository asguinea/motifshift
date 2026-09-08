# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
from pathlib import Path

from motifshift.verify_outputs import verify_outputs


def _run(output_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "motifshift.run_all",
            "--config",
            "configs/smoke.yaml",
            "--outputs",
            str(output_root),
            *args,
        ],
        capture_output=True,
        text=True,
    )


def test_smoke_cli_writes_run_summary(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    result = _run(root)
    assert result.returncode == 0, result.stderr
    summary = (root / "reports/run_summary.md").read_text()
    assert "Run Summary" in summary
    assert "smoke" in summary
    assert str(tmp_path) not in summary
    manifest = json.loads((root / "data/combined_results_manifest.json").read_text())
    assert manifest["missing_artifacts"] == []
    assert manifest["provenance"]["config"]["outputs"]["root"] == "."
    assert verify_outputs(root, strict=True)["status"] == "ok"


def test_smoke_cli_dry_run_does_not_require_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "dry"
    result = _run(root, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "Dry run" in result.stdout
    assert not root.exists()


def test_cli_preserves_existing_outputs(tmp_path: Path) -> None:
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("preserve me")
    result = _run(tmp_path)
    assert result.returncode != 0
    assert "fresh --outputs" in result.stderr
    assert sentinel.read_text() == "preserve me"
    assert list(tmp_path.iterdir()) == [sentinel]
