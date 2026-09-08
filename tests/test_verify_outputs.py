# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
from pathlib import Path

from motifshift.verify_outputs import (
    EXPECTED_RELATIVE_PATHS,
    ROBUSTNESS_RELATIVE_PATHS,
    verify_outputs,
)


def _create_expected_outputs(output_root: Path) -> None:
    for relative_path in EXPECTED_RELATIVE_PATHS:
        path = output_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")


def _create_robustness_outputs(output_root: Path) -> None:
    for relative_path in ROBUSTNESS_RELATIVE_PATHS:
        path = output_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")


def test_verify_outputs_passes_when_expected_files_exist(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _create_expected_outputs(output_root)

    report = verify_outputs(output_root)

    assert report["status"] == "ok"
    assert report["missing"] == []
    assert len(report["checked"]) == len(EXPECTED_RELATIVE_PATHS)


def test_verify_outputs_without_flag_does_not_require_robustness(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    _create_expected_outputs(output_root)

    report = verify_outputs(output_root)

    assert report["status"] == "ok"
    assert len(report["checked"]) == len(EXPECTED_RELATIVE_PATHS)


def test_verify_outputs_with_robustness_requires_robustness_files(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    _create_expected_outputs(output_root)

    report = verify_outputs(output_root, include_robustness=True)

    assert report["status"] == "missing"
    assert str(output_root / ROBUSTNESS_RELATIVE_PATHS[0]) in report["missing"]


def test_verify_outputs_with_robustness_passes_when_files_exist(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    _create_expected_outputs(output_root)
    _create_robustness_outputs(output_root)

    report = verify_outputs(output_root, include_robustness=True)

    assert report["status"] == "ok"
    assert len(report["checked"]) == len(EXPECTED_RELATIVE_PATHS) + len(ROBUSTNESS_RELATIVE_PATHS)


def test_verify_outputs_reports_missing_files(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _create_expected_outputs(output_root)
    (output_root / EXPECTED_RELATIVE_PATHS[0]).unlink()

    report = verify_outputs(output_root)

    assert report["status"] == "missing"
    assert str(output_root / EXPECTED_RELATIVE_PATHS[0]) in report["missing"]


def test_verify_outputs_cli_exit_status_success(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _create_expected_outputs(output_root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "motifshift.verify_outputs",
            "--outputs",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "All required artifacts are present" in result.stdout


def test_verify_outputs_cli_exit_status_failure(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "motifshift.verify_outputs",
            "--outputs",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Missing files" in result.stdout


def test_verify_outputs_json_mode(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _create_expected_outputs(output_root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "motifshift.verify_outputs",
            "--outputs",
            str(output_root),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["status"] == "ok"
    assert payload["missing"] == []
    assert len(payload["checked"]) == len(EXPECTED_RELATIVE_PATHS)


def test_verify_outputs_cli_include_robustness(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _create_expected_outputs(output_root)
    _create_robustness_outputs(output_root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "motifshift.verify_outputs",
            "--outputs",
            str(output_root),
            "--include-robustness",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["status"] == "ok"
    assert len(payload["checked"]) == len(EXPECTED_RELATIVE_PATHS) + len(ROBUSTNESS_RELATIVE_PATHS)
