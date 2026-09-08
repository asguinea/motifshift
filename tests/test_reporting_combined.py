# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from motifshift.reporting.combined import (
    write_artifact_manifest,
    write_combined_diagnostics_table,
    write_combined_transfer_table,
    write_results_summary,
)


def _output_dirs(tmp_path: Path) -> dict[str, Path]:
    return {
        "root": tmp_path,
        "reports": tmp_path / "reports",
        "tables": tmp_path / "tables",
        "figures": tmp_path / "figures",
        "data": tmp_path / "data",
    }


def _global() -> dict:
    return {
        "vocab_size": 10,
        "normalized_effective_vocab_size": 0.25,
    }


def _cross() -> dict:
    return {
        "token_coverage": 0.8,
        "js_divergence": 0.1,
        "bigram_coverage": 0.6,
        "novel_recombination_rate": 0.2,
    }


def _spatiotemporal_results() -> dict:
    return {
        "experiment": "spatiotemporal_diagnostics",
        "representations": {
            "motion": {
                "global": _global(),
                "cross_scene_mean": _cross(),
            }
        },
    }


def _spatiotemporal_transfer_results() -> dict:
    return {
        "experiment": "spatiotemporal_transfer",
        "predictors": {
            "motion_token": {
                "mean": {"mean_error": 0.1},
                "std": {"mean_error": 0.01},
            }
        },
    }


def _rl_results() -> dict:
    return {
        "experiment": "rl_diagnostics_transfer",
        "representations": {
            "situation_action": {
                "global": _global(),
                "cross_layout_mean": _cross(),
                "prediction_transfer_mean": {"next_token_accuracy": 0.7},
            }
        },
    }


def _physical_results() -> dict:
    return {
        "experiment": "physical_field_diagnostics_transfer",
        "representations": {
            "dynamical_regime": {
                "global": _global(),
                "cross_environment_mean": _cross(),
            }
        },
        "predictors": {
            "dynamical_regime_token": {
                "mean": {"mean_abs_error": 0.05},
                "std": {"mean_abs_error": 0.01},
            }
        },
    }


def test_write_combined_diagnostics_table(tmp_path: Path) -> None:
    output_dirs = _output_dirs(tmp_path)

    artifacts = write_combined_diagnostics_table(
        output_dirs, _spatiotemporal_results(), _rl_results(), _physical_results()
    )

    latex = Path(artifacts["combined_diagnostics_latex"])
    csv = Path(artifacts["combined_diagnostics_csv"])
    assert latex.exists()
    assert csv.exists()
    assert "Spatio-temporal" in latex.read_text(encoding="utf-8")
    assert "RL" in csv.read_text(encoding="utf-8")
    assert "Physical field" in csv.read_text(encoding="utf-8")
    assert "Cross-Context" in csv.read_text(encoding="utf-8")


def test_write_combined_transfer_table(tmp_path: Path) -> None:
    output_dirs = _output_dirs(tmp_path)

    artifacts = write_combined_transfer_table(
        output_dirs,
        _spatiotemporal_transfer_results(),
        _rl_results(),
        _physical_results(),
    )

    latex = Path(artifacts["combined_transfer_latex"])
    csv = Path(artifacts["combined_transfer_csv"])
    assert latex.exists()
    assert csv.exists()
    assert "Spatio-temporal" in latex.read_text(encoding="utf-8")
    assert "Next-token acc." in csv.read_text(encoding="utf-8")
    assert "Physical field" in csv.read_text(encoding="utf-8")


def test_write_artifact_manifest(tmp_path: Path) -> None:
    output_dirs = _output_dirs(tmp_path)
    combined = write_combined_diagnostics_table(
        output_dirs, _spatiotemporal_results(), _rl_results()
    )

    path = write_artifact_manifest(
        output_dirs,
        {
            "config_path": "configs/default.yaml",
            "combined": list(combined.values()),
        },
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["config_path"] == "configs/default.yaml"
    assert "combined" in manifest["artifacts"]
    assert "Synthetic diagnostic" in manifest["note"]


def test_write_results_summary(tmp_path: Path) -> None:
    output_dirs = _output_dirs(tmp_path)
    write_combined_diagnostics_table(
        output_dirs, _spatiotemporal_results(), _rl_results(), _physical_results()
    )
    write_combined_transfer_table(
        output_dirs,
        _spatiotemporal_transfer_results(),
        _rl_results(),
        _physical_results(),
    )

    path = write_results_summary(
        output_dirs,
        _spatiotemporal_results(),
        _spatiotemporal_transfer_results(),
        _rl_results(),
        _physical_results(),
    )

    content = path.read_text(encoding="utf-8")
    assert "# Results Summary" in content
    assert "Synthetic diagnostic" in content
    assert "Spatio-temporal" in content
    assert "RL" in content
    assert "Physical-field" in content


def test_combined_reporting_handles_none_inputs(tmp_path: Path) -> None:
    output_dirs = _output_dirs(tmp_path)

    diagnostics = write_combined_diagnostics_table(output_dirs, None, None)
    transfer = write_combined_transfer_table(output_dirs, None, None)
    summary = write_results_summary(output_dirs, None, None, None)

    assert Path(diagnostics["combined_diagnostics_latex"]).exists()
    assert Path(transfer["combined_transfer_csv"]).exists()
    assert "no diagnostic results" in summary.read_text(encoding="utf-8")
