# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import math
from pathlib import Path

from motifshift.physical.experiment import run_physical_field_diagnostics_and_transfer


def _output_dirs(tmp_path: Path) -> dict[str, Path]:
    return {
        "root": tmp_path,
        "reports": tmp_path / "reports",
        "tables": tmp_path / "tables",
        "figures": tmp_path / "figures",
        "data": tmp_path / "data",
    }


def _small_config() -> dict:
    return {
        "experiment": {"seed": 123},
        "physical_field": {
            "enabled": True,
            "synthetic": {
                "n_environments": 3,
                "grid_size": 10,
                "timesteps": 6,
                "noise_std": 0.01,
            },
            "diagnostics": {
                "enabled": True,
                "top_ks": [10, 50],
                "top_k_jaccard": 10,
            },
            "transfer": {"enabled": True},
        },
    }


def test_physical_experiment_writes_expected_artifacts(tmp_path: Path) -> None:
    output_dirs = _output_dirs(tmp_path)

    results = run_physical_field_diagnostics_and_transfer(_small_config(), output_dirs)

    assert results["experiment"] == "physical_field_diagnostics_transfer"
    assert set(results["representations"]) == {
        "raw_cell",
        "value_change",
        "local_patch",
        "dynamical_regime",
    }
    assert set(results["predictors"]) == {
        "mean_value",
        "raw_cell_lookup",
        "value_change_token",
        "local_patch_token",
        "dynamical_regime_token",
    }

    for representation in results["representations"].values():
        assert "global" in representation
        assert "per_environment" in representation
        assert "cross_environment_mean" in representation
        assert math.isfinite(representation["cross_environment_mean"]["token_coverage"])
    for predictor in results["predictors"].values():
        assert "mean" in predictor
        assert math.isfinite(predictor["mean"]["mean_abs_error"])

    expected = [
        output_dirs["data"] / "physical_field_results.json",
        output_dirs["tables"] / "table_physical_field_diagnostics.tex",
        output_dirs["tables"] / "table_physical_field_diagnostics.csv",
        output_dirs["tables"] / "table_physical_field_transfer.tex",
        output_dirs["tables"] / "table_physical_field_transfer.csv",
        output_dirs["figures"] / "physical_field_cross_env_coverage.pdf",
        output_dirs["figures"] / "physical_field_cross_env_coverage.png",
        output_dirs["figures"] / "physical_field_jsd.pdf",
        output_dirs["figures"] / "physical_field_jsd.png",
        output_dirs["figures"] / "physical_field_transfer_error.pdf",
        output_dirs["figures"] / "physical_field_transfer_error.png",
    ]
    for path in expected:
        assert path.exists()


def test_physical_experiment_is_non_degenerate(tmp_path: Path) -> None:
    results = run_physical_field_diagnostics_and_transfer(_small_config(), _output_dirs(tmp_path))
    representations = results["representations"]
    coverages = [
        representation["cross_environment_mean"]["token_coverage"]
        for representation in representations.values()
    ]
    raw = representations["raw_cell"]["cross_environment_mean"]["token_coverage"]
    abstract = [
        representations["value_change"]["cross_environment_mean"]["token_coverage"],
        representations["dynamical_regime"]["cross_environment_mean"]["token_coverage"],
    ]

    assert len(set(round(value, 3) for value in coverages)) > 1
    assert any(value >= raw for value in abstract)
