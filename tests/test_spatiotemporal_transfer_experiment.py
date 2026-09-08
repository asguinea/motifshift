# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import math
from pathlib import Path

from motifshift.spatiotemporal.experiment import run_spatiotemporal_transfer


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
        "experiment": {"name": "test", "seed": 123, "smoke": False},
        "spatiotemporal": {
            "enabled": True,
            "synthetic": {
                "n_scenes": 3,
                "agents_per_scene": 5,
                "timesteps": 10,
            },
            "diagnostics": {
                "enabled": True,
                "top_ks": [10, 50],
                "top_k_jaccard": 10,
            },
            "transfer": {
                "enabled": True,
                "horizon": 1,
                "k_neighbors": 5,
            },
        },
    }


def test_run_spatiotemporal_transfer_writes_expected_artifacts(tmp_path: Path) -> None:
    output_dirs = _output_dirs(tmp_path)

    results = run_spatiotemporal_transfer(_small_config(), output_dirs)

    assert results["experiment"] == "spatiotemporal_transfer"
    assert set(results["predictors"]) == {
        "constant_velocity",
        "mean_displacement",
        "knn_raw",
        "motion_token",
        "situation_motion_token",
    }

    for predictor in results["predictors"].values():
        assert "per_scene" in predictor
        assert "mean" in predictor
        assert "std" in predictor
        assert predictor["mean"]["mean_error"] >= 0
        assert math.isfinite(predictor["mean"]["mean_error"])
        assert math.isfinite(predictor["mean"]["median_error"])

    expected_artifacts = [
        output_dirs["tables"] / "table_spatiotemporal_transfer.tex",
        output_dirs["tables"] / "table_spatiotemporal_transfer.csv",
        output_dirs["data"] / "spatiotemporal_transfer_results.json",
        output_dirs["figures"] / "spatiotemporal_transfer_error.pdf",
        output_dirs["figures"] / "spatiotemporal_transfer_error.png",
        output_dirs["figures"] / "spatiotemporal_transfer_error_std.pdf",
        output_dirs["figures"] / "spatiotemporal_transfer_error_std.png",
    ]
    for artifact in expected_artifacts:
        assert artifact.exists()


def test_spatiotemporal_transfer_is_predictive_not_degenerate(tmp_path: Path) -> None:
    results = run_spatiotemporal_transfer(_small_config(), _output_dirs(tmp_path))
    predictors = results["predictors"]

    assert predictors["constant_velocity"]["mean"]["mean_error"] > 0
    assert predictors["motion_token"]["mean"]["mean_error"] > 0
    assert predictors["situation_motion_token"]["mean"]["mean_error"] > 0
    for predictor in predictors.values():
        assert math.isfinite(predictor["mean"]["mean_error"])


def test_spatiotemporal_transfer_rejects_unsupported_horizon(tmp_path: Path) -> None:
    config = _small_config()
    config["spatiotemporal"]["transfer"]["horizon"] = 2

    try:
        run_spatiotemporal_transfer(config, _output_dirs(tmp_path))
    except ValueError as error:
        assert "horizon=1" in str(error)
    else:
        raise AssertionError("Expected ValueError for unsupported horizon")
