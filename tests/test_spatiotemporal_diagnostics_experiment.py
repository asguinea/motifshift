# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from motifshift.spatiotemporal.experiment import run_spatiotemporal_diagnostics


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
        },
    }


def test_run_spatiotemporal_diagnostics_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    output_dirs = _output_dirs(tmp_path)

    results = run_spatiotemporal_diagnostics(_small_config(), output_dirs)

    assert results["experiment"] == "spatiotemporal_diagnostics"
    assert set(results["representations"]) == {
        "absolute_grid",
        "relative_displacement",
        "motion",
        "situation_motion",
    }

    for representation in results["representations"].values():
        assert "global" in representation
        assert "per_scene" in representation
        assert "cross_scene_mean" in representation
        assert "cross_scene_std" in representation
        assert "effective_vocab_size" in representation["global"]
        assert "token_coverage" in representation["cross_scene_mean"]

    expected_artifacts = [
        output_dirs["tables"] / "table_spatiotemporal_diagnostics.tex",
        output_dirs["tables"] / "table_spatiotemporal_diagnostics.csv",
        output_dirs["data"] / "spatiotemporal_diagnostics_results.json",
        output_dirs["figures"] / "spatiotemporal_cross_scene_coverage.pdf",
        output_dirs["figures"] / "spatiotemporal_cross_scene_coverage.png",
        output_dirs["figures"] / "spatiotemporal_jsd.pdf",
        output_dirs["figures"] / "spatiotemporal_jsd.png",
        output_dirs["figures"] / "spatiotemporal_token_coverage_topk.pdf",
        output_dirs["figures"] / "spatiotemporal_token_coverage_topk.png",
    ]
    for artifact in expected_artifacts:
        assert artifact.exists()


def test_motion_representations_cover_at_least_absolute_grid_cross_scene(
    tmp_path: Path,
) -> None:
    results = run_spatiotemporal_diagnostics(_small_config(), _output_dirs(tmp_path))
    representations = results["representations"]

    absolute_coverage = representations["absolute_grid"]["cross_scene_mean"]["token_coverage"]

    assert representations["motion"]["cross_scene_mean"]["token_coverage"] >= absolute_coverage
    assert (
        representations["situation_motion"]["cross_scene_mean"]["token_coverage"]
        >= absolute_coverage
    )


def test_spatiotemporal_diagnostics_are_not_all_degenerate(tmp_path: Path) -> None:
    results = run_spatiotemporal_diagnostics(_small_config(), _output_dirs(tmp_path))
    representations = results["representations"]

    jsd_values = [
        representation["cross_scene_mean"]["js_divergence"]
        for representation in representations.values()
    ]
    coverage_values = [
        representations[name]["cross_scene_mean"]["token_coverage"]
        for name in ["motion", "situation_motion"]
    ]

    assert any(value > 0 for value in jsd_values)
    assert any(value < 1.0 for value in coverage_values)
