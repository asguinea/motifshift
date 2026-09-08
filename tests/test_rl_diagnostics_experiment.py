# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from motifshift.rl.experiment import run_rl_diagnostics_and_transfer


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
        "rl": {
            "enabled": True,
            "synthetic": {
                "n_layouts": 3,
                "width": 8,
                "height": 8,
                "episodes_per_world": 5,
                "max_steps": 25,
                "noise_prob": 0.1,
            },
            "diagnostics": {
                "enabled": True,
                "top_ks": [10, 50],
                "top_k_jaccard": 10,
            },
            "transfer": {"enabled": True},
        },
    }


def test_run_rl_diagnostics_and_transfer_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    output_dirs = _output_dirs(tmp_path)

    results = run_rl_diagnostics_and_transfer(_small_config(), output_dirs)

    assert results["experiment"] == "rl_diagnostics_transfer"
    assert set(results["representations"]) == {
        "raw_state_action",
        "local_observation_action",
        "situation_action",
        "situation_transition",
    }

    for representation in results["representations"].values():
        assert "global" in representation
        assert "per_layout" in representation
        assert "cross_layout_mean" in representation
        assert "cross_layout_std" in representation
        assert "prediction_transfer_mean" in representation
        assert "prediction_transfer_std" in representation
        assert "effective_vocab_size" in representation["global"]
        assert "token_coverage" in representation["cross_layout_mean"]
        assert "next_token_accuracy" in representation["prediction_transfer_mean"]

    expected_artifacts = [
        output_dirs["tables"] / "table_rl_diagnostics.tex",
        output_dirs["tables"] / "table_rl_diagnostics.csv",
        output_dirs["tables"] / "table_rl_transfer.tex",
        output_dirs["tables"] / "table_rl_transfer.csv",
        output_dirs["data"] / "rl_diagnostics_results.json",
        output_dirs["figures"] / "rl_cross_layout_coverage.pdf",
        output_dirs["figures"] / "rl_cross_layout_coverage.png",
        output_dirs["figures"] / "rl_jsd.pdf",
        output_dirs["figures"] / "rl_jsd.png",
        output_dirs["figures"] / "rl_next_token_accuracy.pdf",
        output_dirs["figures"] / "rl_next_token_accuracy.png",
    ]
    for artifact in expected_artifacts:
        assert artifact.exists()


def test_rl_abstractions_have_expected_cross_layout_reuse(tmp_path: Path) -> None:
    results = run_rl_diagnostics_and_transfer(_small_config(), _output_dirs(tmp_path))
    representations = results["representations"]

    raw_coverage = representations["raw_state_action"]["cross_layout_mean"]["token_coverage"]
    situation_coverage = representations["situation_action"]["cross_layout_mean"]["token_coverage"]
    local_coverage = representations["local_observation_action"]["cross_layout_mean"][
        "token_coverage"
    ]
    situation_accuracy = representations["situation_action"]["prediction_transfer_mean"][
        "next_token_accuracy"
    ]

    assert situation_coverage >= raw_coverage
    assert local_coverage >= raw_coverage
    assert 0 <= situation_accuracy <= 1
