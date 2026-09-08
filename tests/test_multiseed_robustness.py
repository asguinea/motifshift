# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from motifshift.reporting.robustness import run_multiseed_robustness


def _output_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "reports": root / "reports",
        "tables": root / "tables",
        "figures": root / "figures",
        "data": root / "data",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _tiny_config() -> dict[str, Any]:
    return {
        "experiment": {"name": "test", "seed": 123, "smoke": False},
        "runtime": {"verbose": False},
        "robustness": {"enabled": True, "seeds": [123, 456]},
        "spatiotemporal": {
            "enabled": True,
            "synthetic": {
                "n_scenes": 2,
                "agents_per_scene": 4,
                "timesteps": 6,
                "noise_std": 0.02,
                "speed_variation": 0.2,
                "scene_variation": 0.25,
            },
            "diagnostics": {"enabled": True, "top_ks": [5, 10], "top_k_jaccard": 5},
            "transfer": {"enabled": True, "horizon": 1, "k_neighbors": 2},
        },
        "rl": {
            "enabled": True,
            "synthetic": {
                "n_layouts": 2,
                "width": 6,
                "height": 6,
                "episodes_per_world": 3,
                "max_steps": 15,
                "noise_prob": 0.1,
            },
            "diagnostics": {"enabled": True, "top_ks": [5, 10], "top_k_jaccard": 5},
            "transfer": {"enabled": True},
        },
        "physical_field": {
            "enabled": True,
            "synthetic": {
                "n_environments": 2,
                "grid_size": 8,
                "timesteps": 5,
                "noise_std": 0.01,
            },
            "diagnostics": {"enabled": True, "top_ks": [5, 10], "top_k_jaccard": 5},
            "transfer": {"enabled": True},
        },
    }


def _assert_summary_finite(summary: dict[str, float]) -> None:
    assert math.isfinite(summary["mean"])
    assert math.isfinite(summary["std"])
    assert summary["std"] >= 0.0


def test_run_multiseed_robustness_writes_expected_artifacts(tmp_path: Path) -> None:
    output_dirs = _output_dirs(tmp_path / "outputs")

    results = run_multiseed_robustness(_tiny_config(), output_dirs)

    assert results["experiment"] == "multiseed_robustness"
    assert results["seeds"] == [123, 456]
    assert (output_dirs["data"] / "multiseed_results.json").exists()
    assert (output_dirs["tables"] / "table_multiseed_diagnostics.tex").exists()
    assert (output_dirs["tables"] / "table_multiseed_diagnostics.csv").exists()
    assert (output_dirs["tables"] / "table_multiseed_transfer.tex").exists()
    assert (output_dirs["tables"] / "table_multiseed_transfer.csv").exists()
    assert (output_dirs["reports"] / "multiseed_summary.md").exists()

    assert "spatiotemporal" in results["diagnostics"]
    assert "rl" in results["diagnostics"]
    assert "physical_field" in results["diagnostics"]
    _assert_summary_finite(results["diagnostics"]["spatiotemporal"]["motion"]["token_coverage"])
    _assert_summary_finite(results["diagnostics"]["rl"]["situation_action"]["js_divergence"])
    _assert_summary_finite(
        results["diagnostics"]["physical_field"]["value_change"]["bigram_coverage"]
    )
    _assert_summary_finite(results["transfer"]["spatiotemporal"]["motion_token"]["mean_error"])
    _assert_summary_finite(results["transfer"]["rl"]["situation_action"]["next_token_accuracy"])
    _assert_summary_finite(
        results["transfer"]["physical_field"]["value_change_token"]["mean_abs_error"]
    )

    assert (output_dirs["root"] / "robustness" / "seed_123").exists()
    assert not (output_dirs["data"] / "spatiotemporal_diagnostics_results.json").exists()


def test_run_multiseed_robustness_handles_disabled_domain(tmp_path: Path) -> None:
    config = deepcopy(_tiny_config())
    config["rl"]["enabled"] = False
    output_dirs = _output_dirs(tmp_path / "outputs")

    results = run_multiseed_robustness(config, output_dirs)

    assert "spatiotemporal" in results["diagnostics"]
    assert "physical_field" in results["diagnostics"]
    assert "rl" not in results["diagnostics"]
    assert "rl" not in results["transfer"]
