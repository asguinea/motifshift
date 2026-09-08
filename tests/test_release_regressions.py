# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from motifshift.config import load_config, validate_config
from motifshift.metrics.sequences import group_token_sequences, sequence_transfer_summary
from motifshift.physical.field import field_records, generate_physical_field_environments
from motifshift.physical.predictors import (
    LocalPatchTokenPredictor,
    RawCellLookupPredictor,
    RegimeTokenPredictor,
    ValueChangeTokenPredictor,
)
from motifshift.rl.predictors import (
    MostFrequentNextTokenPredictor,
    evaluate_leave_one_layout_prediction,
)
from motifshift.run_all import main
from motifshift.verify_outputs import verify_outputs


def test_sequences_do_not_invent_boundary_bigrams() -> None:
    result = sequence_transfer_summary([["a", "b"], ["c", "d"]], [["b", "c"]])
    assert result["bigram_coverage"] == 0.0
    assert result["novel_recombination_rate"] == 1.0
    result = sequence_transfer_summary([["a", "b"]], [["a"], ["b"]])
    assert result["bigram_coverage"] == 0.0
    assert result["novel_recombination_rate"] == 0.0
    assert group_token_sequences(["a", "b", "c", "d"], [0, 1, 0, 1]) == [["a", "c"], ["b", "d"]]


def test_rl_predictor_excludes_episode_and_layout_boundaries() -> None:
    predictor = MostFrequentNextTokenPredictor().fit_sequences([["a", "b"], ["c", "d"]])
    assert predictor.predict_next("b") == "b"  # Global fallback, never the next episode's c.
    results = evaluate_leave_one_layout_prediction({"one": ["a", "b", "c", "d"], "two": ["b", "c"]})
    assert results["two"]["next_token_accuracy"] == 1.0  # Within one continuous sequence.
    results = evaluate_leave_one_layout_prediction(
        {"one": ["a", "b", "c", "d"], "two": ["b", "c"]},
        sequences_by_layout={"one": [["a", "b"], ["c", "d"]], "two": [["b"], ["c"]]},
    )
    assert results["two"]["next_token_accuracy"] == 0.0  # No valid test pairs.


@pytest.mark.parametrize(
    "predictor_class",
    [
        RawCellLookupPredictor,
        ValueChangeTokenPredictor,
        LocalPatchTokenPredictor,
        RegimeTokenPredictor,
    ],
)
def test_physical_prediction_does_not_read_future_values(predictor_class) -> None:
    environments = generate_physical_field_environments(
        n_environments=2, grid_size=6, timesteps=5, seed=7
    )
    train, test = [field_records(env) for env in environments]
    predictor = predictor_class().fit(train)
    original = predictor.predict_batch(test)
    corrupted = [{**record, "next_value": 1e6, "delta": -1e6} for record in test]
    assert predictor.predict_batch(corrupted) == original
    assert train[0]["prev_delta"] == 0.0
    assert train[36]["prev_delta"] == pytest.approx(train[0]["delta"])


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("experiment", "seed", True),
        ("experiment", "seed", -1),
        ("experiment", "seed", 2**32),
        ("rl", "enabled", "false"),
        ("robustness", "seeds", [1, 1]),
        ("robustness", "seeds", []),
        ("outputs", "data_dir", "../escape"),
        ("outputs", "tables_dir", "reports"),
        ("experiment", "typo", 1),
        ("runtime", "verbose", "yes"),
    ],
)
def test_invalid_config_fails_early(section, key, value) -> None:
    config = copy.deepcopy(load_config("configs/quick.yaml"))
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_manifest_detects_modified_output_and_escaping_paths(tmp_path: Path) -> None:
    assert main(["--config", "configs/smoke.yaml", "--outputs", str(tmp_path)]) == 0
    assert verify_outputs(tmp_path, strict=True)["status"] == "ok"
    report = tmp_path / "reports/run_summary.md"
    report.write_text(report.read_text() + "changed\n")
    assert verify_outputs(tmp_path, strict=True)["status"] == "invalid"
    manifest_path = tmp_path / "data/combined_results_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["expected_artifacts"].append("../outside.txt")
    manifest["artifacts"]["reports"].append("../outside.txt")
    manifest["sha256"]["../outside.txt"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    result = verify_outputs(tmp_path, strict=True)
    assert result["status"] == "invalid"
    assert any("escapes" in error for error in result["errors"])


def test_quick_cli_reproduces_numeric_results_across_hash_seeds(tmp_path: Path) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    for index, root in enumerate(roots):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "motifshift.run_all",
                "--config",
                "configs/quick.yaml",
                "--outputs",
                str(root),
            ],
            env={**os.environ, "PYTHONHASHSEED": str(index + 1)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert verify_outputs(root, strict=True)["status"] == "ok"
    names = [
        "spatiotemporal_diagnostics_results",
        "spatiotemporal_transfer_results",
        "rl_diagnostics_results",
        "physical_field_results",
    ]
    for name in names:
        assert (roots[0] / "data" / f"{name}.json").read_bytes() == (
            roots[1] / "data" / f"{name}.json"
        ).read_bytes()
