# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point for the evaluation suite."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from motifshift.config import load_config
from motifshift.io import ensure_output_dirs
from motifshift.provenance import capture_provenance
from motifshift.reporting.summary import write_run_summary
from motifshift.seed import set_global_seed


def _is_verbose(config: dict[str, Any]) -> bool:
    return bool(config.get("runtime", {}).get("verbose", False))


def _experiment(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("experiment", {})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MotifShift synthetic diagnostics suite.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and report config information without writing outputs.",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run optional multi-seed robustness aggregation after the regular run.",
    )
    parser.add_argument(
        "--outputs", type=Path, help="Override the output root; use a fresh directory for each run."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.outputs is not None:
        config.setdefault("outputs", {})["root"] = str(args.outputs)
    if args.robustness:
        config.setdefault("robustness", {})["enabled"] = True
    experiment = _experiment(config)

    if args.dry_run:
        print("Dry run")
        print(f"Config: {config_path.name}")
        print(f"Experiment: {experiment.get('name', 'unknown')}")
        return 0

    verbose = _is_verbose(config)
    if verbose:
        print(f"Loading config: {config_path.name}")

    seed = int(experiment.get("seed", 0))
    set_global_seed(seed)
    if verbose:
        print(f"Seed set: {seed}")

    output_root = Path(config.get("outputs", {}).get("root", "outputs"))
    if output_root.exists() and any(
        p.is_file() and p.name not in {".gitkeep", ".DS_Store"} for p in output_root.rglob("*")
    ):
        parser.error("output directory already contains files; select a fresh --outputs directory")
    output_dirs = ensure_output_dirs(config)
    provenance = capture_provenance(config, output_dirs)
    if verbose:
        print(f"Output root: {output_dirs['root']}")

    artifact_paths: list[str] = []
    artifact_groups: dict[str, list[str]] = {
        "spatiotemporal_diagnostics": [],
        "spatiotemporal_transfer": [],
        "rl_diagnostics_transfer": [],
        "physical_field": [],
        "robustness": [],
        "combined": [],
        "reports": [],
    }
    spatiotemporal_results: dict[str, Any] | None = None
    spatiotemporal_transfer_results: dict[str, Any] | None = None
    rl_results: dict[str, Any] | None = None
    physical_results: dict[str, Any] | None = None

    spatiotemporal = config.get("spatiotemporal", {})
    diagnostics = spatiotemporal.get("diagnostics", {})
    spatiotemporal_transfer = spatiotemporal.get("transfer", {})
    if spatiotemporal.get("enabled", False):
        if diagnostics.get("enabled", False):
            if verbose:
                print("Running spatio-temporal diagnostics")
            from motifshift.spatiotemporal.experiment import (
                run_spatiotemporal_diagnostics,
            )

            results = run_spatiotemporal_diagnostics(config, output_dirs)
            spatiotemporal_results = results
            artifact_paths.extend(results.get("artifacts", {}).values())
            artifact_groups["spatiotemporal_diagnostics"] = list(
                results.get("artifacts", {}).values()
            )
            if verbose:
                print("Spatio-temporal diagnostics complete")

        if spatiotemporal_transfer.get("enabled", False):
            if verbose:
                print("Running spatio-temporal transfer")
            from motifshift.spatiotemporal.experiment import (
                run_spatiotemporal_transfer,
            )

            results = run_spatiotemporal_transfer(config, output_dirs)
            spatiotemporal_transfer_results = results
            artifact_paths.extend(results.get("artifacts", {}).values())
            artifact_groups["spatiotemporal_transfer"] = list(results.get("artifacts", {}).values())
            if verbose:
                print("Spatio-temporal transfer complete")

    rl = config.get("rl", {})
    rl_diagnostics = rl.get("diagnostics", {})
    rl_transfer = rl.get("transfer", {})
    if rl.get("enabled", False) and (
        rl_diagnostics.get("enabled", False) or rl_transfer.get("enabled", False)
    ):
        if verbose:
            print("Running RL diagnostics and transfer")
        from motifshift.rl.experiment import run_rl_diagnostics_and_transfer

        results = run_rl_diagnostics_and_transfer(config, output_dirs)
        rl_results = results
        artifact_paths.extend(results.get("artifacts", {}).values())
        artifact_groups["rl_diagnostics_transfer"] = list(results.get("artifacts", {}).values())
        if verbose:
            print("RL diagnostics and transfer complete")

    physical_field = config.get("physical_field", {})
    physical_diagnostics = physical_field.get("diagnostics", {})
    physical_transfer = physical_field.get("transfer", {})
    if physical_field.get("enabled", False) and (
        physical_diagnostics.get("enabled", False) or physical_transfer.get("enabled", False)
    ):
        if verbose:
            print("Running physical-field diagnostics and transfer")
        from motifshift.physical.experiment import (
            run_physical_field_diagnostics_and_transfer,
        )

        results = run_physical_field_diagnostics_and_transfer(config, output_dirs)
        physical_results = results
        artifact_paths.extend(results.get("artifacts", {}).values())
        artifact_groups["physical_field"] = list(results.get("artifacts", {}).values())
        if verbose:
            print("Physical-field diagnostics and transfer complete")

    robustness = config.get("robustness", {})
    run_robustness = bool(args.robustness or robustness.get("enabled", False))
    robustness_results: dict[str, Any] | None = None
    if run_robustness:
        if verbose:
            print("Running multi-seed robustness aggregation")
        from motifshift.reporting.robustness import run_multiseed_robustness

        robustness_results = run_multiseed_robustness(config, output_dirs)
        robustness_artifacts = robustness_results.get("artifacts", {})
        artifact_paths.extend(robustness_artifacts.values())
        artifact_groups["robustness"] = list(robustness_artifacts.values())
        if verbose:
            print("Multi-seed robustness aggregation complete")

    if verbose:
        print("Writing combined reporting artifacts")
    from motifshift.reporting.combined import (
        write_artifact_manifest,
        write_combined_cross_context_coverage_figure,
        write_combined_diagnostics_table,
        write_combined_transfer_table,
        write_results_summary,
    )

    combined_artifacts: dict[str, str] = {}
    combined_artifacts.update(
        write_combined_diagnostics_table(
            output_dirs, spatiotemporal_results, rl_results, physical_results
        )
    )
    combined_artifacts.update(
        write_combined_transfer_table(
            output_dirs,
            spatiotemporal_transfer_results,
            rl_results,
            physical_results,
        )
    )
    combined_artifacts.update(
        write_combined_cross_context_coverage_figure(
            output_dirs, spatiotemporal_results, rl_results, physical_results
        )
    )
    results_summary_path = write_results_summary(
        output_dirs,
        spatiotemporal_results,
        spatiotemporal_transfer_results,
        rl_results,
        physical_results,
    )
    artifact_groups["reports"].append(str(results_summary_path))
    artifact_groups["reports"].append(str(output_dirs["reports"] / "run_summary.md"))
    artifact_groups["combined"] = list(combined_artifacts.values())
    artifact_paths.extend(combined_artifacts.values())
    artifact_paths.append(str(results_summary_path))
    config_snapshot = output_dirs["data"] / "config.json"
    artifact_groups["reports"].append(str(config_snapshot))
    artifact_paths.append(str(config_snapshot))
    summary_path = write_run_summary(config, output_dirs, artifacts=artifact_paths)
    write_artifact_manifest(
        output_dirs,
        {**artifact_groups, "config_path": config_path.name, "provenance": provenance},
    )
    if verbose:
        print("Combined reporting artifacts complete")

    if verbose:
        print(f"Run summary: {summary_path}")
        print("Done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
