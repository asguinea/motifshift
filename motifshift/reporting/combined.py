# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Combined cross-domain reporting across diagnostic domains."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import motifshift
from motifshift.io import write_text
from motifshift.provenance import sha256_file
from motifshift.reporting.figures import save_bar_chart
from motifshift.reporting.tables import format_float, write_csv, write_latex_table

ST_REPRESENTATION_LABELS = {
    "absolute_grid": "Absolute grid",
    "relative_displacement": "Relative displacement",
    "motion": "Motion",
    "situation_motion": "Situation-motion",
}

RL_REPRESENTATION_LABELS = {
    "raw_state_action": "Raw state-action",
    "local_observation_action": "Local observation-action",
    "situation_action": "Situation-action",
    "situation_transition": "Situation-transition",
}

ST_PREDICTOR_LABELS = {
    "constant_velocity": "Constant velocity",
    "mean_displacement": "Mean displacement",
    "knn_raw": "Raw kNN",
    "motion_token": "Motion-token",
    "situation_motion_token": "Situation-motion",
}

PHYSICAL_REPRESENTATION_LABELS = {
    "raw_cell": "Raw cell",
    "value_change": "Value-change",
    "local_patch": "Local patch",
    "dynamical_regime": "Dynamical regime",
}

PHYSICAL_PREDICTOR_LABELS = {
    "mean_value": "Mean value",
    "raw_cell_lookup": "Raw cell lookup",
    "value_change_token": "Value-change token",
    "local_patch_token": "Local patch token",
    "dynamical_regime_token": "Dynamical regime token",
}


def _path(output_dirs: dict[str, Path], group: str, name: str) -> Path:
    return Path(output_dirs[group]) / name


def _diagnostic_rows(
    spatiotemporal_results: dict[str, Any] | None,
    rl_results: dict[str, Any] | None,
    physical_results: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if spatiotemporal_results:
        for name, result in spatiotemporal_results.get("representations", {}).items():
            global_summary = result["global"]
            cross_context = result["cross_scene_mean"]
            rows.append(
                {
                    "Domain": "Spatio-temporal",
                    "Representation": ST_REPRESENTATION_LABELS.get(name, name),
                    "Vocab": int(global_summary["vocab_size"]),
                    "Eff. Vocab / N": format_float(
                        global_summary["normalized_effective_vocab_size"]
                    ),
                    "Cross-Context Cov.": format_float(cross_context["token_coverage"]),
                    "JSD": format_float(cross_context["js_divergence"]),
                    "Bigram Cov.": format_float(cross_context["bigram_coverage"]),
                    "Novel Recomb.": format_float(cross_context["novel_recombination_rate"]),
                }
            )

    if rl_results:
        for name, result in rl_results.get("representations", {}).items():
            global_summary = result["global"]
            cross_context = result["cross_layout_mean"]
            rows.append(
                {
                    "Domain": "RL",
                    "Representation": RL_REPRESENTATION_LABELS.get(name, name),
                    "Vocab": int(global_summary["vocab_size"]),
                    "Eff. Vocab / N": format_float(
                        global_summary["normalized_effective_vocab_size"]
                    ),
                    "Cross-Context Cov.": format_float(cross_context["token_coverage"]),
                    "JSD": format_float(cross_context["js_divergence"]),
                    "Bigram Cov.": format_float(cross_context["bigram_coverage"]),
                    "Novel Recomb.": format_float(cross_context["novel_recombination_rate"]),
                }
            )

    if physical_results:
        for name, result in physical_results.get("representations", {}).items():
            global_summary = result["global"]
            cross_context = result["cross_environment_mean"]
            rows.append(
                {
                    "Domain": "Physical field",
                    "Representation": PHYSICAL_REPRESENTATION_LABELS.get(name, name),
                    "Vocab": int(global_summary["vocab_size"]),
                    "Eff. Vocab / N": format_float(
                        global_summary["normalized_effective_vocab_size"]
                    ),
                    "Cross-Context Cov.": format_float(cross_context["token_coverage"]),
                    "JSD": format_float(cross_context["js_divergence"]),
                    "Bigram Cov.": format_float(cross_context["bigram_coverage"]),
                    "Novel Recomb.": format_float(cross_context["novel_recombination_rate"]),
                }
            )

    if not rows:
        rows.append(
            {
                "Domain": "Synthetic diagnostic",
                "Representation": "No experiment results",
                "Vocab": "N/A",
                "Eff. Vocab / N": "N/A",
                "Cross-Context Cov.": "N/A",
                "JSD": "N/A",
                "Bigram Cov.": "N/A",
                "Novel Recomb.": "N/A",
            }
        )
    return rows


def write_combined_diagnostics_table(
    output_dirs: dict[str, Path],
    spatiotemporal_results: dict[str, Any] | None,
    rl_results: dict[str, Any] | None,
    physical_results: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write combined recurrence/compositionality diagnostics tables."""
    rows = _diagnostic_rows(spatiotemporal_results, rl_results, physical_results)
    columns = [
        "Domain",
        "Representation",
        "Vocab",
        "Eff. Vocab / N",
        "Cross-Context Cov.",
        "JSD",
        "Bigram Cov.",
        "Novel Recomb.",
    ]
    latex_path = write_latex_table(
        _path(output_dirs, "tables", "table_combined_diagnostics.tex"),
        caption="Combined synthetic diagnostic summary across domains.",
        label="tab:combined_diagnostics",
        columns=columns,
        rows=[[row[column] for column in columns] for row in rows],
    )
    csv_path = write_csv(_path(output_dirs, "tables", "table_combined_diagnostics.csv"), rows)
    return {
        "combined_diagnostics_latex": str(latex_path),
        "combined_diagnostics_csv": str(csv_path),
    }


def _transfer_rows(
    spatiotemporal_transfer_results: dict[str, Any] | None,
    rl_results: dict[str, Any] | None,
    physical_results: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if spatiotemporal_transfer_results:
        for name, result in spatiotemporal_transfer_results.get("predictors", {}).items():
            rows.append(
                {
                    "Domain": "Spatio-temporal",
                    "Method": ST_PREDICTOR_LABELS.get(name, name),
                    "Transfer Metric": "Mean error",
                    "Value": format_float(result["mean"]["mean_error"]),
                    "Cross-Context Cov.": "N/A",
                    "Novel Recomb.": "N/A",
                }
            )

    if rl_results:
        for name, result in rl_results.get("representations", {}).items():
            rows.append(
                {
                    "Domain": "RL",
                    "Method": RL_REPRESENTATION_LABELS.get(name, name),
                    "Transfer Metric": "Next-token acc.",
                    "Value": format_float(
                        result["prediction_transfer_mean"]["next_token_accuracy"]
                    ),
                    "Cross-Context Cov.": format_float(
                        result["cross_layout_mean"]["token_coverage"]
                    ),
                    "Novel Recomb.": format_float(
                        result["cross_layout_mean"]["novel_recombination_rate"]
                    ),
                }
            )

    if physical_results:
        for name, result in physical_results.get("predictors", {}).items():
            rows.append(
                {
                    "Domain": "Physical field",
                    "Method": PHYSICAL_PREDICTOR_LABELS.get(name, name),
                    "Transfer Metric": "Mean abs. error",
                    "Value": format_float(result["mean"]["mean_abs_error"]),
                    "Cross-Context Cov.": "N/A",
                    "Novel Recomb.": "N/A",
                }
            )

    if not rows:
        rows.append(
            {
                "Domain": "Synthetic diagnostic",
                "Method": "No transfer results",
                "Transfer Metric": "N/A",
                "Value": "N/A",
                "Cross-Context Cov.": "N/A",
                "Novel Recomb.": "N/A",
            }
        )
    return rows


def write_combined_transfer_table(
    output_dirs: dict[str, Path],
    spatiotemporal_transfer_results: dict[str, Any] | None,
    rl_results: dict[str, Any] | None,
    physical_results: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write combined transfer evidence tables."""
    rows = _transfer_rows(spatiotemporal_transfer_results, rl_results, physical_results)
    columns = [
        "Domain",
        "Method",
        "Transfer Metric",
        "Value",
        "Cross-Context Cov.",
        "Novel Recomb.",
    ]
    latex_path = write_latex_table(
        _path(output_dirs, "tables", "table_combined_transfer.tex"),
        caption=(
            "Combined transfer diagnostics. Metrics are domain-specific and "
            "intended as synthetic diagnostic evidence."
        ),
        label="tab:combined_transfer",
        columns=columns,
        rows=[[row[column] for column in columns] for row in rows],
    )
    csv_path = write_csv(_path(output_dirs, "tables", "table_combined_transfer.csv"), rows)
    return {"combined_transfer_latex": str(latex_path), "combined_transfer_csv": str(csv_path)}


def write_combined_cross_context_coverage_figure(
    output_dirs: dict[str, Path],
    spatiotemporal_results: dict[str, Any] | None,
    rl_results: dict[str, Any] | None,
    physical_results: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write a combined cross-context coverage bar chart when data is available."""
    labels: list[str] = []
    values: list[float] = []

    if spatiotemporal_results:
        for name, result in spatiotemporal_results.get("representations", {}).items():
            labels.append(f"ST {ST_REPRESENTATION_LABELS.get(name, name)}")
            values.append(float(result["cross_scene_mean"]["token_coverage"]))

    if rl_results:
        for name, result in rl_results.get("representations", {}).items():
            labels.append(f"RL {RL_REPRESENTATION_LABELS.get(name, name)}")
            values.append(float(result["cross_layout_mean"]["token_coverage"]))

    if physical_results:
        for name, result in physical_results.get("representations", {}).items():
            labels.append(f"PF {PHYSICAL_REPRESENTATION_LABELS.get(name, name)}")
            values.append(float(result["cross_environment_mean"]["token_coverage"]))

    if not values:
        return {}

    pdf_path, png_path = save_bar_chart(
        _path(output_dirs, "figures", "combined_cross_context_coverage.pdf"),
        _path(output_dirs, "figures", "combined_cross_context_coverage.png"),
        labels,
        values,
        ylabel="Token coverage",
        title="Combined cross-context token coverage",
    )
    return {
        "combined_cross_context_coverage_pdf": str(pdf_path),
        "combined_cross_context_coverage_png": str(png_path),
    }


def _existing(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if path.exists()]


def _infer_standard_artifacts(output_dirs: dict[str, Path]) -> dict[str, list[str]]:
    tables = Path(output_dirs["tables"])
    data = Path(output_dirs["data"])
    figures = Path(output_dirs["figures"])
    reports = Path(output_dirs["reports"])
    return {
        "spatiotemporal_diagnostics": _existing(
            [
                tables / "table_spatiotemporal_diagnostics.tex",
                tables / "table_spatiotemporal_diagnostics.csv",
                data / "spatiotemporal_diagnostics_results.json",
                figures / "spatiotemporal_cross_scene_coverage.pdf",
                figures / "spatiotemporal_cross_scene_coverage.png",
                figures / "spatiotemporal_jsd.pdf",
                figures / "spatiotemporal_jsd.png",
                figures / "spatiotemporal_token_coverage_topk.pdf",
                figures / "spatiotemporal_token_coverage_topk.png",
            ]
        ),
        "spatiotemporal_transfer": _existing(
            [
                tables / "table_spatiotemporal_transfer.tex",
                tables / "table_spatiotemporal_transfer.csv",
                data / "spatiotemporal_transfer_results.json",
                figures / "spatiotemporal_transfer_error.pdf",
                figures / "spatiotemporal_transfer_error.png",
                figures / "spatiotemporal_transfer_error_std.pdf",
                figures / "spatiotemporal_transfer_error_std.png",
            ]
        ),
        "rl_diagnostics_transfer": _existing(
            [
                tables / "table_rl_diagnostics.tex",
                tables / "table_rl_diagnostics.csv",
                tables / "table_rl_transfer.tex",
                tables / "table_rl_transfer.csv",
                data / "rl_diagnostics_results.json",
                figures / "rl_cross_layout_coverage.pdf",
                figures / "rl_cross_layout_coverage.png",
                figures / "rl_jsd.pdf",
                figures / "rl_jsd.png",
                figures / "rl_next_token_accuracy.pdf",
                figures / "rl_next_token_accuracy.png",
            ]
        ),
        "physical_field": _existing(
            [
                tables / "table_physical_field_diagnostics.tex",
                tables / "table_physical_field_diagnostics.csv",
                tables / "table_physical_field_transfer.tex",
                tables / "table_physical_field_transfer.csv",
                data / "physical_field_results.json",
                figures / "physical_field_cross_env_coverage.pdf",
                figures / "physical_field_cross_env_coverage.png",
                figures / "physical_field_jsd.pdf",
                figures / "physical_field_jsd.png",
                figures / "physical_field_transfer_error.pdf",
                figures / "physical_field_transfer_error.png",
            ]
        ),
        "robustness": _existing(
            [
                tables / "table_multiseed_diagnostics.tex",
                tables / "table_multiseed_diagnostics.csv",
                tables / "table_multiseed_transfer.tex",
                tables / "table_multiseed_transfer.csv",
                data / "multiseed_results.json",
                reports / "multiseed_summary.md",
                figures / "multiseed_cross_context_coverage.pdf",
                figures / "multiseed_cross_context_coverage.png",
            ]
        ),
        "combined": _existing(
            [
                tables / "table_combined_diagnostics.tex",
                tables / "table_combined_diagnostics.csv",
                tables / "table_combined_transfer.tex",
                tables / "table_combined_transfer.csv",
                data / "combined_results_manifest.json",
                figures / "combined_cross_context_coverage.pdf",
                figures / "combined_cross_context_coverage.png",
            ]
        ),
        "reports": _existing(
            [
                reports / "run_summary.md",
                reports / "results_summary.md",
            ]
        ),
    }


def write_artifact_manifest(
    output_dirs: dict[str, Path],
    artifacts: dict[str, Any],
) -> Path:
    """Write a machine-readable manifest of generated artifacts."""
    groups = _infer_standard_artifacts(output_dirs)
    for group in groups:
        if group in artifacts:
            values = artifacts[group]
            if isinstance(values, dict):
                groups[group] = list(values.values())
            elif isinstance(values, list):
                groups[group] = [str(value) for value in values]

    manifest_path = _path(output_dirs, "data", "combined_results_manifest.json")
    if str(manifest_path) not in groups["combined"]:
        groups["combined"].append(str(manifest_path))
    if groups.get("robustness"):
        groups["robustness_runs"] = [
            str(path)
            for path in (Path(output_dirs["root"]) / "robustness").rglob("*")
            if path.is_file()
        ]
    root = Path(output_dirs["root"]).resolve()

    def relative(value: str) -> str:
        return Path(value).resolve().relative_to(root).as_posix()

    groups = {key: sorted({relative(path) for path in values}) for key, values in groups.items()}
    expected_paths = sorted({path for values in groups.values() for path in values})
    manifest_relative = manifest_path.resolve().relative_to(root).as_posix()
    existing_paths = [p for p in expected_paths if (root / p).is_file() or p == manifest_relative]
    missing_paths = [p for p in expected_paths if p not in existing_paths]
    checksums = {p: sha256_file(root / p) for p in existing_paths if p != manifest_relative}

    manifest = {
        "schema_version": 1,
        "sha256": checksums,
        "provenance": artifacts.get("provenance", {}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_version": motifshift.__version__,
        "config_path": artifacts.get("config_path"),
        "artifacts": groups,
        "expected_artifacts": expected_paths,
        "existing_artifacts": existing_paths,
        "missing_artifacts": missing_paths,
        "note": "Synthetic diagnostic stress tests, not SOTA benchmarks.",
    }
    write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _best_representation(
    results: dict[str, Any] | None,
    domain: str,
    metric: str,
    lower_is_better: bool = False,
) -> str:
    if not results:
        return f"- {domain}: no diagnostic results available."

    rows: list[tuple[str, float]] = []
    for name, result in results.get("representations", {}).items():
        if "cross_scene_mean" in result:
            value = result["cross_scene_mean"][metric]
            label = ST_REPRESENTATION_LABELS.get(name, name)
        elif "cross_layout_mean" in result:
            value = result["cross_layout_mean"][metric]
            label = RL_REPRESENTATION_LABELS.get(name, name)
        else:
            value = result["cross_environment_mean"][metric]
            label = PHYSICAL_REPRESENTATION_LABELS.get(name, name)
        rows.append((label, float(value)))
    if not rows:
        return f"- {domain}: no diagnostic results available."

    best = (
        min(rows, key=lambda item: item[1])
        if lower_is_better
        else max(rows, key=lambda item: item[1])
    )
    return f"- {domain}: {best[0]} ({metric}={format_float(best[1])})."


def _best_spatiotemporal_transfer(
    results: dict[str, Any] | None,
) -> str:
    if not results:
        return "- Spatio-temporal transfer: no transfer results available."
    rows = [
        (ST_PREDICTOR_LABELS.get(name, name), float(result["mean"]["mean_error"]))
        for name, result in results.get("predictors", {}).items()
    ]
    if not rows:
        return "- Spatio-temporal transfer: no transfer results available."
    best = min(rows, key=lambda item: item[1])
    return f"- Spatio-temporal transfer: {best[0]} has the lowest mean error ({format_float(best[1])})."


def _best_rl_transfer(results: dict[str, Any] | None) -> str:
    if not results:
        return "- RL transfer: no transfer results available."
    rows = [
        (
            RL_REPRESENTATION_LABELS.get(name, name),
            float(result["prediction_transfer_mean"]["next_token_accuracy"]),
        )
        for name, result in results.get("representations", {}).items()
    ]
    if not rows:
        return "- RL transfer: no transfer results available."
    best = max(rows, key=lambda item: item[1])
    return (
        f"- RL transfer: {best[0]} has the highest next-token accuracy ({format_float(best[1])})."
    )


def _best_physical_transfer(results: dict[str, Any] | None) -> str:
    if not results:
        return "- Physical-field transfer: no transfer results available."
    rows = [
        (
            PHYSICAL_PREDICTOR_LABELS.get(name, name),
            float(result["mean"]["mean_abs_error"]),
        )
        for name, result in results.get("predictors", {}).items()
    ]
    if not rows:
        return "- Physical-field transfer: no transfer results available."
    best = min(rows, key=lambda item: item[1])
    return f"- Physical-field transfer: {best[0]} has the lowest mean absolute error ({format_float(best[1])})."


def write_results_summary(
    output_dirs: dict[str, Path],
    spatiotemporal_results: dict[str, Any] | None,
    spatiotemporal_transfer_results: dict[str, Any] | None,
    rl_results: dict[str, Any] | None,
    physical_results: dict[str, Any] | None = None,
) -> Path:
    """Write a human-readable markdown summary of generated artifacts."""
    tables = Path(output_dirs["tables"])
    data = Path(output_dirs["data"])
    figures = Path(output_dirs["figures"])
    summary_path = _path(output_dirs, "reports", "results_summary.md")

    table_paths = [
        tables / "table_spatiotemporal_diagnostics.tex",
        tables / "table_spatiotemporal_transfer.tex",
        tables / "table_rl_diagnostics.tex",
        tables / "table_rl_transfer.tex",
        tables / "table_physical_field_diagnostics.tex",
        tables / "table_physical_field_transfer.tex",
        tables / "table_combined_diagnostics.tex",
        tables / "table_combined_transfer.tex",
    ]
    figure_paths = [
        figures / "spatiotemporal_cross_scene_coverage.pdf",
        figures / "spatiotemporal_transfer_error.pdf",
        figures / "rl_cross_layout_coverage.pdf",
        figures / "rl_next_token_accuracy.pdf",
        figures / "physical_field_cross_env_coverage.pdf",
        figures / "physical_field_transfer_error.pdf",
        figures / "combined_cross_context_coverage.pdf",
        data / "combined_results_manifest.json",
    ]

    lines = [
        "# Results Summary",
        "",
        "## Scope",
        "",
        "This run contains lightweight synthetic diagnostic evidence. Synthetic diagnostic results are intended to test representation diagnostics, not SOTA performance.",
        "",
        "## Generated Tables",
        "",
    ]
    lines.extend(
        f"- {path.relative_to(output_dirs['root'])}" for path in table_paths if path.exists()
    )
    lines.extend(["", "## Generated Figures", ""])
    lines.extend(
        f"- {path.relative_to(output_dirs['root'])}" for path in figure_paths[:-1] if path.exists()
    )
    lines.extend(["", "## Generated Data", ""])
    lines.extend(
        f"- {path.relative_to(output_dirs['root'])}" for path in figure_paths[-1:] if path.exists()
    )
    lines.extend(
        [
            "",
            "## Key Quantities to Inspect",
            "",
            "In this synthetic diagnostic setting, useful quantities include:",
            _best_representation(
                spatiotemporal_results,
                "Spatio-temporal highest coverage",
                "token_coverage",
            ),
            _best_representation(
                spatiotemporal_results,
                "Spatio-temporal lowest JSD",
                "js_divergence",
                lower_is_better=True,
            ),
            _best_representation(
                spatiotemporal_results,
                "Spatio-temporal novel recombination",
                "novel_recombination_rate",
            ),
            _best_representation(rl_results, "RL highest coverage", "token_coverage"),
            _best_representation(
                rl_results,
                "RL lowest JSD",
                "js_divergence",
                lower_is_better=True,
            ),
            _best_representation(
                rl_results,
                "RL novel recombination",
                "novel_recombination_rate",
            ),
            _best_spatiotemporal_transfer(spatiotemporal_transfer_results),
            _best_rl_transfer(rl_results),
            _best_representation(
                physical_results,
                "Physical-field highest coverage",
                "token_coverage",
            ),
            _best_representation(
                physical_results,
                "Physical-field lowest JSD",
                "js_divergence",
                lower_is_better=True,
            ),
            _best_representation(
                physical_results,
                "Physical-field novel recombination",
                "novel_recombination_rate",
            ),
            _best_physical_transfer(physical_results),
            "",
            "## Interpretation Notes",
            "",
            "- Compare coverage with prediction error: coarse tokens can increase coverage while losing useful information.",
            "- Layout identifiers in raw RL tokens force zero overlap for held-out layouts by construction.",
            "- Novel recombination metrics distinguish reuse of units from memorization of full transitions.",
            "- Spatio-temporal transfer predicts future displacement from current and previous motion context.",
            "- Constant velocity is a baseline and is not trivially perfect under the non-uniform synthetic dynamics.",
            "- The spatio-temporal generator uses reusable but non-identical motifs across scenes.",
            "- Physical-field experiments are synthetic advection/diffusion-like stress tests.",
            "- Physical systems can contain recurring local dynamical motifs when represented through local regimes.",
            "- Transfer results are diagnostic rather than SOTA benchmarks.",
            "- These results are intended as evidence of the diagnostic methodology, not task-leading performance.",
            "",
        ]
    )

    write_text(summary_path, "\n".join(lines))
    return summary_path
