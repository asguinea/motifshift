# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Multi-seed robustness reporting for synthetic diagnostics."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from motifshift.io import write_text
from motifshift.reporting.figures import save_bar_chart
from motifshift.reporting.tables import format_float, write_csv, write_latex_table

DIAGNOSTIC_METRICS = (
    "token_coverage",
    "js_divergence",
    "bigram_coverage",
    "novel_recombination_rate",
)

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

PHYSICAL_REPRESENTATION_LABELS = {
    "raw_cell": "Raw cell",
    "value_change": "Value-change",
    "local_patch": "Local patch",
    "dynamical_regime": "Dynamical regime",
}

ST_PREDICTOR_LABELS = {
    "constant_velocity": "Constant velocity",
    "mean_displacement": "Mean displacement",
    "knn_raw": "Raw kNN",
    "motion_token": "Motion-token",
    "situation_motion_token": "Situation-motion",
}

PHYSICAL_PREDICTOR_LABELS = {
    "mean_value": "Mean value",
    "raw_cell_lookup": "Raw cell lookup",
    "value_change_token": "Value-change token",
    "local_patch_token": "Local patch token",
    "dynamical_regime_token": "Dynamical regime token",
}


def _seed_output_dirs(output_dirs: dict[str, Path], seed: int) -> dict[str, Path]:
    root = Path(output_dirs["root"]) / "robustness" / f"seed_{seed}"
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


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "std": float(np.std(array))}


def _empty_metric_store() -> dict[str, list[float]]:
    return {metric: [] for metric in DIAGNOSTIC_METRICS}


def _add_diagnostics(
    store: dict[str, dict[str, dict[str, list[float]]]],
    domain: str,
    representation: str,
    metrics: dict[str, Any],
) -> None:
    representation_store = store.setdefault(domain, {}).setdefault(
        representation, _empty_metric_store()
    )
    for metric in DIAGNOSTIC_METRICS:
        representation_store[metric].append(float(metrics[metric]))


def _add_transfer(
    store: dict[str, dict[str, dict[str, list[float]]]],
    domain: str,
    method: str,
    metric: str,
    value: float,
) -> None:
    store.setdefault(domain, {}).setdefault(method, {}).setdefault(metric, []).append(float(value))


def _aggregate_nested(
    values: dict[str, dict[str, dict[str, list[float]]]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    return {
        domain: {
            name: {
                metric: _mean_std(metric_values)
                for metric, metric_values in metrics.items()
                if metric_values
            }
            for name, metrics in names.items()
        }
        for domain, names in values.items()
    }


def _domain_label(domain: str) -> str:
    return {
        "spatiotemporal": "Spatio-temporal",
        "rl": "RL",
        "physical_field": "Physical field",
    }.get(domain, domain)


def _representation_label(domain: str, name: str) -> str:
    if domain == "spatiotemporal":
        return ST_REPRESENTATION_LABELS.get(name, name)
    if domain == "rl":
        return RL_REPRESENTATION_LABELS.get(name, name)
    if domain == "physical_field":
        return PHYSICAL_REPRESENTATION_LABELS.get(name, name)
    return name


def _method_label(domain: str, name: str) -> str:
    if domain == "spatiotemporal":
        return ST_PREDICTOR_LABELS.get(name, name)
    if domain == "rl":
        return RL_REPRESENTATION_LABELS.get(name, name)
    if domain == "physical_field":
        return PHYSICAL_PREDICTOR_LABELS.get(name, name)
    return name


def _pm(metric: dict[str, float]) -> str:
    return f"{format_float(metric['mean'])} $\\pm$ {format_float(metric['std'])}"


def _write_diagnostics_tables(
    output_dirs: dict[str, Path],
    diagnostics: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> dict[str, str]:
    csv_rows: list[dict[str, Any]] = []
    latex_rows: list[list[Any]] = []
    for domain, representations in diagnostics.items():
        for name, metrics in representations.items():
            csv_rows.append(
                {
                    "domain": _domain_label(domain),
                    "representation": _representation_label(domain, name),
                    "coverage_mean": metrics["token_coverage"]["mean"],
                    "coverage_std": metrics["token_coverage"]["std"],
                    "jsd_mean": metrics["js_divergence"]["mean"],
                    "jsd_std": metrics["js_divergence"]["std"],
                    "bigram_coverage_mean": metrics["bigram_coverage"]["mean"],
                    "bigram_coverage_std": metrics["bigram_coverage"]["std"],
                    "novel_recombination_mean": metrics["novel_recombination_rate"]["mean"],
                    "novel_recombination_std": metrics["novel_recombination_rate"]["std"],
                }
            )
            latex_rows.append(
                [
                    _domain_label(domain),
                    _representation_label(domain, name),
                    _pm(metrics["token_coverage"]),
                    _pm(metrics["js_divergence"]),
                    _pm(metrics["bigram_coverage"]),
                    _pm(metrics["novel_recombination_rate"]),
                ]
            )

    if not csv_rows:
        csv_rows.append(
            {
                "domain": "Synthetic diagnostic",
                "representation": "No enabled domains",
                "coverage_mean": 0.0,
                "coverage_std": 0.0,
                "jsd_mean": 0.0,
                "jsd_std": 0.0,
                "bigram_coverage_mean": 0.0,
                "bigram_coverage_std": 0.0,
                "novel_recombination_mean": 0.0,
                "novel_recombination_std": 0.0,
            }
        )
        latex_rows.append(
            ["Synthetic diagnostic", "No enabled domains", "N/A", "N/A", "N/A", "N/A"]
        )

    latex_path = write_latex_table(
        output_dirs["tables"] / "table_multiseed_diagnostics.tex",
        caption="Multi-seed diagnostic robustness across synthetic domains.",
        label="tab:multiseed_diagnostics",
        columns=["Domain", "Rep.", "Cov.", "JSD", "BiCov.", "NovRec."],
        rows=latex_rows,
    )
    csv_path = write_csv(output_dirs["tables"] / "table_multiseed_diagnostics.csv", csv_rows)
    return {
        "multiseed_diagnostics_latex": str(latex_path),
        "multiseed_diagnostics_csv": str(csv_path),
    }


def _write_transfer_tables(
    output_dirs: dict[str, Path],
    transfer: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> dict[str, str]:
    csv_rows: list[dict[str, Any]] = []
    latex_rows: list[list[Any]] = []
    metric_labels = {
        "mean_error": "Mean displacement error",
        "next_token_accuracy": "Next-token accuracy",
        "mean_abs_error": "Mean absolute error",
    }
    for domain, methods in transfer.items():
        for name, metrics in methods.items():
            for metric, summary in metrics.items():
                csv_rows.append(
                    {
                        "domain": _domain_label(domain),
                        "method": _method_label(domain, name),
                        "metric": metric_labels.get(metric, metric),
                        "value_mean": summary["mean"],
                        "value_std": summary["std"],
                    }
                )
                latex_rows.append(
                    [
                        _domain_label(domain),
                        _method_label(domain, name),
                        metric_labels.get(metric, metric),
                        _pm(summary),
                    ]
                )

    if not csv_rows:
        csv_rows.append(
            {
                "domain": "Synthetic diagnostic",
                "method": "No enabled domains",
                "metric": "N/A",
                "value_mean": 0.0,
                "value_std": 0.0,
            }
        )
        latex_rows.append(["Synthetic diagnostic", "No enabled domains", "N/A", "N/A"])

    latex_path = write_latex_table(
        output_dirs["tables"] / "table_multiseed_transfer.tex",
        caption="Multi-seed transfer robustness across synthetic domains.",
        label="tab:multiseed_transfer",
        columns=["Domain", "Method", "Metric", "Value"],
        rows=latex_rows,
    )
    csv_path = write_csv(output_dirs["tables"] / "table_multiseed_transfer.csv", csv_rows)
    return {
        "multiseed_transfer_latex": str(latex_path),
        "multiseed_transfer_csv": str(csv_path),
    }


def _write_coverage_figure(
    output_dirs: dict[str, Path],
    diagnostics: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> dict[str, str]:
    labels: list[str] = []
    values: list[float] = []
    for domain, representations in diagnostics.items():
        for name, metrics in representations.items():
            labels.append(f"{_domain_label(domain)} {_representation_label(domain, name)}")
            values.append(float(metrics["token_coverage"]["mean"]))

    if not values:
        return {}

    pdf_path, png_path = save_bar_chart(
        output_dirs["figures"] / "multiseed_cross_context_coverage.pdf",
        output_dirs["figures"] / "multiseed_cross_context_coverage.png",
        labels,
        values,
        ylabel="Mean token coverage",
        title="Multi-seed cross-context token coverage",
    )
    return {
        "multiseed_cross_context_coverage_pdf": str(pdf_path),
        "multiseed_cross_context_coverage_png": str(png_path),
    }


def _best_coverage_note(
    diagnostics: dict[str, dict[str, dict[str, dict[str, float]]]], domain: str
) -> str:
    representations = diagnostics.get(domain, {})
    if not representations:
        return f"- {_domain_label(domain)}: not included."
    best = max(
        representations.items(),
        key=lambda item: item[1]["token_coverage"]["mean"],
    )
    lowest_jsd = min(
        representations.items(),
        key=lambda item: item[1]["js_divergence"]["mean"],
    )
    return (
        f"- {_domain_label(domain)}: highest mean coverage is "
        f"{_representation_label(domain, best[0])} "
        f"({format_float(best[1]['token_coverage']['mean'])}); lowest mean JSD is "
        f"{_representation_label(domain, lowest_jsd[0])} "
        f"({format_float(lowest_jsd[1]['js_divergence']['mean'])})."
    )


def _write_summary(
    output_dirs: dict[str, Path],
    seeds: list[int],
    diagnostics: dict[str, dict[str, dict[str, dict[str, float]]]],
    transfer: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> Path:
    domains = [_domain_label(domain) for domain in diagnostics]
    lines = [
        "# Multi-Seed Robustness Summary",
        "",
        "## Scope",
        "",
        "This report aggregates the synthetic diagnostic and transfer experiments across multiple random seeds. It is intended as robustness evidence for qualitative patterns, not as a SOTA benchmark.",
        "",
        "## Seeds",
        "",
        ", ".join(str(seed) for seed in seeds),
        "",
        "## Domains Included",
        "",
    ]
    lines.extend(f"- {domain}" for domain in domains)
    if not domains:
        lines.append("- No enabled diagnostic domains.")

    lines.extend(
        [
            "",
            "## Diagnostic Stability",
            "",
            _best_coverage_note(diagnostics, "spatiotemporal"),
            _best_coverage_note(diagnostics, "rl"),
            _best_coverage_note(diagnostics, "physical_field"),
            "",
            "## Transfer Stability",
            "",
        ]
    )
    if transfer:
        for domain, methods in transfer.items():
            for name, metrics in methods.items():
                metric, summary = next(iter(metrics.items()))
                lines.append(
                    f"- {_domain_label(domain)} {_method_label(domain, name)}: "
                    f"{metric}={format_float(summary['mean'])} +/- "
                    f"{format_float(summary['std'])}."
                )
    else:
        lines.append("- No transfer results were available.")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Mean/std values summarize sensitivity to synthetic random seeds.",
            "- Standard deviations use ddof=0 across seed-level means; they are not confidence intervals.",
            "- Large standard deviations should be interpreted as a sign that a synthetic setting is seed-sensitive.",
            "- These are synthetic stress tests and do not replace full downstream benchmarks.",
            "",
        ]
    )

    path = output_dirs["reports"] / "multiseed_summary.md"
    write_text(path, "\n".join(lines))
    return path


def _run_seed(config: dict[str, Any], output_dirs: dict[str, Path], seed: int) -> dict[str, Any]:
    seed_config = copy.deepcopy(config)
    seed_config.setdefault("experiment", {})["seed"] = int(seed)
    seed_dirs = _seed_output_dirs(output_dirs, seed)

    results: dict[str, Any] = {"output_root": str(seed_dirs["root"])}

    spatiotemporal = seed_config.get("spatiotemporal", {})
    if spatiotemporal.get("enabled", False):
        if spatiotemporal.get("diagnostics", {}).get("enabled", False):
            from motifshift.spatiotemporal.experiment import (
                run_spatiotemporal_diagnostics,
            )

            results["spatiotemporal_diagnostics"] = run_spatiotemporal_diagnostics(
                seed_config, seed_dirs
            )
        if spatiotemporal.get("transfer", {}).get("enabled", False):
            from motifshift.spatiotemporal.experiment import (
                run_spatiotemporal_transfer,
            )

            results["spatiotemporal_transfer"] = run_spatiotemporal_transfer(seed_config, seed_dirs)

    rl = seed_config.get("rl", {})
    if rl.get("enabled", False) and (
        rl.get("diagnostics", {}).get("enabled", False)
        or rl.get("transfer", {}).get("enabled", False)
    ):
        from motifshift.rl.experiment import run_rl_diagnostics_and_transfer

        results["rl"] = run_rl_diagnostics_and_transfer(seed_config, seed_dirs)

    physical = seed_config.get("physical_field", {})
    if physical.get("enabled", False) and (
        physical.get("diagnostics", {}).get("enabled", False)
        or physical.get("transfer", {}).get("enabled", False)
    ):
        from motifshift.physical.experiment import (
            run_physical_field_diagnostics_and_transfer,
        )

        results["physical_field"] = run_physical_field_diagnostics_and_transfer(
            seed_config, seed_dirs
        )

    return results


def _collect_seed_results(
    seed_results: dict[str, Any],
    diagnostics_store: dict[str, dict[str, dict[str, list[float]]]],
    transfer_store: dict[str, dict[str, dict[str, list[float]]]],
) -> None:
    st_results = seed_results.get("spatiotemporal_diagnostics")
    if st_results:
        for name, result in st_results.get("representations", {}).items():
            _add_diagnostics(
                diagnostics_store,
                "spatiotemporal",
                name,
                result["cross_scene_mean"],
            )

    st_transfer = seed_results.get("spatiotemporal_transfer")
    if st_transfer:
        for name, result in st_transfer.get("predictors", {}).items():
            _add_transfer(
                transfer_store,
                "spatiotemporal",
                name,
                "mean_error",
                result["mean"]["mean_error"],
            )

    rl_results = seed_results.get("rl")
    if rl_results:
        for name, result in rl_results.get("representations", {}).items():
            _add_diagnostics(
                diagnostics_store,
                "rl",
                name,
                result["cross_layout_mean"],
            )
            _add_transfer(
                transfer_store,
                "rl",
                name,
                "next_token_accuracy",
                result["prediction_transfer_mean"]["next_token_accuracy"],
            )

    physical_results = seed_results.get("physical_field")
    if physical_results:
        for name, result in physical_results.get("representations", {}).items():
            _add_diagnostics(
                diagnostics_store,
                "physical_field",
                name,
                result["cross_environment_mean"],
            )
        for name, result in physical_results.get("predictors", {}).items():
            _add_transfer(
                transfer_store,
                "physical_field",
                name,
                "mean_abs_error",
                result["mean"]["mean_abs_error"],
            )


def run_multiseed_robustness(
    config: dict[str, Any], output_dirs: dict[str, Path]
) -> dict[str, Any]:
    """Run enabled synthetic experiments across multiple seeds and aggregate metrics."""
    robustness = config.get("robustness", {})
    seeds = [int(seed) for seed in robustness.get("seeds", [123])]
    if len(set(seeds)) != len(seeds):
        raise ValueError("robustness.seeds must be unique")
    if not seeds:
        raise ValueError("robustness.seeds must contain at least one seed")

    diagnostics_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    transfer_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    seed_output_roots: dict[str, str] = {}

    for seed in seeds:
        seed_results = _run_seed(config, output_dirs, seed)
        seed_output_roots[str(seed)] = (
            Path(seed_results["output_root"]).relative_to(output_dirs["root"]).as_posix()
        )
        _collect_seed_results(seed_results, diagnostics_values, transfer_values)

    diagnostics = _aggregate_nested(diagnostics_values)
    transfer = _aggregate_nested(transfer_values)

    results: dict[str, Any] = {
        "experiment": "multiseed_robustness",
        "seeds": seeds,
        "diagnostics": diagnostics,
        "transfer": transfer,
        "seed_output_roots": seed_output_roots,
    }

    artifacts: dict[str, str] = {}
    artifacts.update(_write_diagnostics_tables(output_dirs, diagnostics))
    artifacts.update(_write_transfer_tables(output_dirs, transfer))
    artifacts.update(_write_coverage_figure(output_dirs, diagnostics))
    summary_path = _write_summary(output_dirs, seeds, diagnostics, transfer)
    artifacts["multiseed_summary"] = str(summary_path)

    json_path = output_dirs["data"] / "multiseed_results.json"
    results["artifacts"] = artifacts | {"multiseed_results_json": str(json_path)}
    portable = {
        **results,
        "artifacts": {
            key: Path(value).relative_to(output_dirs["root"]).as_posix()
            for key, value in results["artifacts"].items()
        },
    }
    write_text(json_path, json.dumps(portable, indent=2, sort_keys=True) + "\n")
    return results
