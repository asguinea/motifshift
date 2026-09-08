# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Physical-field diagnostics and transfer experiment."""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from motifshift.io import write_text
from motifshift.metrics import diagnostic_summary
from motifshift.metrics.sequences import group_token_sequences, sequence_transfer_summary
from motifshift.physical.field import (
    FieldEnvironment,
    field_records,
    generate_physical_field_environments,
)
from motifshift.physical.predictors import (
    LocalPatchTokenPredictor,
    MeanValuePredictor,
    RawCellLookupPredictor,
    RegimeTokenPredictor,
    ValueChangeTokenPredictor,
    aggregate_prediction_results,
    evaluate_leave_one_environment_prediction,
)
from motifshift.physical.representations import (
    dynamical_regime_tokens,
    local_patch_tokens,
    raw_cell_tokens,
    value_change_tokens,
)
from motifshift.reporting.figures import save_bar_chart
from motifshift.reporting.tables import format_float, write_csv, write_latex_table

Record = Mapping[str, Any]
Tokenizer = Callable[[Sequence[Record]], list[Hashable]]

REPRESENTATION_REGISTRY: dict[str, Tokenizer] = {
    "raw_cell": raw_cell_tokens,
    "value_change": value_change_tokens,
    "local_patch": local_patch_tokens,
    "dynamical_regime": dynamical_regime_tokens,
}

REPRESENTATION_LABELS = {
    "raw_cell": "Raw cell",
    "value_change": "Value-change",
    "local_patch": "Local patch",
    "dynamical_regime": "Dynamical regime",
}

PREDICTOR_FACTORIES = {
    "mean_value": MeanValuePredictor,
    "raw_cell_lookup": RawCellLookupPredictor,
    "value_change_token": ValueChangeTokenPredictor,
    "local_patch_token": LocalPatchTokenPredictor,
    "dynamical_regime_token": RegimeTokenPredictor,
}

PREDICTOR_LABELS = {
    "mean_value": "Mean value",
    "raw_cell_lookup": "Raw cell lookup",
    "value_change_token": "Value-change token",
    "local_patch_token": "Local patch token",
    "dynamical_regime_token": "Dynamical regime token",
}


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _mean_std(rows: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    if not rows:
        return {}, {}
    keys = rows[0].keys()
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        means[key] = float(np.mean(values))
        stds[key] = float(np.std(values))
    return means, stds


def _records_by_environment(
    environments: Sequence[FieldEnvironment],
) -> dict[str, list[dict[str, Any]]]:
    return {env.env_id: field_records(env) for env in environments}


def _tokens_by_environment(
    records_by_env: Mapping[str, Sequence[Record]], tokenizer: Tokenizer
) -> dict[str, list[Hashable]]:
    return {env_id: tokenizer(records) for env_id, records in records_by_env.items()}


def _select_top_coverage_key(global_summary: dict[str, Any]) -> str:
    if "top_50_coverage" in global_summary:
        return "top_50_coverage"
    coverage_keys = [
        key for key in global_summary if key.startswith("top_") and key.endswith("_coverage")
    ]
    if not coverage_keys:
        raise ValueError("global summary does not contain top-k coverage")
    return max(coverage_keys, key=lambda key: int(key.split("_")[1]))


def _build_diagnostic_rows(results: dict[str, Any]) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    latex_rows: list[list[Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for name, representation in results["representations"].items():
        global_summary = representation["global"]
        cross_env = representation["cross_environment_mean"]
        top_key = _select_top_coverage_key(global_summary)
        row = {
            "Representation": REPRESENTATION_LABELS[name],
            "Vocab": int(global_summary["vocab_size"]),
            "Eff. Vocab / N": format_float(global_summary["normalized_effective_vocab_size"]),
            "Top-k Cov.": format_float(global_summary[top_key]),
            "Cross-Env Cov.": format_float(cross_env["token_coverage"]),
            "JSD": format_float(cross_env["js_divergence"]),
            "Bigram Cov.": format_float(cross_env["bigram_coverage"]),
            "Novel Recomb.": format_float(cross_env["novel_recombination_rate"]),
        }
        csv_rows.append(row)
        latex_rows.append(list(row.values()))
    return latex_rows, csv_rows


def _build_transfer_rows(results: dict[str, Any]) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    latex_rows: list[list[Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for name, predictor in results["predictors"].items():
        mean = predictor["mean"]
        std = predictor["std"]
        row = {
            "Predictor": PREDICTOR_LABELS[name],
            "Mean Abs. Error": format_float(mean["mean_abs_error"]),
            "Median Abs. Error": format_float(mean["median_abs_error"]),
            "Transfer Std.": format_float(std["mean_abs_error"]),
            "N Records": format_float(mean["n_records"], digits=1),
        }
        csv_rows.append(row)
        latex_rows.append(list(row.values()))
    return latex_rows, csv_rows


def _write_artifacts(results: dict[str, Any], output_dirs: dict[str, Path]) -> dict[str, str]:
    artifacts: dict[str, str] = {}

    diagnostic_rows, diagnostic_csv = _build_diagnostic_rows(results)
    diagnostic_columns = [
        "Representation",
        "Vocab",
        "Eff. Vocab / N",
        "Top-k Cov.",
        "Cross-Env Cov.",
        "JSD",
        "Bigram Cov.",
        "Novel Recomb.",
    ]
    diagnostic_tex = write_latex_table(
        output_dirs["tables"] / "table_physical_field_diagnostics.tex",
        caption="Synthetic physical-field diagnostics across representations.",
        label="tab:physical_field_diagnostics",
        columns=diagnostic_columns,
        rows=diagnostic_rows,
    )
    diagnostic_csv_path = write_csv(
        output_dirs["tables"] / "table_physical_field_diagnostics.csv",
        diagnostic_csv,
    )

    transfer_rows, transfer_csv = _build_transfer_rows(results)
    transfer_columns = [
        "Predictor",
        "Mean Abs. Error",
        "Median Abs. Error",
        "Transfer Std.",
        "N Records",
    ]
    transfer_tex = write_latex_table(
        output_dirs["tables"] / "table_physical_field_transfer.tex",
        caption="Synthetic physical-field next-step transfer.",
        label="tab:physical_field_transfer",
        columns=transfer_columns,
        rows=transfer_rows,
    )
    transfer_csv_path = write_csv(
        output_dirs["tables"] / "table_physical_field_transfer.csv",
        transfer_csv,
    )

    json_path = output_dirs["data"] / "physical_field_results.json"
    write_text(json_path, json.dumps(_to_builtin(results), indent=2, sort_keys=True) + "\n")

    labels = [REPRESENTATION_LABELS[name] for name in results["representations"]]
    coverage = [
        representation["cross_environment_mean"]["token_coverage"]
        for representation in results["representations"].values()
    ]
    jsd = [
        representation["cross_environment_mean"]["js_divergence"]
        for representation in results["representations"].values()
    ]
    predictor_labels = [PREDICTOR_LABELS[name] for name in results["predictors"]]
    transfer_error = [
        predictor["mean"]["mean_abs_error"] for predictor in results["predictors"].values()
    ]

    coverage_pdf, coverage_png = save_bar_chart(
        output_dirs["figures"] / "physical_field_cross_env_coverage.pdf",
        output_dirs["figures"] / "physical_field_cross_env_coverage.png",
        labels,
        coverage,
        ylabel="Token coverage",
        title="Physical-field cross-environment token coverage",
    )
    jsd_pdf, jsd_png = save_bar_chart(
        output_dirs["figures"] / "physical_field_jsd.pdf",
        output_dirs["figures"] / "physical_field_jsd.png",
        labels,
        jsd,
        ylabel="Jensen-Shannon divergence",
        title="Physical-field cross-environment distribution shift",
    )
    transfer_pdf, transfer_png = save_bar_chart(
        output_dirs["figures"] / "physical_field_transfer_error.pdf",
        output_dirs["figures"] / "physical_field_transfer_error.png",
        predictor_labels,
        transfer_error,
        ylabel="Mean absolute error",
        title="Physical-field next-step transfer error",
    )

    artifacts.update(
        {
            "diagnostics_latex_table": str(diagnostic_tex),
            "diagnostics_csv_table": str(diagnostic_csv_path),
            "transfer_latex_table": str(transfer_tex),
            "transfer_csv_table": str(transfer_csv_path),
            "json_results": str(json_path),
            "cross_env_coverage_pdf": str(coverage_pdf),
            "cross_env_coverage_png": str(coverage_png),
            "jsd_pdf": str(jsd_pdf),
            "jsd_png": str(jsd_png),
            "transfer_error_pdf": str(transfer_pdf),
            "transfer_error_png": str(transfer_png),
        }
    )
    return artifacts


def run_physical_field_diagnostics_and_transfer(
    config: dict[str, Any], output_dirs: dict[str, Path]
) -> dict[str, Any]:
    """Run synthetic physical-field diagnostics and transfer."""
    seed = int(config.get("experiment", {}).get("seed", 123))
    physical_config = config.get("physical_field", {})
    synthetic = physical_config.get("synthetic", {})
    diagnostics = physical_config.get("diagnostics", {})

    n_environments = int(synthetic.get("n_environments", 5))
    grid_size = int(synthetic.get("grid_size", 24))
    timesteps = int(synthetic.get("timesteps", 30))
    noise_std = float(synthetic.get("noise_std", 0.01))
    top_ks = tuple(int(k) for k in diagnostics.get("top_ks", (10, 50, 100)))
    top_k_jaccard = int(diagnostics.get("top_k_jaccard", 50))

    if n_environments < 2:
        raise ValueError("physical-field diagnostics require at least two environments")

    environments = generate_physical_field_environments(
        n_environments=n_environments,
        grid_size=grid_size,
        timesteps=timesteps,
        seed=seed,
        noise_std=noise_std,
    )
    records_by_env = _records_by_environment(environments)

    results: dict[str, Any] = {
        "experiment": "physical_field_diagnostics_transfer",
        "representations": {},
        "predictors": {},
    }

    for name, tokenizer in REPRESENTATION_REGISTRY.items():
        env_tokens = _tokens_by_environment(records_by_env, tokenizer)
        env_sequences = {
            env_id: group_token_sequences(
                tokens, [(r["x"], r["y"]) for r in records_by_env[env_id]]
            )
            for env_id, tokens in env_tokens.items()
        }
        all_tokens = [token for env_id in sorted(env_tokens) for token in env_tokens[env_id]]
        per_environment = {
            env_id: diagnostic_summary(tokens, top_ks=top_ks)
            for env_id, tokens in env_tokens.items()
        }
        transfer_rows: list[dict[str, float]] = []
        cross_context: dict[str, dict[str, float]] = {}
        for held_out_env, test_tokens in env_tokens.items():
            train_sequences = [
                sequence
                for env_id, sequences in env_sequences.items()
                if env_id != held_out_env
                for sequence in sequences
            ]
            transfer_rows.append(
                sequence_transfer_summary(
                    train_sequences, env_sequences[held_out_env], top_k=top_k_jaccard
                )
            )

            cross_context[held_out_env] = transfer_rows[-1]

        cross_mean, cross_std = _mean_std(transfer_rows)
        results["representations"][name] = {
            "global": diagnostic_summary(all_tokens, top_ks=top_ks),
            "per_environment": per_environment,
            "cross_environment": cross_context,
            "cross_environment_mean": cross_mean,
            "cross_environment_std": cross_std,
        }

    for name, factory in PREDICTOR_FACTORIES.items():
        per_environment = evaluate_leave_one_environment_prediction(records_by_env, factory)
        mean, std = aggregate_prediction_results(per_environment)
        results["predictors"][name] = {
            "per_environment": per_environment,
            "mean": mean,
            "std": std,
        }

    results["artifacts"] = _write_artifacts(results, output_dirs)
    return _to_builtin(results)
