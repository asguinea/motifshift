# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""RL recurrence diagnostics and cross-layout transfer experiment."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from motifshift.io import write_text
from motifshift.metrics import diagnostic_summary
from motifshift.metrics.sequences import group_token_sequences, sequence_transfer_summary
from motifshift.reporting.figures import save_bar_chart
from motifshift.reporting.tables import format_float, write_csv, write_latex_table
from motifshift.representations.rl import (
    local_observation_action_tokens,
    raw_state_action_tokens,
    situation_action_tokens,
    situation_transition_tokens,
)
from motifshift.rl.gridworld import (
    GridWorld,
    RLStep,
    generate_episodes,
    generate_gridworlds,
)
from motifshift.rl.predictors import (
    aggregate_prediction_results,
    evaluate_leave_one_layout_prediction,
)

Tokenizer = Callable[[Sequence[RLStep], Mapping[str, GridWorld]], list[Hashable]]

REPRESENTATION_REGISTRY: dict[str, Tokenizer] = {
    "raw_state_action": raw_state_action_tokens,
    "local_observation_action": local_observation_action_tokens,
    "situation_action": situation_action_tokens,
    "situation_transition": situation_transition_tokens,
}

REPRESENTATION_LABELS = {
    "raw_state_action": "Raw state-action",
    "local_observation_action": "Local observation-action",
    "situation_action": "Situation-action",
    "situation_transition": "Situation-transition",
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


def _group_steps_by_layout(steps: Sequence[RLStep]) -> dict[str, list[RLStep]]:
    grouped: dict[str, list[RLStep]] = defaultdict(list)
    for step in steps:
        grouped[step.layout_id].append(step)
    return {layout_id: grouped[layout_id] for layout_id in sorted(grouped)}


def _tokens_by_layout(
    steps_by_layout: Mapping[str, Sequence[RLStep]],
    worlds_by_id: Mapping[str, GridWorld],
    tokenizer: Tokenizer,
) -> dict[str, list[Hashable]]:
    return {
        layout_id: tokenizer(steps, worlds_by_id) for layout_id, steps in steps_by_layout.items()
    }


def _select_top_coverage_key(global_summary: dict[str, Any]) -> str:
    if "top_50_coverage" in global_summary:
        return "top_50_coverage"
    coverage_keys = [
        key for key in global_summary if key.startswith("top_") and key.endswith("_coverage")
    ]
    if not coverage_keys:
        raise ValueError("global summary does not contain top-k coverage")
    return max(coverage_keys, key=lambda key: int(key.split("_")[1]))


def _build_diagnostics_rows(
    results: dict[str, Any],
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    latex_rows: list[list[Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for name, representation in results["representations"].items():
        global_summary = representation["global"]
        cross_layout = representation["cross_layout_mean"]
        top_coverage_key = _select_top_coverage_key(global_summary)
        values = {
            "Representation": REPRESENTATION_LABELS[name],
            "Vocab": int(global_summary["vocab_size"]),
            "Eff. Vocab / N": format_float(global_summary["normalized_effective_vocab_size"]),
            "Top-k Cov.": format_float(global_summary[top_coverage_key]),
            "Cross-Layout Cov.": format_float(cross_layout["token_coverage"]),
            "JSD": format_float(cross_layout["js_divergence"]),
            "Bigram Cov.": format_float(cross_layout["bigram_coverage"]),
            "Novel Recomb.": format_float(cross_layout["novel_recombination_rate"]),
        }
        csv_rows.append(values)
        latex_rows.append(list(values.values()))

    return latex_rows, csv_rows


def _build_transfer_rows(results: dict[str, Any]) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    latex_rows: list[list[Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for name, representation in results["representations"].items():
        cross_layout = representation["cross_layout_mean"]
        prediction = representation["prediction_transfer_mean"]
        values = {
            "Representation": REPRESENTATION_LABELS[name],
            "Next-token Acc.": format_float(prediction["next_token_accuracy"]),
            "Cross-Layout Cov.": format_float(cross_layout["token_coverage"]),
            "Bigram Cov.": format_float(cross_layout["bigram_coverage"]),
            "Novel Recomb.": format_float(cross_layout["novel_recombination_rate"]),
        }
        csv_rows.append(values)
        latex_rows.append(list(values.values()))

    return latex_rows, csv_rows


def _write_artifacts(results: dict[str, Any], output_dirs: dict[str, Path]) -> dict[str, str]:
    artifacts: dict[str, str] = {}

    diagnostics_rows, diagnostics_csv = _build_diagnostics_rows(results)
    diagnostics_columns = [
        "Representation",
        "Vocab",
        "Eff. Vocab / N",
        "Top-k Cov.",
        "Cross-Layout Cov.",
        "JSD",
        "Bigram Cov.",
        "Novel Recomb.",
    ]
    diagnostics_tex = write_latex_table(
        output_dirs["tables"] / "table_rl_diagnostics.tex",
        caption="RL cross-layout diagnostics across representations.",
        label="tab:rl_diagnostics",
        columns=diagnostics_columns,
        rows=diagnostics_rows,
    )
    diagnostics_csv_path = write_csv(
        output_dirs["tables"] / "table_rl_diagnostics.csv", diagnostics_csv
    )

    transfer_rows, transfer_csv = _build_transfer_rows(results)
    transfer_columns = [
        "Representation",
        "Next-token Acc.",
        "Cross-Layout Cov.",
        "Bigram Cov.",
        "Novel Recomb.",
    ]
    transfer_tex = write_latex_table(
        output_dirs["tables"] / "table_rl_transfer.tex",
        caption="RL next-token transfer across layouts.",
        label="tab:rl_transfer",
        columns=transfer_columns,
        rows=transfer_rows,
    )
    transfer_csv_path = write_csv(output_dirs["tables"] / "table_rl_transfer.csv", transfer_csv)

    serializable = _to_builtin(results)
    json_path = output_dirs["data"] / "rl_diagnostics_results.json"
    write_text(json_path, json.dumps(serializable, indent=2, sort_keys=True) + "\n")

    labels = [REPRESENTATION_LABELS[name] for name in results["representations"]]
    cross_layout_coverage = [
        representation["cross_layout_mean"]["token_coverage"]
        for representation in results["representations"].values()
    ]
    jsd = [
        representation["cross_layout_mean"]["js_divergence"]
        for representation in results["representations"].values()
    ]
    next_token_accuracy = [
        representation["prediction_transfer_mean"]["next_token_accuracy"]
        for representation in results["representations"].values()
    ]

    coverage_pdf, coverage_png = save_bar_chart(
        output_dirs["figures"] / "rl_cross_layout_coverage.pdf",
        output_dirs["figures"] / "rl_cross_layout_coverage.png",
        labels,
        cross_layout_coverage,
        ylabel="Token coverage",
        title="RL cross-layout token coverage",
    )
    jsd_pdf, jsd_png = save_bar_chart(
        output_dirs["figures"] / "rl_jsd.pdf",
        output_dirs["figures"] / "rl_jsd.png",
        labels,
        jsd,
        ylabel="Jensen-Shannon divergence",
        title="RL cross-layout distribution shift",
    )
    accuracy_pdf, accuracy_png = save_bar_chart(
        output_dirs["figures"] / "rl_next_token_accuracy.pdf",
        output_dirs["figures"] / "rl_next_token_accuracy.png",
        labels,
        next_token_accuracy,
        ylabel="Next-token accuracy",
        title="RL leave-one-layout next-token transfer",
    )

    artifacts.update(
        {
            "diagnostics_latex_table": str(diagnostics_tex),
            "diagnostics_csv_table": str(diagnostics_csv_path),
            "transfer_latex_table": str(transfer_tex),
            "transfer_csv_table": str(transfer_csv_path),
            "json_results": str(json_path),
            "cross_layout_coverage_pdf": str(coverage_pdf),
            "cross_layout_coverage_png": str(coverage_png),
            "jsd_pdf": str(jsd_pdf),
            "jsd_png": str(jsd_png),
            "next_token_accuracy_pdf": str(accuracy_pdf),
            "next_token_accuracy_png": str(accuracy_png),
        }
    )
    return artifacts


def run_rl_diagnostics_and_transfer(
    config: dict[str, Any], output_dirs: dict[str, Path]
) -> dict[str, Any]:
    """Run synthetic RL diagnostics and cross-layout transfer."""
    seed = int(config.get("experiment", {}).get("seed", 123))
    rl_config = config.get("rl", {})
    synthetic = rl_config.get("synthetic", {})
    diagnostics = rl_config.get("diagnostics", {})

    n_layouts = int(synthetic.get("n_layouts", 5))
    width = int(synthetic.get("width", 12))
    height = int(synthetic.get("height", 12))
    episodes_per_world = int(synthetic.get("episodes_per_world", 30))
    max_steps = int(synthetic.get("max_steps", 50))
    noise_prob = float(synthetic.get("noise_prob", 0.15))
    top_ks = tuple(int(k) for k in diagnostics.get("top_ks", (10, 50, 100)))
    top_k_jaccard = int(diagnostics.get("top_k_jaccard", 50))

    if n_layouts < 2:
        raise ValueError("RL diagnostics require at least two layouts")

    worlds = generate_gridworlds(
        n_layouts=n_layouts,
        width=width,
        height=height,
        seed=seed,
    )
    steps = generate_episodes(
        worlds,
        episodes_per_world=episodes_per_world,
        max_steps=max_steps,
        seed=seed,
        noise_prob=noise_prob,
    )
    worlds_by_id = {world.layout_id: world for world in worlds}
    steps_by_layout = _group_steps_by_layout(steps)

    results: dict[str, Any] = {
        "experiment": "rl_diagnostics_transfer",
        "representations": {},
    }

    for name, tokenizer in REPRESENTATION_REGISTRY.items():
        layout_tokens = _tokens_by_layout(steps_by_layout, worlds_by_id, tokenizer)
        layout_sequences = {
            layout_id: group_token_sequences(
                tokens, [step.episode_id for step in steps_by_layout[layout_id]]
            )
            for layout_id, tokens in layout_tokens.items()
        }
        all_tokens = [
            token for layout_id in sorted(layout_tokens) for token in layout_tokens[layout_id]
        ]
        per_layout = {
            layout_id: diagnostic_summary(tokens, top_ks=top_ks)
            for layout_id, tokens in layout_tokens.items()
        }

        transfer_rows: list[dict[str, float]] = []
        cross_context: dict[str, dict[str, float]] = {}
        for held_out_layout, test_tokens in layout_tokens.items():
            train_sequences = [
                sequence
                for layout_id, sequences in layout_sequences.items()
                if layout_id != held_out_layout
                for sequence in sequences
            ]
            transfer_rows.append(
                sequence_transfer_summary(
                    train_sequences, layout_sequences[held_out_layout], top_k=top_k_jaccard
                )
            )

            cross_context[held_out_layout] = transfer_rows[-1]

        cross_layout_mean, cross_layout_std = _mean_std(transfer_rows)
        prediction_by_layout = evaluate_leave_one_layout_prediction(
            layout_tokens, sequences_by_layout=layout_sequences
        )
        prediction_mean, prediction_std = aggregate_prediction_results(prediction_by_layout)

        results["representations"][name] = {
            "global": diagnostic_summary(all_tokens, top_ks=top_ks),
            "per_layout": per_layout,
            "cross_layout": cross_context,
            "cross_layout_mean": cross_layout_mean,
            "cross_layout_std": cross_layout_std,
            "prediction_by_layout": prediction_by_layout,
            "prediction_transfer_mean": prediction_mean,
            "prediction_transfer_std": prediction_std,
        }

    results["artifacts"] = _write_artifacts(results, output_dirs)
    return _to_builtin(results)
