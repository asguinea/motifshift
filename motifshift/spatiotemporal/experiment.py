# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Spatio-temporal recurrence and compositionality diagnostics."""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from motifshift.io import write_text
from motifshift.metrics import diagnostic_summary
from motifshift.metrics.sequences import group_token_sequences, sequence_transfer_summary
from motifshift.reporting.figures import save_bar_chart
from motifshift.reporting.tables import format_float, write_csv, write_latex_table
from motifshift.representations.spatiotemporal import (
    _sorted_points,
    _transitions,
    absolute_grid_tokens,
    motion_tokens,
    relative_displacement_grid_tokens,
    situation_motion_tokens,
)
from motifshift.spatiotemporal.data import (
    Scene,
    TrajectoryPoint,
    generate_synthetic_scenes,
    predictive_trajectory_steps,
)
from motifshift.spatiotemporal.predictors import (
    ConstantVelocityPredictor,
    KNNDisplacementPredictor,
    MeanDisplacementPredictor,
    MotionTokenMarkovPredictor,
    SituationMotionTokenPredictor,
    aggregate_error_results,
    evaluate_leave_one_scene_prediction,
)

Tokenizer = Callable[[Sequence[TrajectoryPoint]], list[Hashable]]

REPRESENTATION_REGISTRY: dict[str, Tokenizer] = {
    "absolute_grid": absolute_grid_tokens,
    "relative_displacement": relative_displacement_grid_tokens,
    "motion": motion_tokens,
    "situation_motion": situation_motion_tokens,
}

REPRESENTATION_LABELS = {
    "absolute_grid": "Absolute grid",
    "relative_displacement": "Relative displacement",
    "motion": "Motion",
    "situation_motion": "Situation-motion",
}

PREDICTOR_LABELS = {
    "constant_velocity": "Constant velocity",
    "mean_displacement": "Mean displacement",
    "knn_raw": "Raw kNN",
    "motion_token": "Motion-token",
    "situation_motion_token": "Situation-motion",
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


def _tokens_by_scene(scenes: list[Scene], tokenizer: Tokenizer) -> dict[str, list[Hashable]]:
    return {scene.scene_id: tokenizer(scene.points) for scene in scenes}


def _select_top_coverage_key(global_summary: dict[str, Any]) -> str:
    if "top_50_coverage" in global_summary:
        return "top_50_coverage"
    coverage_keys = [
        key for key in global_summary if key.startswith("top_") and key.endswith("_coverage")
    ]
    if not coverage_keys:
        raise ValueError("global summary does not contain top-k coverage")
    return max(coverage_keys, key=lambda key: int(key.split("_")[1]))


def _build_table_rows(results: dict[str, Any]) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    latex_rows: list[list[Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for name, representation in results["representations"].items():
        global_summary = representation["global"]
        cross_scene = representation["cross_scene_mean"]
        top_coverage_key = _select_top_coverage_key(global_summary)

        values = {
            "Representation": REPRESENTATION_LABELS[name],
            "Vocab": int(global_summary["vocab_size"]),
            "Eff. Vocab / N": format_float(global_summary["normalized_effective_vocab_size"]),
            "Top-k Cov.": format_float(global_summary[top_coverage_key]),
            "Cross-Scene Cov.": format_float(cross_scene["token_coverage"]),
            "JSD": format_float(cross_scene["js_divergence"]),
            "Bigram Cov.": format_float(cross_scene["bigram_coverage"]),
            "Novel Recomb.": format_float(cross_scene["novel_recombination_rate"]),
        }
        csv_rows.append(values)
        latex_rows.append(list(values.values()))

    return latex_rows, csv_rows


def _write_artifacts(
    results: dict[str, Any],
    output_dirs: dict[str, Path],
) -> dict[str, str]:
    artifacts: dict[str, str] = {}

    latex_rows, csv_rows = _build_table_rows(results)
    columns = [
        "Representation",
        "Vocab",
        "Eff. Vocab / N",
        "Top-k Cov.",
        "Cross-Scene Cov.",
        "JSD",
        "Bigram Cov.",
        "Novel Recomb.",
    ]

    latex_path = write_latex_table(
        output_dirs["tables"] / "table_spatiotemporal_diagnostics.tex",
        caption="Spatio-temporal diagnostics across representations.",
        label="tab:spatiotemporal_diagnostics",
        columns=columns,
        rows=latex_rows,
    )
    csv_path = write_csv(output_dirs["tables"] / "table_spatiotemporal_diagnostics.csv", csv_rows)

    serializable = _to_builtin(results)
    json_path = output_dirs["data"] / "spatiotemporal_diagnostics_results.json"
    write_text(json_path, json.dumps(serializable, indent=2, sort_keys=True) + "\n")

    labels = [REPRESENTATION_LABELS[name] for name in results["representations"]]
    cross_scene_coverage = [
        representation["cross_scene_mean"]["token_coverage"]
        for representation in results["representations"].values()
    ]
    jsd = [
        representation["cross_scene_mean"]["js_divergence"]
        for representation in results["representations"].values()
    ]
    topk_coverage = [
        representation["global"][_select_top_coverage_key(representation["global"])]
        for representation in results["representations"].values()
    ]

    coverage_pdf, coverage_png = save_bar_chart(
        output_dirs["figures"] / "spatiotemporal_cross_scene_coverage.pdf",
        output_dirs["figures"] / "spatiotemporal_cross_scene_coverage.png",
        labels,
        cross_scene_coverage,
        ylabel="Token coverage",
        title="Cross-scene token coverage",
    )
    jsd_pdf, jsd_png = save_bar_chart(
        output_dirs["figures"] / "spatiotemporal_jsd.pdf",
        output_dirs["figures"] / "spatiotemporal_jsd.png",
        labels,
        jsd,
        ylabel="Jensen-Shannon divergence",
        title="Cross-scene distribution shift",
    )
    top50_pdf, top50_png = save_bar_chart(
        output_dirs["figures"] / "spatiotemporal_token_coverage_topk.pdf",
        output_dirs["figures"] / "spatiotemporal_token_coverage_topk.png",
        labels,
        topk_coverage,
        ylabel="Top-k coverage",
        title="Global top-k token coverage",
    )

    artifacts.update(
        {
            "latex_table": str(latex_path),
            "csv_table": str(csv_path),
            "json_results": str(json_path),
            "cross_scene_coverage_pdf": str(coverage_pdf),
            "cross_scene_coverage_png": str(coverage_png),
            "jsd_pdf": str(jsd_pdf),
            "jsd_png": str(jsd_png),
            "topk_coverage_pdf": str(top50_pdf),
            "topk_coverage_png": str(top50_png),
        }
    )
    return artifacts


def run_spatiotemporal_diagnostics(
    config: dict[str, Any], output_dirs: dict[str, Path]
) -> dict[str, Any]:
    """Run synthetic spatio-temporal diagnostics and write artifacts."""
    synthetic_settings = _synthetic_settings(config)
    spatiotemporal = config.get("spatiotemporal", {})
    diagnostics = spatiotemporal.get("diagnostics", {})

    top_ks = tuple(int(k) for k in diagnostics.get("top_ks", (10, 50, 100)))
    top_k_jaccard = int(diagnostics.get("top_k_jaccard", 50))

    if synthetic_settings["n_scenes"] < 2:
        raise ValueError("spatiotemporal diagnostics require at least two scenes")
    if synthetic_settings["n_steps"] < 2:
        raise ValueError("spatiotemporal diagnostics require at least two timesteps")

    scenes = _generate_scenes_from_settings(synthetic_settings)

    results: dict[str, Any] = {
        "experiment": "spatiotemporal_diagnostics",
        "representations": {},
    }

    for name, tokenizer in REPRESENTATION_REGISTRY.items():
        scene_tokens = _tokens_by_scene(scenes, tokenizer)
        scene_sequences = {}
        for scene in scenes:
            points = (
                _sorted_points(scene.points)
                if name == "absolute_grid"
                else [current for _, current in _transitions(scene.points)]
            )
            scene_sequences[scene.scene_id] = group_token_sequences(
                scene_tokens[scene.scene_id], [point.agent_id for point in points]
            )
        all_tokens = [token for scene_id in scene_tokens for token in scene_tokens[scene_id]]
        per_scene = {
            scene_id: diagnostic_summary(tokens, top_ks=top_ks)
            for scene_id, tokens in scene_tokens.items()
        }

        transfer_rows: list[dict[str, float]] = []
        cross_context: dict[str, dict[str, float]] = {}
        for held_out_scene_id, test_tokens in scene_tokens.items():
            train_sequences = [
                sequence
                for scene_id, sequences in scene_sequences.items()
                if scene_id != held_out_scene_id
                for sequence in sequences
            ]
            transfer_rows.append(
                sequence_transfer_summary(
                    train_sequences, scene_sequences[held_out_scene_id], top_k=top_k_jaccard
                )
            )

            cross_context[held_out_scene_id] = transfer_rows[-1]

        cross_scene_mean, cross_scene_std = _mean_std(transfer_rows)
        results["representations"][name] = {
            "global": diagnostic_summary(all_tokens, top_ks=top_ks),
            "per_scene": per_scene,
            "cross_scene": cross_context,
            "cross_scene_mean": cross_scene_mean,
            "cross_scene_std": cross_scene_std,
        }

    results["artifacts"] = _write_artifacts(results, output_dirs)
    return _to_builtin(results)


def _synthetic_settings(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("experiment", {}).get("seed", 123))
    synthetic = config.get("spatiotemporal", {}).get("synthetic", {})
    return {
        "seed": seed,
        "n_scenes": int(synthetic.get("n_scenes", 3)),
        "n_agents": int(synthetic.get("agents_per_scene", synthetic.get("n_agents", 4))),
        "n_steps": int(synthetic.get("timesteps", synthetic.get("n_steps", 8))),
        "noise_std": float(synthetic.get("noise_std", 0.03)),
        "speed_variation": float(synthetic.get("speed_variation", 0.25)),
        "scene_variation": float(synthetic.get("scene_variation", 0.35)),
    }


def _generate_scenes_from_settings(settings: dict[str, Any]) -> list[Scene]:
    return generate_synthetic_scenes(
        n_scenes=settings["n_scenes"],
        n_agents=settings["n_agents"],
        n_steps=settings["n_steps"],
        seed=settings["seed"],
        noise_std=settings["noise_std"],
        speed_variation=settings["speed_variation"],
        scene_variation=settings["scene_variation"],
    )


def _transition_steps_by_scene(scenes: list[Scene]) -> dict[str, list[dict[str, Any]]]:
    steps_by_scene: dict[str, list[dict[str, Any]]] = {}
    for scene in scenes:
        by_agent: dict[int, list[TrajectoryPoint]] = {}
        for point in scene.points:
            by_agent.setdefault(point.agent_id, []).append(point)

        scene_steps: list[dict[str, Any]] = []
        for agent_id, points in by_agent.items():
            ordered = sorted(points, key=lambda point: point.t)
            for current, next_point in zip(ordered[:-1], ordered[1:]):
                dx = next_point.x - current.x
                dy = next_point.y - current.y
                scene_steps.append(
                    {
                        "scene_id": scene.scene_id,
                        "agent_id": agent_id,
                        "t": current.t,
                        "x": current.x,
                        "y": current.y,
                        "next_x": next_point.x,
                        "next_y": next_point.y,
                        "dx": dx,
                        "dy": dy,
                    }
                )
        steps_by_scene[scene.scene_id] = sorted(
            scene_steps, key=lambda step: (step["scene_id"], step["agent_id"], step["t"])
        )
    return steps_by_scene


def _transfer_predictor_factories(k_neighbors: int) -> dict[str, Callable[[], Any]]:
    return {
        "constant_velocity": ConstantVelocityPredictor,
        "mean_displacement": MeanDisplacementPredictor,
        "knn_raw": lambda: KNNDisplacementPredictor(k=k_neighbors),
        "motion_token": MotionTokenMarkovPredictor,
        "situation_motion_token": SituationMotionTokenPredictor,
    }


def _build_transfer_table_rows(
    results: dict[str, Any],
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    latex_rows: list[list[Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for name, predictor in results["predictors"].items():
        mean = predictor["mean"]
        std = predictor["std"]
        values = {
            "Predictor": PREDICTOR_LABELS[name],
            "Mean Error": format_float(mean["mean_error"]),
            "Median Error": format_float(mean["median_error"]),
            "Transfer Std.": format_float(std["mean_error"]),
            "N Steps": format_float(mean["n_steps"], digits=1),
        }
        csv_rows.append(values)
        latex_rows.append(list(values.values()))
    return latex_rows, csv_rows


def _write_transfer_artifacts(
    results: dict[str, Any], output_dirs: dict[str, Path]
) -> dict[str, str]:
    latex_rows, csv_rows = _build_transfer_table_rows(results)
    columns = ["Predictor", "Mean Error", "Median Error", "Transfer Std.", "N Steps"]

    latex_path = write_latex_table(
        output_dirs["tables"] / "table_spatiotemporal_transfer.tex",
        caption="Spatio-temporal leave-one-scene-out transfer.",
        label="tab:spatiotemporal_transfer",
        columns=columns,
        rows=latex_rows,
    )
    csv_path = write_csv(output_dirs["tables"] / "table_spatiotemporal_transfer.csv", csv_rows)

    json_path = output_dirs["data"] / "spatiotemporal_transfer_results.json"
    write_text(json_path, json.dumps(_to_builtin(results), indent=2, sort_keys=True) + "\n")

    labels = [PREDICTOR_LABELS[name] for name in results["predictors"]]
    mean_errors = [predictor["mean"]["mean_error"] for predictor in results["predictors"].values()]
    std_errors = [predictor["std"]["mean_error"] for predictor in results["predictors"].values()]
    error_pdf, error_png = save_bar_chart(
        output_dirs["figures"] / "spatiotemporal_transfer_error.pdf",
        output_dirs["figures"] / "spatiotemporal_transfer_error.png",
        labels,
        mean_errors,
        ylabel="Mean displacement error",
        title="Spatio-temporal transfer error",
    )
    std_pdf, std_png = save_bar_chart(
        output_dirs["figures"] / "spatiotemporal_transfer_error_std.pdf",
        output_dirs["figures"] / "spatiotemporal_transfer_error_std.png",
        labels,
        std_errors,
        ylabel="Std. of mean error",
        title="Spatio-temporal transfer variability",
    )

    return {
        "latex_table": str(latex_path),
        "csv_table": str(csv_path),
        "json_results": str(json_path),
        "transfer_error_pdf": str(error_pdf),
        "transfer_error_png": str(error_png),
        "transfer_error_std_pdf": str(std_pdf),
        "transfer_error_std_png": str(std_png),
    }


def run_spatiotemporal_transfer(
    config: dict[str, Any], output_dirs: dict[str, Path]
) -> dict[str, Any]:
    """Run leave-one-scene-out spatio-temporal transfer predictors."""
    synthetic_settings = _synthetic_settings(config)
    transfer_config = config.get("spatiotemporal", {}).get("transfer", {})
    horizon = int(transfer_config.get("horizon", 1))
    k_neighbors = int(transfer_config.get("k_neighbors", 5))

    if horizon != 1:
        raise ValueError("only horizon=1 is supported")
    if synthetic_settings["n_scenes"] < 2:
        raise ValueError("spatiotemporal transfer requires at least two scenes")
    if synthetic_settings["n_steps"] < 3:
        raise ValueError("spatiotemporal transfer requires at least three timesteps")

    scenes = _generate_scenes_from_settings(synthetic_settings)
    steps_by_scene = {scene.scene_id: predictive_trajectory_steps(scene) for scene in scenes}

    results: dict[str, Any] = {
        "experiment": "spatiotemporal_transfer",
        "predictors": {},
    }

    for name, factory in _transfer_predictor_factories(k_neighbors).items():
        per_scene = evaluate_leave_one_scene_prediction(steps_by_scene, factory)
        mean, std = aggregate_error_results(per_scene)
        results["predictors"][name] = {
            "per_scene": per_scene,
            "mean": mean,
            "std": std,
        }

    results["artifacts"] = _write_transfer_artifacts(results, output_dirs)
    return _to_builtin(results)
