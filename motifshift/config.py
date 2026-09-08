# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file as a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config must parse to a dictionary: {config_path}")

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Reject malformed settings before experiments create any output files."""
    import math

    def mapping(value: Any, key: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a mapping")
        return value

    def integer(value: Any, key: str, minimum: int = 0) -> None:
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")

    def boolean(value: Any, key: str) -> None:
        if type(value) is not bool:
            raise ValueError(f"{key} must be true or false")

    def known(section: dict[str, Any], allowed: set[str], path: str) -> None:
        extra = set(section) - allowed
        if extra:
            raise ValueError(f"unknown {path} settings: {sorted(extra)}")

    known(
        config,
        {
            "experiment",
            "outputs",
            "runtime",
            "spatiotemporal",
            "rl",
            "physical_field",
            "robustness",
        },
        "config",
    )
    for key, section in config.items():
        mapping(section, key)
    experiment = config.get("experiment", {})
    known(experiment, {"name", "seed", "smoke"}, "experiment")
    integer(experiment.get("seed", 0), "experiment.seed")
    if experiment.get("seed", 0) >= 2**32:
        raise ValueError("experiment.seed must be less than 2**32")
    boolean(experiment.get("smoke", False), "experiment.smoke")
    if "name" in experiment and not isinstance(experiment["name"], str):
        raise ValueError("experiment.name must be a string")
    runtime = config.get("runtime", {})
    known(runtime, {"verbose"}, "runtime")
    boolean(runtime.get("verbose", False), "runtime.verbose")
    outputs = config.get("outputs", {})
    known(outputs, {"root", "reports_dir", "tables_dir", "figures_dir", "data_dir"}, "outputs")
    subdirs = []
    for key, default in [
        ("reports_dir", "reports"),
        ("tables_dir", "tables"),
        ("figures_dir", "figures"),
        ("data_dir", "data"),
    ]:
        value = outputs.get(key, default)
        if (
            not isinstance(value, str)
            or not value
            or Path(value).is_absolute()
            or ".." in Path(value).parts
            or value == "."
        ):
            raise ValueError(f"outputs.{key} must be a relative subdirectory without ..")
        subdirs.append(Path(value))
    if any(
        a == b or a in b.parents or b in a.parents
        for i, a in enumerate(subdirs)
        for b in subdirs[i + 1 :]
    ):
        raise ValueError("output subdirectories must be distinct and non-overlapping")
    if not isinstance(outputs.get("root", "outputs"), str) or not outputs.get("root", "outputs"):
        raise ValueError("outputs.root must be a non-empty path string")
    specs = {
        "spatiotemporal": {
            "n_scenes": 2,
            "agents_per_scene": 1,
            "n_agents": 1,
            "timesteps": 3,
            "n_steps": 3,
        },
        "rl": {"n_layouts": 2, "width": 6, "height": 6, "episodes_per_world": 1, "max_steps": 2},
        "physical_field": {"n_environments": 2, "grid_size": 5, "timesteps": 3},
    }
    floats = {
        "spatiotemporal": {"noise_std", "speed_variation", "scene_variation"},
        "rl": {"noise_prob"},
        "physical_field": {"noise_std"},
    }
    for domain, ints in specs.items():
        section = config.get(domain, {})
        known(section, {"enabled", "synthetic", "diagnostics", "transfer"}, domain)
        boolean(section.get("enabled", False), f"{domain}.enabled")
        if experiment.get("smoke") and section.get("enabled"):
            raise ValueError(
                "smoke mode requires all domains to be disabled; use the quick config for real experiments"
            )
        synthetic = mapping(section.get("synthetic", {}), f"{domain}.synthetic")
        known(synthetic, set(ints) | floats[domain], f"{domain}.synthetic")
        for key, minimum in ints.items():
            if key in synthetic:
                integer(synthetic[key], f"{domain}.synthetic.{key}", minimum)
        for key in floats[domain]:
            if key in synthetic:
                value = synthetic[key]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                    or (key == "noise_prob" and value > 1)
                ):
                    raise ValueError(f"{domain}.synthetic.{key} is outside its valid range")
        diagnostic = mapping(section.get("diagnostics", {}), f"{domain}.diagnostics")
        transfer = mapping(section.get("transfer", {}), f"{domain}.transfer")
        known(diagnostic, {"enabled", "top_ks", "top_k_jaccard"}, f"{domain}.diagnostics")
        known(
            transfer,
            {"enabled", "horizon", "k_neighbors"} if domain == "spatiotemporal" else {"enabled"},
            f"{domain}.transfer",
        )
        for key, value in [("diagnostics", diagnostic), ("transfer", transfer)]:
            boolean(value.get("enabled", False), f"{domain}.{key}.enabled")
        top_ks = diagnostic.get("top_ks", [10, 50, 100])
        if not isinstance(top_ks, list) or not top_ks:
            raise ValueError(f"{domain}.diagnostics.top_ks must be a non-empty list")
        for k in top_ks:
            integer(k, f"{domain}.diagnostics.top_ks", 1)
        integer(diagnostic.get("top_k_jaccard", 50), f"{domain}.diagnostics.top_k_jaccard", 1)
        if (
            domain != "spatiotemporal"
            and section.get("enabled")
            and diagnostic.get("enabled", False) != transfer.get("enabled", False)
        ):
            raise ValueError(
                f"{domain} currently runs diagnostics and transfer together; set both flags identically"
            )
        if domain == "spatiotemporal":
            integer(transfer.get("horizon", 1), "spatiotemporal.transfer.horizon", 1)
            if transfer.get("horizon", 1) != 1:
                raise ValueError("only horizon=1 is supported")
            integer(transfer.get("k_neighbors", 5), "spatiotemporal.transfer.k_neighbors", 1)
    robustness = config.get("robustness", {})
    known(robustness, {"enabled", "seeds"}, "robustness")
    boolean(robustness.get("enabled", False), "robustness.enabled")
    seeds = robustness.get("seeds", [123])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("robustness.seeds must be a non-empty list")
    for seed in seeds:
        integer(seed, "robustness.seeds")
        if seed >= 2**32:
            raise ValueError("robustness seeds must be less than 2**32")
    if len(set(seeds)) != len(seeds):
        raise ValueError("robustness.seeds must be unique")
