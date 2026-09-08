# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import yaml

from motifshift.config import load_config


def test_load_smoke_config_returns_dict() -> None:
    config = load_config("configs/smoke.yaml")

    assert isinstance(config, dict)
    assert config["experiment"]["name"] == "smoke"


def test_missing_config_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("configs/missing.yaml")


def test_non_dict_config_raises_value_error(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(config_path)


def test_invalid_yaml_is_not_swallowed(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("experiment: [unclosed\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_config(config_path)
