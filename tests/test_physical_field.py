# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from motifshift.physical.field import (
    field_records,
    generate_physical_field_environments,
)


def test_physical_field_generation_is_deterministic() -> None:
    envs_a = generate_physical_field_environments(
        n_environments=2, grid_size=8, timesteps=6, seed=123
    )
    envs_b = generate_physical_field_environments(
        n_environments=2, grid_size=8, timesteps=6, seed=123
    )

    assert envs_a[0].env_id == "field_0"
    assert np.allclose(envs_a[0].fields, envs_b[0].fields)
    assert envs_a[0].wind == envs_b[0].wind


def test_different_seeds_produce_different_fields() -> None:
    envs_a = generate_physical_field_environments(
        n_environments=1, grid_size=8, timesteps=6, seed=123
    )
    envs_b = generate_physical_field_environments(
        n_environments=1, grid_size=8, timesteps=6, seed=456
    )

    assert not np.allclose(envs_a[0].fields, envs_b[0].fields)


def test_physical_fields_have_expected_shape_and_range() -> None:
    envs = generate_physical_field_environments(
        n_environments=3, grid_size=10, timesteps=7, seed=123
    )

    assert len(envs) == 3
    for env in envs:
        assert env.fields.shape == (7, 10, 10)
        assert np.isfinite(env.fields).all()
        assert float(env.fields.min()) >= 0.0
        assert float(env.fields.max()) <= 1.0


def test_field_records_are_nonempty_and_have_required_keys() -> None:
    env = generate_physical_field_environments(
        n_environments=1, grid_size=6, timesteps=4, seed=123
    )[0]

    records = field_records(env)

    assert len(records) == (4 - 1) * 6 * 6
    required = {
        "env_id",
        "t",
        "x",
        "y",
        "value",
        "next_value",
        "delta",
        "north",
        "south",
        "east",
        "west",
        "grad_x",
        "grad_y",
        "grad_mag",
    }
    assert required <= set(records[0])
    assert all(
        np.isfinite(
            list(
                record_value
                for record_value in [
                    records[0]["value"],
                    records[0]["next_value"],
                    records[0]["delta"],
                    records[0]["grad_mag"],
                ]
            )
        )
    )
