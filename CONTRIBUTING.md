# Contributing to MotifShift

Small, focused contributions are welcome. Open an issue to discuss changes to
experimental protocols or new dependencies before investing in a large change.

## Development

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

Use `uv run ruff format .` to apply formatting. Add regression tests for bugs and
new behavior. Keep generated outputs, virtual environments, and credentials out of
commits. Existing output directories are protected against accidental overwriting;
use a new `--outputs` path for each run.

Changes to generators, token boundaries, predictor features, splits, or aggregation
must describe the effect on numerical results. Hold out entire contexts, fit only
on training contexts, and keep prediction inputs independent of future targets.
Retain raw numerical results before rounding for display.

For dependency updates, change `pyproject.toml` if needed, run `uv lock --upgrade-package NAME`,
and regenerate the pip export:

```bash
uv export --locked --no-dev --no-emit-project --format requirements.txt --output-file requirements.txt
```

Run the complete tests and quick experiment after changing the lockfile. The
`requirements.txt` file is generated from `uv.lock`; do not edit it manually.

## Collaboration and licensing

Be respectful, specific, and constructive when discussing changes. Contributions
are accepted under Apache-2.0, the project's license. Include attribution for any
third-party material and ensure you have the right to contribute it.
