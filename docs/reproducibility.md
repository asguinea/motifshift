# Reproducing a run

## Environment

Use the repository's `uv.lock` with `uv sync --locked`. The lockfile records exact
direct and transitive dependency versions and distribution hashes. The generated
`requirements.txt` provides a hashed pip installation route. The build backend is
pinned in `pyproject.toml`; the uv development environment also locks its dependencies.
Installing dependencies may access PyPI; experiment execution is local.

Python 3.11 is the minimum. Python version, platform, installed package versions,
MotifShift version, and source fingerprint are recorded in each run manifest.
Fix Python, dependencies, configuration, and source for close numerical reproduction.
Different architectures or library versions can change floating-point rounding.
PDF metadata contains generation times, so figure files are not expected to be
byte-identical across runs. Manifests also contain timestamps. Raw domain JSON is
tested for exact equality across fresh processes with different Python hash seeds
in the same environment.

## Run and verify

```bash
uv sync --locked
uv run --locked motifshift --config configs/quick.yaml --outputs outputs/reproduction
uv run --locked motifshift-verify --outputs outputs/reproduction --strict --json
```

A successful strict check returns status `ok` and exit code 0. Missing or modified
files, inconsistent configuration, invalid JSON, or unsafe paths fail verification.
The source fingerprint covers Python files inside `motifshift`; the configuration
fingerprint covers the portable JSON snapshot. The manifest records SHA-256 hashes
for the artifacts, including individual seed artifacts when robustness runs.
The manifest itself is not self-hashed. This detects accidental changes; someone
who changes both a result and its manifest can generate new matching checksums.

For multi-seed reproduction, append `--robustness` to the run command and
`--include-robustness` to verification. The quick configuration uses two seeds;
the default uses five. Seeds must be unique. The regular single-seed run is still
produced before the multi-seed results, so the base seed is evaluated again.

The saved `data/config.json` can be used directly as a config; JSON is accepted by
the YAML loader. Its output root is normalized to `.` for portability. Always give
it a new destination:

```bash
uv run --locked motifshift --config outputs/reproduction/data/config.json --outputs outputs/replay
```

## Configuration

- `experiment`: descriptive name, integer seed in [0, 2^32), smoke flag.
- `outputs`: output root plus relative data/report/table/figure subdirectories.
- `runtime.verbose`: progress logging.
- `spatiotemporal`, `rl`, `physical_field`: enabled flag, generator settings, and diagnostic/transfer settings.
- `robustness`: enabled flag and distinct integer seeds.

The smoke configuration disables all domain experiments. Use the quick config for
an actual miniature experiment. Trajectory diagnostics and transfer can be enabled
separately. Gridworld and physical-field stages currently compute diagnostics and
transfer together; their two flags must agree. Unsupported settings and malformed
values are rejected before outputs are written. Output directories must be fresh.

## Inspectable outputs

Each domain writes numerical JSON before display rounding. Context-level diagnostics
and aggregate statistics are recorded there; predictive summaries include held-out
scores. Multi-seed aggregate JSON is accompanied by the individual seed directories.
Synthetic observations themselves are regenerated from config and seed rather than
stored as external datasets.

Compare numerical results by domain and representation; do not compare the entire
manifest byte-for-byte because timestamps and environment information differ.
If results differ, compare source/config fingerprints and installed dependencies,
then inspect context-level values before rounded tables.

## Checks used for releases

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest --cov=motifshift --cov-fail-under=85
uv run --locked motifshift --config configs/quick.yaml --robustness --outputs outputs/release-check
uv run --locked motifshift-verify --outputs outputs/release-check --strict --include-robustness
uv run --locked python -m build --no-isolation
```

The regression suite checks sequence boundaries, absence of future-target inputs,
repeatability, invalid configurations, stale-output protection, and checksum failures,
alongside the domain unit tests. A package build and an installation outside the
source directory check that the CLI does not depend on a repository-local import path.
