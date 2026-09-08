# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
"""Verify that expected experiment artifacts exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motifshift.provenance import sha256_file

EXPECTED_RELATIVE_PATHS = [
    "reports/run_summary.md",
    "reports/results_summary.md",
    "data/spatiotemporal_diagnostics_results.json",
    "data/spatiotemporal_transfer_results.json",
    "data/rl_diagnostics_results.json",
    "data/physical_field_results.json",
    "data/combined_results_manifest.json",
    "tables/table_spatiotemporal_diagnostics.tex",
    "tables/table_spatiotemporal_transfer.tex",
    "tables/table_rl_diagnostics.tex",
    "tables/table_rl_transfer.tex",
    "tables/table_physical_field_diagnostics.tex",
    "tables/table_physical_field_transfer.tex",
    "tables/table_combined_diagnostics.tex",
    "tables/table_combined_transfer.tex",
    "figures/spatiotemporal_cross_scene_coverage.pdf",
    "figures/spatiotemporal_transfer_error.pdf",
    "figures/rl_cross_layout_coverage.pdf",
    "figures/rl_next_token_accuracy.pdf",
    "figures/physical_field_cross_env_coverage.pdf",
    "figures/physical_field_transfer_error.pdf",
    "figures/combined_cross_context_coverage.pdf",
]

ROBUSTNESS_RELATIVE_PATHS = [
    "data/multiseed_results.json",
    "tables/table_multiseed_diagnostics.tex",
    "tables/table_multiseed_transfer.tex",
    "reports/multiseed_summary.md",
]


def expected_artifact_paths(outputs: str | Path, include_robustness: bool = False) -> list[Path]:
    """Return expected artifact paths under an output root."""
    output_root = Path(outputs)
    relative_paths = list(EXPECTED_RELATIVE_PATHS)
    if include_robustness:
        relative_paths.extend(ROBUSTNESS_RELATIVE_PATHS)
    return [output_root / relative_path for relative_path in relative_paths]


def verify_outputs(
    outputs: str | Path, include_robustness: bool = False, strict: bool = False
) -> dict[str, list[str] | str]:
    """Check required output artifacts and return a status report."""
    if strict:
        return verify_manifest(outputs, include_robustness=include_robustness)
    checked = [
        str(path)
        for path in expected_artifact_paths(outputs, include_robustness=include_robustness)
    ]
    missing = [
        path for path in checked if not Path(path).is_file() or Path(path).stat().st_size == 0
    ]
    return {
        "status": "ok" if not missing else "missing",
        "missing": missing,
        "checked": checked,
    }


def verify_manifest(
    outputs: str | Path, include_robustness: bool = False
) -> dict[str, list[str] | str]:
    """Validate a portable run inventory; hashes detect changes, not authenticity."""
    root = Path(outputs).resolve()
    errors: list[str] = []
    missing: list[str] = []
    checked: list[str] = []
    candidates = list(root.rglob("combined_results_manifest.json"))
    try:
        if len(candidates) != 1:
            raise ValueError("expected exactly one combined_results_manifest.json")
        manifest_path = candidates[0]
        if not manifest_path.resolve().is_relative_to(root):
            raise ValueError("manifest points outside the output directory")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported manifest schema_version")
        expected = manifest["expected_artifacts"]
        groups = manifest["artifacts"]
        hashes = manifest["sha256"]
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(p, str) for p in expected)
        ):
            raise ValueError("expected_artifacts must be a non-empty list of paths")
        if len(expected) != len(set(expected)):
            raise ValueError("duplicate artifact paths")
        if not isinstance(groups, dict) or not all(isinstance(v, list) for v in groups.values()):
            raise ValueError("artifacts must contain grouped path lists")
        if set(expected) != {p for paths in groups.values() for p in paths}:
            raise ValueError("artifact groups and expected inventory disagree")
        own_path = manifest_path.relative_to(root).as_posix()
        if own_path not in expected or set(hashes) != set(expected) - {own_path}:
            raise ValueError("checksum inventory must cover every artifact except the manifest")
        if manifest.get("missing_artifacts"):
            errors.append("run manifest records missing artifacts")
        if include_robustness and not groups.get("robustness"):
            errors.append("multi-seed robustness artifacts are required")
        for relative in expected:
            path = root / relative
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not path.resolve().is_relative_to(root)
            ):
                raise ValueError("artifact path escapes the output directory")
            checked.append(str(path))
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
                continue
            if relative != own_path and sha256_file(path) != hashes[relative]:
                errors.append(f"checksum mismatch: {relative}")
            if path.suffix == ".json":

                def invalid_constant(value: str) -> None:
                    raise ValueError(f"non-finite JSON value: {value}")

                json.loads(path.read_text(), parse_constant=invalid_constant)
            elif path.suffix == ".pdf" and not path.read_bytes().startswith(b"%PDF-"):
                errors.append(f"invalid PDF header: {relative}")
            elif path.suffix == ".png" and not path.read_bytes().startswith(
                bytes([137, 80, 78, 71, 13, 10, 26, 10])
            ):
                errors.append(f"invalid PNG header: {relative}")
        config = manifest["provenance"]["config"]
        data_dir = config.get("outputs", {}).get("data_dir", "data")
        snapshot = root / data_dir / "config.json"
        if (
            not snapshot.resolve().is_relative_to(root)
            or snapshot.relative_to(root).as_posix() not in expected
        ):
            raise ValueError("configuration snapshot missing from inventory")
        if (
            json.loads(snapshot.read_text()) != config
            or sha256_file(snapshot) != manifest["provenance"]["config_sha256"]
        ):
            errors.append("configuration snapshot disagrees with provenance")
        for kind, filename in [("reports", "run_summary.md"), ("reports", "results_summary.md")]:
            relative = (
                Path(config.get("outputs", {}).get(kind + "_dir", kind)) / filename
            ).as_posix()
            if relative not in expected:
                errors.append(f"missing required inventory entry: {relative}")
        # Require enabled domain results independently of the recorded artifact list.
        required_results = []
        st = config.get("spatiotemporal", {})
        if st.get("enabled"):
            for section in ["diagnostics", "transfer"]:
                if st.get(section, {}).get("enabled"):
                    required_results.append(f"spatiotemporal_{section}_results.json")
        for domain, filename in [
            ("rl", "rl_diagnostics_results.json"),
            ("physical_field", "physical_field_results.json"),
        ]:
            section = config.get(domain, {})
            if section.get("enabled") and (
                section.get("diagnostics", {}).get("enabled")
                or section.get("transfer", {}).get("enabled")
            ):
                required_results.append(filename)
        if config.get("robustness", {}).get("enabled"):
            required_results.append("multiseed_results.json")
        for filename in required_results:
            if (Path(data_dir) / filename).as_posix() not in expected:
                errors.append(f"missing required domain result: {filename}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        errors.append(f"invalid run manifest or artifact: {exc}")
    return {
        "status": "missing" if missing else ("invalid" if errors else "ok"),
        "missing": missing,
        "checked": checked,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify generated experiment artifacts.")
    parser.add_argument("--outputs", required=True, help="Output directory root.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text report.",
    )
    parser.add_argument(
        "--include-robustness",
        action="store_true",
        help="Also require optional multi-seed robustness artifacts.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Verify the run manifest, configuration, file contents, and SHA-256 checksums.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = verify_outputs(
        args.outputs, include_robustness=args.include_robustness, strict=args.strict
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Verification status: {report['status']}")
        print(f"Checked files: {len(report['checked'])}")
        for error in report.get("errors", []):
            print(f"Error: {error}")
        if report["missing"]:
            print("Missing files:")
            for path in report["missing"]:
                print(f"- {path}")
        elif report["status"] == "ok":
            print(
                "All required artifacts are present."
                + (" Checksums verified." if args.strict else "")
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
