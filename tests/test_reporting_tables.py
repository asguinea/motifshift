# SPDX-FileCopyrightText: 2026 Alejandro Sánchez Guinea
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from motifshift.reporting.tables import format_float, write_csv, write_latex_table


def test_format_float() -> None:
    assert format_float(1 / 3) == "0.333"
    assert format_float(1 / 3, digits=2) == "0.33"


def test_write_latex_table_contains_caption_label_and_escaped_underscores(
    tmp_path: Path,
) -> None:
    path = tmp_path / "table.tex"

    write_latex_table(
        path,
        caption="Example caption",
        label="tab:example_table",
        columns=["Representation", "Score"],
        rows=[["absolute_grid", "0.500"]],
    )

    content = path.read_text(encoding="utf-8")
    assert r"\begin{table}" in content
    assert r"\centering" in content
    assert r"\caption{Example caption}" in content
    assert r"\label{tab:example_table}" in content
    assert "absolute\\_grid" in content


def test_write_csv_writes_header_and_row(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"

    write_csv(path, [{"Representation": "Motion", "Score": "1.000"}])

    content = path.read_text(encoding="utf-8")
    assert "Representation,Score" in content
    assert "Motion,1.000" in content
