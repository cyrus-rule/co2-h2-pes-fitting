"""Structural parser and coefficient filler for supplied YUMI templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .basis import BasisIndex

_INDEX_LINE = re.compile(r"^\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)\s*$")


def _float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


@dataclass(frozen=True)
class YumiTemplate:
    """Parsed template structure; masked input coefficients may be NaN."""

    title: str
    header_lines: tuple[str, ...]
    radii: NDArray[np.float64]
    terms: tuple[BasisIndex, ...]
    coefficients: NDArray[np.float64]

    def validate_terms(self, allowed_basis: tuple[BasisIndex, ...]) -> None:
        unknown = sorted(set(self.terms).difference(allowed_basis))
        if unknown:
            raise ValueError(f"YUMI template contains unknown terms: {unknown[:5]}")

    def with_coefficients(self, coefficients: NDArray[np.float64]) -> "YumiTemplate":
        values = np.asarray(coefficients, dtype=float)
        if values.shape != self.coefficients.shape:
            raise ValueError(
                f"Expected coefficient shape {self.coefficients.shape}, got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("YUMI output coefficients must be finite; NaN is forbidden.")
        return YumiTemplate(
            title=self.title,
            header_lines=self.header_lines,
            radii=self.radii.copy(),
            terms=self.terms,
            coefficients=values,
        )

    def to_text(self) -> str:
        if not np.isfinite(self.coefficients).all():
            raise ValueError("Cannot serialize masked, NaN, or infinite coefficients.")
        lines = [self.title, *self.header_lines]
        lines.append(" ".join(_format_number(value) for value in self.radii))
        for (l1, l2, L), values in zip(self.terms, self.coefficients, strict=True):
            lines.append(f"{l1:d} 0 {l2:d} {L:d}")
            lines.append(" ".join(_format_number(value) for value in values))
        return "\n".join(lines) + "\n"


def _format_number(value: float) -> str:
    return "0.0" if float(value) == 0.0 else f"{float(value):.16e}"


def parse_yumi_template(
    path: str | Path,
    *,
    radial_count: int,
    header_line_count: int = 1,
) -> YumiTemplate:
    """Parse a template without assuming the unresolved header semantics.

    ``header_line_count`` lines after the title are preserved verbatim. The
    caller supplies ``radial_count`` because the observed header's ``68`` may
    count anisotropic terms rather than the radial grid.
    """

    if radial_count < 1 or header_line_count < 0:
        raise ValueError("radial_count must be positive and header_line_count nonnegative.")
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2 + header_line_count:
        raise ValueError("YUMI template is too short.")
    title = lines[0]
    header_lines = tuple(lines[1 : 1 + header_line_count])
    cursor = 1 + header_line_count

    first_index = None
    for position in range(cursor, len(lines)):
        match = _INDEX_LINE.match(lines[position])
        if match and int(match.group(2)) == 0:
            first_index = position
            break
    if first_index is None:
        raise ValueError("No YUMI four-index coefficient block was found.")

    radial_tokens = " ".join(lines[cursor:first_index]).split()
    if len(radial_tokens) != radial_count:
        raise ValueError(
            f"Expected {radial_count} radial values, found {len(radial_tokens)}."
        )
    radii = np.array([_float(token) for token in radial_tokens], dtype=float)
    if not np.isfinite(radii).all():
        raise ValueError("Radial grid contains NaN or infinity.")
    if np.unique(radii).size != radii.size:
        raise ValueError("Radial grid contains duplicate values.")

    terms: list[BasisIndex] = []
    blocks: list[np.ndarray] = []
    cursor = first_index
    while cursor < len(lines):
        match = _INDEX_LINE.match(lines[cursor])
        if not match or int(match.group(2)) != 0:
            raise ValueError(f"Expected a four-index YUMI row at line {cursor + 1}.")
        term = (int(match.group(1)), int(match.group(3)), int(match.group(4)))
        if term in terms:
            raise ValueError(f"Duplicate YUMI term {term}.")
        terms.append(term)
        cursor += 1

        tokens: list[str] = []
        while cursor < len(lines) and len(tokens) < radial_count:
            tokens.extend(lines[cursor].split())
            cursor += 1
        if len(tokens) != radial_count:
            raise ValueError(
                f"Term {term} has {len(tokens)} coefficients; expected {radial_count}."
            )
        values = []
        for token in tokens:
            try:
                values.append(_float(token))
            except ValueError:
                values.append(float("nan"))
        blocks.append(np.asarray(values, dtype=float))

    return YumiTemplate(
        title=title,
        header_lines=header_lines,
        radii=radii,
        terms=tuple(terms),
        coefficients=np.vstack(blocks),
    )


def fill_yumi_template(
    template: YumiTemplate,
    coefficient_table: pd.DataFrame,
    *,
    method: str | None = None,
) -> YumiTemplate:
    """Fill required YUMI blocks by tuple, writing omitted terms as exact zero."""

    table = coefficient_table.copy()
    if "method" in table.columns:
        methods = table["method"].dropna().unique().tolist()
        if method is None:
            if len(methods) != 1:
                raise ValueError("Specify method when the coefficient table has multiple methods.")
            method = str(methods[0])
        table = table.loc[table["method"] == method]
    required = {"R", "l1", "l2", "L", "coefficient"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing coefficient columns: {sorted(missing)}")
    if not np.isfinite(table["coefficient"].to_numpy(dtype=float)).all():
        raise ValueError("Coefficient table contains NaN or infinity.")
    table_radii = table["R"].to_numpy(dtype=float)
    missing_radii = [
        float(radius)
        for radius in template.radii
        if not np.isclose(table_radii, radius).any()
    ]
    if missing_radii:
        raise ValueError(f"Coefficient table is missing YUMI radii: {missing_radii}")

    result = np.zeros((len(template.terms), template.radii.size), dtype=float)
    for row, term in enumerate(template.terms):
        term_rows = table.loc[
            (table["l1"] == term[0])
            & (table["l2"] == term[1])
            & (table["L"] == term[2])
        ]
        if term_rows.empty:
            continue
        for column, radius in enumerate(template.radii):
            matches = term_rows.loc[
                np.isclose(term_rows["R"].to_numpy(dtype=float), radius)
            ]
            if len(matches) > 1:
                raise ValueError(f"Duplicate coefficient for term {term} at R={radius}.")
            if len(matches) == 1:
                result[row, column] = float(matches.iloc[0]["coefficient"])
    return template.with_coefficients(result)
