"""Tuple-keyed coefficient tables for comparisons and interchange."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .basis import BasisIndex, CANONICAL_BASIS


def basis_table(
    basis: Iterable[BasisIndex] = CANONICAL_BASIS,
) -> pd.DataFrame:
    """Return the basis order together with its scientific tuple identifiers."""

    basis = tuple(basis)
    table = pd.DataFrame(basis, columns=["l1", "l2", "L"])
    table.insert(0, "basis_position", np.arange(len(table), dtype=int))
    if table.duplicated(subset=["l1", "l2", "L"]).any():
        raise ValueError("The basis contains duplicate index tuples.")
    return table


def coefficient_matrix_to_table(
    method: str,
    coefficient_matrix: ArrayLike,
    radial_values: Sequence[float],
    *,
    basis: Iterable[BasisIndex] = CANONICAL_BASIS,
) -> pd.DataFrame:
    """Convert ``(basis, R)`` coefficients to canonical long-form rows."""

    basis = tuple(basis)
    radial_values = np.asarray(radial_values, dtype=float)
    coefficients = np.asarray(coefficient_matrix, dtype=float)
    expected_shape = (len(basis), radial_values.size)
    if coefficients.shape != expected_shape:
        raise ValueError(
            f"{method}: expected coefficient shape {expected_shape}, "
            f"received {coefficients.shape}."
        )
    if not np.isfinite(coefficients).all():
        raise ValueError(f"{method}: coefficients contain NaN or infinity.")

    template = basis_table(basis)
    frames = []
    for radial_index, R in enumerate(radial_values):
        radial_frame = template.copy()
        radial_frame.insert(0, "R", float(R))
        radial_frame.insert(0, "method", method)
        radial_frame["coefficient"] = coefficients[:, radial_index]
        frames.append(radial_frame)
    return pd.concat(frames, ignore_index=True)


def coefficient_table_to_matrix(
    table: pd.DataFrame,
    radial_values: Sequence[float],
    *,
    basis: Iterable[BasisIndex] = CANONICAL_BASIS,
) -> NDArray[np.float64]:
    """Rebuild a coefficient matrix by tuple, independent of table row order."""

    basis = tuple(basis)
    radial_values = np.asarray(radial_values, dtype=float)
    required = {"R", "l1", "l2", "L", "coefficient"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing coefficient-table columns: {sorted(missing)}")
    if table.duplicated(subset=["R", "l1", "l2", "L"]).any():
        raise ValueError("The coefficient table contains duplicate tuple rows.")

    basis_index = pd.MultiIndex.from_tuples(basis, names=["l1", "l2", "L"])
    reconstructed = np.empty((len(basis), radial_values.size), dtype=float)

    table_r = table["R"].to_numpy(dtype=float)
    for radial_index, R in enumerate(radial_values):
        radial_rows = table.loc[np.isclose(table_r, R)]
        coefficients = (
            radial_rows.set_index(["l1", "l2", "L"])["coefficient"]
            .reindex(basis_index)
        )
        if coefficients.isna().any():
            missing_terms = coefficients[coefficients.isna()].index.tolist()
            raise ValueError(f"Missing coefficients at R={R}: {missing_terms[:5]}")
        reconstructed[:, radial_index] = coefficients.to_numpy(dtype=float)

    if not np.isfinite(reconstructed).all():
        raise ValueError("Reconstructed coefficients contain NaN or infinity.")
    return reconstructed

