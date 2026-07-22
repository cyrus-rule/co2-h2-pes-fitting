"""Common fitting-method interface used by reproducible runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fitting import fit_coefficients


@dataclass(frozen=True)
class FitResult:
    """Numerical result returned by every angular fitting method."""

    method_id: str
    coefficients: NDArray[np.float64]
    selected_orientation_ids: NDArray[np.int64]
    rank: int
    singular_values: NDArray[np.float64]
    condition_number: float
    parameters: dict[str, object]


class AngularFitMethod(Protocol):
    """Protocol separating a fitting method from artifact generation."""

    method_id: str

    def fit(
        self,
        design_matrix: ArrayLike,
        radial_potentials: ArrayLike,
    ) -> FitResult:
        """Fit all radial columns using a shared angular grid."""


@dataclass(frozen=True)
class FullGridLeastSquares:
    """Reference fit using every supplied orientation and raw potential."""

    rcond: float | None = None
    method_id: str = "full500_raw_reference"

    def fit(
        self,
        design_matrix: ArrayLike,
        radial_potentials: ArrayLike,
    ) -> FitResult:
        design = np.asarray(design_matrix, dtype=float)
        potentials = np.asarray(radial_potentials, dtype=float)
        if design.ndim != 2 or potentials.ndim != 2:
            raise ValueError("Design and radial potentials must both be matrices.")
        if design.shape[0] != potentials.shape[0]:
            raise ValueError("Design and potential matrices must share row count.")

        coefficients, rank, singular_values = fit_coefficients(
            design,
            potentials,
            rcond=self.rcond,
        )
        condition_number = (
            float(singular_values[0] / singular_values[-1])
            if singular_values.size and singular_values[-1] > 0.0
            else float("inf")
        )
        return FitResult(
            method_id=self.method_id,
            coefficients=np.asarray(coefficients, dtype=float),
            selected_orientation_ids=np.arange(design.shape[0], dtype=np.int64),
            rank=rank,
            singular_values=np.asarray(singular_values, dtype=float),
            condition_number=condition_number,
            parameters={"solver": "numpy.linalg.lstsq", "rcond": self.rcond},
        )
