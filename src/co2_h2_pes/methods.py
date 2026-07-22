"""Common fitting-method interface used by reproducible runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.linalg import qr
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


def _validated_fit_matrices(
    design_matrix: ArrayLike,
    radial_potentials: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return finite, shape-compatible design and target matrices."""

    design = np.asarray(design_matrix, dtype=float)
    potentials = np.asarray(radial_potentials, dtype=float)
    if design.ndim != 2 or potentials.ndim != 2:
        raise ValueError("Design and radial potentials must both be matrices.")
    if design.shape[0] != potentials.shape[0]:
        raise ValueError("Design and potential matrices must share row count.")
    if not np.isfinite(design).all() or not np.isfinite(potentials).all():
        raise ValueError("Design and potential matrices must be finite.")
    return design, potentials


def whiten_candidate_design(design_matrix: ArrayLike) -> NDArray[np.float64]:
    """Whiten columns against the complete candidate information matrix.

    For the returned matrix ``W``, ``W.T @ W`` is the identity up to floating-
    point error. Whitening makes row selection invariant to nonsingular linear
    rescalings of the fitted basis columns.
    """

    design = np.asarray(design_matrix, dtype=float)
    if design.ndim != 2:
        raise ValueError("Candidate design must be a matrix.")
    if not np.isfinite(design).all():
        raise ValueError("Candidate design must be finite.")
    n_candidates, n_parameters = design.shape
    if n_candidates < n_parameters:
        raise ValueError("Fewer candidate rows than fitted parameters.")

    information = design.T @ design
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tolerance = (
        np.finfo(float).eps * max(design.shape) * scale
        if scale > 0.0
        else 0.0
    )
    if eigenvalues.size == 0 or float(eigenvalues[0]) <= tolerance:
        raise np.linalg.LinAlgError(
            "Candidate design matrix is not full column rank."
        )

    inverse_square_root = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    return np.asarray(design @ inverse_square_root, dtype=float)


def qr_seeded_greedy_d_optimal_order(
    design_matrix: ArrayLike,
    *,
    point_count: int | None = None,
    recompute_interval: int = 50,
) -> NDArray[np.int64]:
    """Select a deterministic nested D-optimal-style row ordering.

    Column-pivoted QR of the whitened transpose supplies one full-rank row per
    parameter. Further rows maximize predictive variance, equivalently the
    immediate information-determinant gain. This is a greedy approximation;
    it is not a claim of globally exact D-optimality.
    """

    whitened = whiten_candidate_design(design_matrix)
    n_candidates, n_parameters = whitened.shape
    if point_count is None:
        point_count = n_candidates
    if not n_parameters <= point_count <= n_candidates:
        raise ValueError(
            "point_count must be between the fitted parameter count and the "
            "candidate row count."
        )
    if recompute_interval <= 0:
        raise ValueError("recompute_interval must be positive.")

    _, _, pivot_order = qr(whitened.T, pivoting=True, mode="economic")
    selected_order = list(np.asarray(pivot_order[:n_parameters], dtype=int))
    selected_mask = np.zeros(n_candidates, dtype=bool)
    selected_mask[selected_order] = True

    selected_design = whitened[np.asarray(selected_order, dtype=int)]
    information_inverse = np.linalg.inv(selected_design.T @ selected_design)
    additions = point_count - n_parameters
    for addition in range(additions):
        predictive_variance = np.einsum(
            "ij,jk,ik->i",
            whitened,
            information_inverse,
            whitened,
        )
        predictive_variance[selected_mask] = -np.inf
        next_index = int(np.argmax(predictive_variance))
        selected_order.append(next_index)
        selected_mask[next_index] = True

        next_row = whitened[next_index]
        inverse_times_row = information_inverse @ next_row
        denominator = float(1.0 + next_row @ inverse_times_row)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise np.linalg.LinAlgError(
                "D-optimal rank-one information update became unstable."
            )
        information_inverse = information_inverse - (
            np.outer(inverse_times_row, inverse_times_row) / denominator
        )

        if (
            (addition + 1) % recompute_interval == 0
            and addition + 1 < additions
        ):
            selected_design = whitened[np.asarray(selected_order, dtype=int)]
            information_inverse = np.linalg.inv(
                selected_design.T @ selected_design
            )

    return np.asarray(selected_order, dtype=np.int64)


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
        design, potentials = _validated_fit_matrices(
            design_matrix,
            radial_potentials,
        )

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


@dataclass(frozen=True)
class QRGreedyDOptimalLeastSquares:
    """Raw-potential fit on an energy-blind D-optimal-style angular subset."""

    point_count: int = 180
    rcond: float | None = None
    recompute_interval: int = 50

    @property
    def method_id(self) -> str:
        return f"doptimal{self.point_count}_candidate"

    def fit(
        self,
        design_matrix: ArrayLike,
        radial_potentials: ArrayLike,
    ) -> FitResult:
        design, potentials = _validated_fit_matrices(
            design_matrix,
            radial_potentials,
        )
        selected = qr_seeded_greedy_d_optimal_order(
            design,
            point_count=self.point_count,
            recompute_interval=self.recompute_interval,
        )
        fit_design = design[selected]
        coefficients, rank, singular_values = fit_coefficients(
            fit_design,
            potentials[selected],
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
            selected_orientation_ids=selected,
            rank=rank,
            singular_values=np.asarray(singular_values, dtype=float),
            condition_number=condition_number,
            parameters={
                "solver": "numpy.linalg.lstsq",
                "rcond": self.rcond,
                "point_count": self.point_count,
                "selection_algorithm": "qr_seeded_greedy_doptimal_style_v1",
                "selection_uses_potential_energies": False,
                "qr_seed_count": int(design.shape[1]),
                "recompute_interval": self.recompute_interval,
            },
        )
