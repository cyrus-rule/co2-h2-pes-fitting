"""Shared target transformations and linear least-squares fits."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

CURRENT_SWITCH_T1 = 1000.0
CURRENT_SWITCH_T2 = 5000.0
DEFAULT_LSTSQ_RCOND = None


def apply_legacy_switch(
    potential: ArrayLike,
    *,
    lower_threshold: float = CURRENT_SWITCH_T1,
    upper_threshold: float = CURRENT_SWITCH_T2,
) -> NDArray[np.float64]:
    """Apply the legacy bounded high-energy target transformation.

    The mapping is continuous but is not C1 at ``lower_threshold``. It is
    retained to reproduce a negative-result experiment, not as a production
    default.
    """

    if upper_threshold <= lower_threshold:
        raise ValueError("upper_threshold must be greater than lower_threshold.")

    potential = np.asarray(potential, dtype=float)
    switched = np.empty_like(potential)
    below = potential <= lower_threshold
    above = potential > upper_threshold
    middle = ~(below | above)

    switched[below] = potential[below]
    cap = lower_threshold + (2.0 / np.pi) * (
        upper_threshold - lower_threshold
    )
    switched[above] = cap

    u = (potential[middle] - lower_threshold) / (
        upper_threshold - lower_threshold
    )
    smooth_u = (2.0 / np.pi) * np.sin(
        (np.pi / 2.0) * np.sin((np.pi / 2.0) * u)
    )
    switched[middle] = lower_threshold + (
        upper_threshold - lower_threshold
    ) * smooth_u
    return switched


def apply_current_switch(
    potential: ArrayLike,
    *,
    lower_threshold: float = CURRENT_SWITCH_T1,
    upper_threshold: float = CURRENT_SWITCH_T2,
) -> NDArray[np.float64]:
    """Compatibility alias for :func:`apply_legacy_switch`."""

    return apply_legacy_switch(
        potential,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )


def fit_coefficients(
    design_matrix: ArrayLike,
    potential: ArrayLike,
    *,
    rcond: float | None = DEFAULT_LSTSQ_RCOND,
) -> tuple[NDArray[np.float64], int, NDArray[np.float64]]:
    """Fit one or several potential columns by linear least squares.

    ``rcond=None`` uses NumPy's neutral machine-precision rank threshold. Any
    truncated-SVD policy must pass and record a numerical cutoff explicitly.
    """

    design_matrix = np.asarray(design_matrix, dtype=float)
    potential = np.asarray(potential, dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design_matrix,
        potential,
        rcond=rcond,
    )
    return coefficients, int(rank), singular_values


def fit_radial_coefficients(
    design_matrix: ArrayLike,
    radial_potentials: ArrayLike,
    selected_ids: ArrayLike | None = None,
    *,
    rcond: float | None = DEFAULT_LSTSQ_RCOND,
) -> tuple[NDArray[np.float64], int, float]:
    """Fit a coefficient matrix with one potential column per radial slice."""

    design_matrix = np.asarray(design_matrix, dtype=float)
    radial_potentials = np.asarray(radial_potentials, dtype=float)
    if selected_ids is not None:
        selected_ids = np.asarray(selected_ids, dtype=int)
        design_matrix = design_matrix[selected_ids]
        radial_potentials = radial_potentials[selected_ids]

    coefficients, rank, singular_values = fit_coefficients(
        design_matrix,
        radial_potentials,
        rcond=rcond,
    )
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else np.inf
    )
    return coefficients, rank, condition_number

