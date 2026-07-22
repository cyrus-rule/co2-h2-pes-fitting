"""Angular basis conventions used by the CO2-H2 fits."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import lpmv
from sympy.physics.wigner import wigner_3j

BasisIndex: TypeAlias = tuple[int, int, int]


@lru_cache(maxsize=None)
def cached_wigner_3j(
    l1: int,
    l2: int,
    L: int,
    m1: int,
    m2: int,
    m3: int,
) -> float:
    """Return a cached floating-point Wigner 3-j value."""

    return float(wigner_3j(l1, l2, L, m1, m2, m3))


def normalized_associated_legendre(
    l: int,
    m: int,
    x: ArrayLike,
) -> NDArray[np.float64]:
    """Evaluate the associated Legendre function with Appendix C7 normalization."""

    normalization = np.sqrt(
        ((2 * l + 1) / 2.0)
        * math.factorial(l - m)
        / math.factorial(l + m)
    )
    return normalization * lpmv(m, l, np.asarray(x, dtype=float))


def generate_candidate_basis_v1() -> tuple[BasisIndex, ...]:
    """Return the current candidate 158-term ``(l1, l2, L)`` universe.

    The bounds and high-order truncation reproduce the convention used by the
    July 2026 research notebook. The published methodology does not fully list
    its omitted high-order terms, so this basis is deliberately versioned and
    labelled *candidate* until checked against a trusted production term list.
    Reduced-basis methods should select subsets from this universe rather than
    silently redefining it.
    """

    basis: list[BasisIndex] = []
    for l1 in range(0, 25, 2):
        for l2 in range(0, 7, 2):
            if l1 >= 22 and l2 > 0:
                continue
            for L in range(abs(l1 - l2), l1 + l2 + 1):
                if (l1 + l2 + L) % 2 == 0:
                    basis.append((l1, l2, L))
    return tuple(basis)


BASIS_ID = "co2-h2-candidate-158-v1"
CANDIDATE_BASIS_V1 = generate_candidate_basis_v1()

# Compatibility aliases for the original notebook extraction. New code should
# use the explicit versioned names above so provisional status is visible.
CANONICAL_BASIS = CANDIDATE_BASIS_V1


def generate_basis_indices() -> tuple[BasisIndex, ...]:
    """Compatibility wrapper for :func:`generate_candidate_basis_v1`."""

    return generate_candidate_basis_v1()


def evaluate_angular_basis(
    l1: int,
    l2: int,
    L: int,
    theta1_rad: ArrayLike,
    theta2_rad: ArrayLike,
    phi_rad: ArrayLike,
) -> NDArray[np.float64]:
    """Evaluate one normalized coupled angular basis function."""

    theta1_rad = np.asarray(theta1_rad, dtype=float)
    theta2_rad = np.asarray(theta2_rad, dtype=float)
    phi_rad = np.asarray(phi_rad, dtype=float)
    cos_t1 = np.cos(theta1_rad)
    cos_t2 = np.cos(theta2_rad)

    wj_zero = cached_wigner_3j(l1, l2, L, 0, 0, 0)
    p_l1_0 = normalized_associated_legendre(l1, 0, cos_t1)
    p_l2_0 = normalized_associated_legendre(l2, 0, cos_t2)
    total = ((-1) ** (l1 - l2)) * wj_zero * p_l1_0 * p_l2_0

    for m in range(1, min(l1, l2) + 1):
        wj_m = cached_wigner_3j(l1, l2, L, m, -m, 0)
        p_l1 = normalized_associated_legendre(l1, m, cos_t1)
        p_l2 = normalized_associated_legendre(l2, m, cos_t2)
        total += (
            2.0
            * ((-1) ** (m + l1 - l2))
            * wj_m
            * p_l1
            * p_l2
            * np.cos(m * phi_rad)
        )

    return np.sqrt((2 * L + 1) / (4.0 * np.pi)) * total


def build_design_matrix(
    theta1_rad: ArrayLike,
    theta2_rad: ArrayLike,
    phi_rad: ArrayLike,
    basis: Iterable[BasisIndex] = CANDIDATE_BASIS_V1,
) -> NDArray[np.float64]:
    """Construct the angular design matrix for vectorized orientations."""

    theta1_rad = np.asarray(theta1_rad, dtype=float)
    theta2_rad = np.asarray(theta2_rad, dtype=float)
    phi_rad = np.asarray(phi_rad, dtype=float)
    if not (theta1_rad.shape == theta2_rad.shape == phi_rad.shape):
        raise ValueError("All three angular arrays must have the same shape.")
    if theta1_rad.ndim != 1:
        raise ValueError("Angular arrays must be one-dimensional.")

    basis = tuple(basis)
    matrix = np.empty((theta1_rad.size, len(basis)), dtype=float)
    for column, (l1, l2, L) in enumerate(basis):
        matrix[:, column] = evaluate_angular_basis(
            l1,
            l2,
            L,
            theta1_rad,
            theta2_rad,
            phi_rad,
        )
    return matrix

