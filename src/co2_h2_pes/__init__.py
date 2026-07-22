"""Core tools for the CO2-H2 potential-energy-surface fits."""

from .basis import (
    BasisIndex,
    build_design_matrix,
    evaluate_angular_basis,
    generate_basis_indices,
)
from .coefficients import coefficient_matrix_to_table, coefficient_table_to_matrix
from .data import load_ab_initio_data, reference_orientations
from .fitting import apply_current_switch, fit_coefficients, fit_radial_coefficients

__all__ = [
    "BasisIndex",
    "apply_current_switch",
    "build_design_matrix",
    "coefficient_matrix_to_table",
    "coefficient_table_to_matrix",
    "evaluate_angular_basis",
    "fit_coefficients",
    "fit_radial_coefficients",
    "generate_basis_indices",
    "load_ab_initio_data",
    "reference_orientations",
]

