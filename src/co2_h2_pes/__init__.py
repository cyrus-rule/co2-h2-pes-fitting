"""Core tools for the CO2-H2 potential-energy-surface fits."""

from .basis import (
    BASIS_ID,
    BasisIndex,
    CANDIDATE_BASIS_V1,
    build_design_matrix,
    evaluate_angular_basis,
    generate_candidate_basis_v1,
    generate_basis_indices,
)
from .coefficients import coefficient_matrix_to_table, coefficient_table_to_matrix
from .data import load_ab_initio_data, radial_potential_matrix, reference_orientations
from .fitting import (
    apply_current_switch,
    apply_legacy_switch,
    fit_coefficients,
    fit_radial_coefficients,
)
from .methods import FullGridLeastSquares
from .pipeline import run_full_grid_reference
from .yumi import fill_yumi_template, parse_yumi_template

__all__ = [
    "BASIS_ID",
    "BasisIndex",
    "CANDIDATE_BASIS_V1",
    "FullGridLeastSquares",
    "apply_current_switch",
    "apply_legacy_switch",
    "build_design_matrix",
    "coefficient_matrix_to_table",
    "coefficient_table_to_matrix",
    "evaluate_angular_basis",
    "fill_yumi_template",
    "fit_coefficients",
    "fit_radial_coefficients",
    "generate_candidate_basis_v1",
    "generate_basis_indices",
    "load_ab_initio_data",
    "parse_yumi_template",
    "radial_potential_matrix",
    "reference_orientations",
    "run_full_grid_reference",
]
