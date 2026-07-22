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
from .evaluation import DEFAULT_EVALUATION_SPEC, EvaluationSpec, hybrid_tolerance
from .methods import (
    FullGridLeastSquares,
    QRGreedyDOptimalLeastSquares,
    qr_seeded_greedy_d_optimal_order,
    whiten_candidate_design,
)
from .pipeline import run_doptimal_candidate, run_full_grid_reference
from .design_study import run_design_study
from .offgrid import run_offgrid_comparison
from .radial import run_radial_diagnostics
from .validation import run_validation_request, run_validation_scoring
from .yumi import fill_yumi_template, parse_yumi_template

__all__ = [
    "BASIS_ID",
    "BasisIndex",
    "CANDIDATE_BASIS_V1",
    "FullGridLeastSquares",
    "DEFAULT_EVALUATION_SPEC",
    "EvaluationSpec",
    "QRGreedyDOptimalLeastSquares",
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
    "hybrid_tolerance",
    "parse_yumi_template",
    "qr_seeded_greedy_d_optimal_order",
    "radial_potential_matrix",
    "reference_orientations",
    "run_doptimal_candidate",
    "run_design_study",
    "run_full_grid_reference",
    "run_offgrid_comparison",
    "run_radial_diagnostics",
    "run_validation_request",
    "run_validation_scoring",
    "whiten_candidate_design",
]
