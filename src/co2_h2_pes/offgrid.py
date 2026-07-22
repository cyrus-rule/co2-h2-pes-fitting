"""Deterministic off-grid model--model stress diagnostics.

These diagnostics compare fitted surfaces where no independent ab initio label
exists. They are useful for constructing a prospective validation request, but
their outputs are never labelled as validation errors.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import qmc

from .artifacts import (
    RUN_SCHEMA_VERSION,
    current_git_revision,
    software_versions,
    utc_timestamp,
    write_json,
)
from .basis import CANDIDATE_BASIS_V1, build_design_matrix
from .coefficients import coefficient_table_to_matrix
from .data import sha256_file
from .evaluation import DEFAULT_EVALUATION_SPEC, EvaluationSpec, hybrid_tolerance

OFFGRID_DIAGNOSTIC_ID = "sobol-model-disagreement-v1"
DEFAULT_SAMPLE_SIZE = 4096
DEFAULT_SEED = 20260715
DEFAULT_MARGIN = 1.0e-3
DEFAULT_DERIVATIVE_STEP = 1.0e-4
DEFAULT_ANCHOR_RADII = (4.4, 4.8, 5.0, 5.8, 6.6)
DEFAULT_MINIMUM_RADII = (4.4, 4.8, 5.0)


@dataclass(frozen=True)
class OffGridGrid:
    """One deterministic angular stress grid and its numerical derivatives."""

    mode: str
    theta1: NDArray[np.float64]
    theta2: NDArray[np.float64]
    phi: NDArray[np.float64]
    design: NDArray[np.float64]
    derivatives: tuple[NDArray[np.float64], ...]


@dataclass(frozen=True)
class OffGridStressResult:
    """Tables and deterministic grids required by the validation generator."""

    summary: pd.DataFrame
    candidates: pd.DataFrame
    local_minima: pd.DataFrame
    grids: dict[str, OffGridGrid]


def make_offgrid_angles(
    mode: str,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    margin: float = DEFAULT_MARGIN,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return a scrambled Sobol grid under one of two angular measures."""

    exponent = int(np.log2(sample_size))
    if sample_size < 1 or 2**exponent != sample_size:
        raise ValueError("Sobol sample_size must be a positive power of two.")
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5).")
    seed_offset = 0 if mode == "coordinate-uniform" else 1
    unit = qmc.Sobol(d=3, scramble=True, seed=seed + seed_offset).random_base2(
        exponent
    )
    unit = margin + (1.0 - 2.0 * margin) * unit

    if mode == "coordinate-uniform":
        theta1 = np.pi * unit[:, 0]
        theta2 = np.pi * unit[:, 1]
    elif mode == "solid-angle":
        theta1 = np.arccos(1.0 - 2.0 * unit[:, 0])
        theta2 = np.arccos(1.0 - 2.0 * unit[:, 1])
    else:
        raise ValueError(f"Unknown off-grid sampling mode: {mode}")
    return theta1, theta2, np.pi * unit[:, 2]


def offgrid_grid(
    mode: str,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    margin: float = DEFAULT_MARGIN,
    derivative_step: float = DEFAULT_DERIVATIVE_STEP,
) -> OffGridGrid:
    """Build the off-grid design and centered finite-difference derivatives."""

    if derivative_step <= 0.0:
        raise ValueError("derivative_step must be positive.")
    coordinates = make_offgrid_angles(
        mode,
        sample_size=sample_size,
        seed=seed,
        margin=margin,
    )
    design = build_design_matrix(*coordinates)
    derivatives: list[NDArray[np.float64]] = []
    for coordinate_index in range(3):
        plus = [values.copy() for values in coordinates]
        minus = [values.copy() for values in coordinates]
        plus[coordinate_index] += derivative_step
        minus[coordinate_index] -= derivative_step
        derivatives.append(
            (build_design_matrix(*plus) - build_design_matrix(*minus))
            / (2.0 * derivative_step)
        )
    return OffGridGrid(
        mode=mode,
        theta1=coordinates[0],
        theta2=coordinates[1],
        phi=coordinates[2],
        design=design,
        derivatives=tuple(derivatives),
    )


def _radial_index(radii: NDArray[np.float64], radius: float) -> int:
    matches = np.flatnonzero(np.isclose(radii, radius))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one coefficient column at R={radius}.")
    return int(matches[0])


def _single_value(coefficients: NDArray[np.float64], angles: NDArray[np.float64]) -> float:
    design = build_design_matrix(
        np.array([angles[0]]),
        np.array([angles[1]]),
        np.array([angles[2]]),
    )
    return float(design[0] @ coefficients)


def compare_coefficient_surfaces_offgrid(
    radii: NDArray[np.float64],
    reference_coefficients: NDArray[np.float64],
    candidate_coefficients: NDArray[np.float64],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    margin: float = DEFAULT_MARGIN,
    derivative_step: float = DEFAULT_DERIVATIVE_STEP,
    anchor_radii: tuple[float, ...] = DEFAULT_ANCHOR_RADII,
    minimum_radii: tuple[float, ...] = DEFAULT_MINIMUM_RADII,
    evaluation_spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> OffGridStressResult:
    """Compare two coefficient surfaces on deterministic unlabelled angles."""

    radii = np.asarray(radii, dtype=float)
    reference_coefficients = np.asarray(reference_coefficients, dtype=float)
    candidate_coefficients = np.asarray(candidate_coefficients, dtype=float)
    expected_shape = (len(CANDIDATE_BASIS_V1), len(radii))
    if reference_coefficients.shape != expected_shape:
        raise ValueError("Reference coefficient matrix has an incompatible shape.")
    if candidate_coefficients.shape != expected_shape:
        raise ValueError("Candidate coefficient matrix has an incompatible shape.")

    grids = {
        mode: offgrid_grid(
            mode,
            sample_size=sample_size,
            seed=seed,
            margin=margin,
            derivative_step=derivative_step,
        )
        for mode in ("coordinate-uniform", "solid-angle")
    }
    summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for mode, grid in grids.items():
        for radius in anchor_radii:
            column = _radial_index(radii, radius)
            reference_vector = reference_coefficients[:, column]
            candidate_vector = candidate_coefficients[:, column]
            reference_values = grid.design @ reference_vector
            candidate_values = grid.design @ candidate_vector
            absolute = np.abs(candidate_values - reference_values)
            normalized = absolute / hybrid_tolerance(reference_values, evaluation_spec)
            reference_gradient = np.column_stack(
                [derivative @ reference_vector for derivative in grid.derivatives]
            )
            candidate_gradient = np.column_stack(
                [derivative @ candidate_vector for derivative in grid.derivatives]
            )
            gradient = np.linalg.norm(candidate_gradient - reference_gradient, axis=1)
            severe = (
                (reference_values >= evaluation_spec.physical_ceiling_cm1)
                & (candidate_values < evaluation_spec.guardrail_warning_cm1)
            )
            low_crossing = (
                (reference_values >= evaluation_spec.low_repulsive_upper_cm1)
                & (candidate_values < evaluation_spec.low_repulsive_upper_cm1)
            )
            summary_rows.append(
                {
                    "sampling_measure": mode,
                    "R": radius,
                    "points": len(reference_values),
                    "p95_absolute_disagreement": np.percentile(absolute, 95),
                    "maximum_absolute_disagreement": absolute.max(),
                    "p95_normalized_disagreement": np.percentile(normalized, 95),
                    "maximum_normalized_disagreement": normalized.max(),
                    "p95_gradient_disagreement_cm1_per_rad": np.percentile(
                        gradient, 95
                    ),
                    "maximum_gradient_disagreement_cm1_per_rad": gradient.max(),
                    "reference_high_candidate_below_warning": int(severe.sum()),
                    "reference_above_1000_candidate_below_1000": int(
                        low_crossing.sum()
                    ),
                }
            )

            reasons: dict[int, set[str]] = {}

            def add(indices: NDArray[np.int64], reason: str) -> None:
                for index in np.asarray(indices, dtype=int):
                    reasons.setdefault(int(index), set()).add(reason)

            add(np.flatnonzero(severe), "reference-high / candidate-below-3000")
            add(np.flatnonzero(low_crossing), "reference-above-1000 / candidate-below-1000")
            add(np.argsort(normalized)[-5:], "largest normalized disagreement")
            add(np.argsort(absolute)[-5:], "largest absolute disagreement")
            add(np.argsort(gradient)[-5:], "largest gradient disagreement")

            for index, row_reasons in reasons.items():
                candidate_rows.append(
                    {
                        "sampling_measure": mode,
                        "R": radius,
                        "Theta_1": np.degrees(grid.theta1[index]),
                        "Theta_2": np.degrees(grid.theta2[index]),
                        "Phi": np.degrees(grid.phi[index]),
                        "V_reference": reference_values[index],
                        "V_candidate": candidate_values[index],
                        "absolute_disagreement": absolute[index],
                        "normalized_disagreement": normalized[index],
                        "gradient_disagreement_cm1_per_rad": gradient[index],
                        "reference_high_candidate_below_warning": bool(severe[index]),
                        "reference_above_1000_candidate_below_1000": bool(
                            low_crossing[index]
                        ),
                        "selection_reason": "; ".join(sorted(row_reasons)),
                    }
                )

    minimum_rows: list[dict[str, object]] = []
    coordinate_grid = grids["coordinate-uniform"]
    bounds = [(0.0, np.pi), (0.0, np.pi), (0.0, np.pi)]
    for radius in minimum_radii:
        column = _radial_index(radii, radius)
        reference_vector = reference_coefficients[:, column]
        candidate_vector = candidate_coefficients[:, column]
        reference_values = coordinate_grid.design @ reference_vector
        candidate_values = coordinate_grid.design @ candidate_vector
        normalized = np.abs(candidate_values - reference_values) / hybrid_tolerance(
            reference_values, evaluation_spec
        )
        starts = np.unique(
            np.concatenate(
                [np.argsort(candidate_values)[:8], np.argsort(normalized)[-8:]]
            )
        )
        seen: set[tuple[float, ...]] = set()
        for start_index in starts:
            start = np.array(
                [
                    coordinate_grid.theta1[start_index],
                    coordinate_grid.theta2[start_index],
                    coordinate_grid.phi[start_index],
                ]
            )
            result = minimize(
                lambda angles: _single_value(candidate_vector, angles),
                x0=start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1.0e-10},
            )
            key = tuple(np.round(result.x, 4))
            if key in seen:
                continue
            seen.add(key)
            candidate_value = _single_value(candidate_vector, result.x)
            reference_value = _single_value(reference_vector, result.x)
            minimum_rows.append(
                {
                    "R": radius,
                    "Theta_1": np.degrees(result.x[0]),
                    "Theta_2": np.degrees(result.x[1]),
                    "Phi": np.degrees(result.x[2]),
                    "V_candidate": candidate_value,
                    "V_reference": reference_value,
                    "candidate_minus_reference": candidate_value - reference_value,
                    "optimizer_success": bool(result.success),
                }
            )

    return OffGridStressResult(
        summary=pd.DataFrame(summary_rows),
        candidates=pd.DataFrame(candidate_rows),
        local_minima=(
            pd.DataFrame(minimum_rows)
            .sort_values(["R", "V_candidate"])
            .reset_index(drop=True)
        ),
        grids=grids,
    )


def _read_run_coefficients(path: Path) -> tuple[dict[str, object], NDArray[np.float64], NDArray[np.float64]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    radii = np.asarray(manifest["input"]["radii_bohr"], dtype=float)
    table = pd.read_csv(path / "coefficients.csv")
    matrix = coefficient_table_to_matrix(table, radii, basis=CANDIDATE_BASIS_V1)
    return manifest, radii, matrix


def run_offgrid_comparison(
    reference_directory: str | Path,
    candidate_directory: str | Path,
    output_directory: str | Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    margin: float = DEFAULT_MARGIN,
    derivative_step: float = DEFAULT_DERIVATIVE_STEP,
    evaluation_spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> Path:
    """Write an immutable off-grid model--model diagnostic artifact."""

    reference_path = Path(reference_directory).resolve()
    candidate_path = Path(candidate_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Off-grid output already exists: {output}")
    reference_manifest, reference_radii, reference = _read_run_coefficients(
        reference_path
    )
    candidate_manifest, candidate_radii, candidate = _read_run_coefficients(
        candidate_path
    )
    if reference_manifest["basis"]["id"] != candidate_manifest["basis"]["id"]:
        raise ValueError("Off-grid comparison requires the same basis contract.")
    np.testing.assert_allclose(reference_radii, candidate_radii, rtol=0.0, atol=0.0)
    result = compare_coefficient_surfaces_offgrid(
        reference_radii,
        reference,
        candidate,
        sample_size=sample_size,
        seed=seed,
        margin=margin,
        derivative_step=derivative_step,
        evaluation_spec=evaluation_spec,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        result.summary.to_csv(
            temporary / "offgrid_summary.csv", index=False, float_format="%.17g"
        )
        result.candidates.to_csv(
            temporary / "offgrid_candidates.csv", index=False, float_format="%.17g"
        )
        result.local_minima.to_csv(
            temporary / "local_minima.csv", index=False, float_format="%.17g"
        )
        files = {
            "summary": "offgrid_summary.csv",
            "candidates": "offgrid_candidates.csv",
            "local_minima": "local_minima.csv",
        }
        write_json(
            temporary / "manifest.json",
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "artifact_id": OFFGRID_DIAGNOSTIC_ID,
                "created_utc": utc_timestamp(),
                "evidence_type": "off-grid model-model diagnostic; not independent validation",
                "reference_method": reference_manifest["method"]["id"],
                "candidate_method": candidate_manifest["method"]["id"],
                "reference_manifest_sha256": sha256_file(reference_path / "manifest.json"),
                "candidate_manifest_sha256": sha256_file(candidate_path / "manifest.json"),
                "basis_id": reference_manifest["basis"]["id"],
                "parameters": {
                    "sampling_modes": ["coordinate-uniform", "solid-angle"],
                    "sample_size_per_mode": sample_size,
                    "sobol_seed": seed,
                    "angular_margin": margin,
                    "derivative_step_rad": derivative_step,
                    "anchor_radii_bohr": list(DEFAULT_ANCHOR_RADII),
                    "local_minimum_radii_bohr": list(DEFAULT_MINIMUM_RADII),
                },
                "evaluation_spec": evaluation_spec.to_manifest(),
                "code": {
                    "git_revision_at_invocation": current_git_revision(
                        Path(__file__).resolve().parents[2]
                    ),
                    "software": software_versions(),
                },
                "files": files,
                "file_sha256": {
                    key: sha256_file(temporary / filename)
                    for key, filename in files.items()
                },
            },
        )
        temporary.rename(output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
