"""Radial coefficient-curve and leave-one-radius-out diagnostics."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from .artifacts import (
    RUN_SCHEMA_VERSION,
    current_git_revision,
    software_versions,
    utc_timestamp,
    write_json,
)
from .basis import CANDIDATE_BASIS_V1, build_design_matrix
from .coefficients import coefficient_table_to_matrix
from .data import (
    angles_in_radians,
    load_ab_initio_data,
    radial_potential_matrix,
    reference_orientations,
    sha256_file,
)
from .evaluation import (
    DEFAULT_EVALUATION_SPEC,
    EvaluationSpec,
    energy_band_masks,
    hybrid_tolerance,
)

RADIAL_DIAGNOSTIC_ID = "coefficient-radial-holdout-v1"


def coefficient_curve_diagnostics(
    radii: NDArray[np.float64],
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    design: NDArray[np.float64],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare coefficient vectors, projections, and numerical derivatives."""

    radii = np.asarray(radii, dtype=float)
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    expected = (design.shape[1], len(radii))
    if reference.shape != expected or candidate.shape != expected:
        raise ValueError("Coefficient matrices are incompatible with radii/design.")

    difference = candidate - reference
    projection_difference = design @ difference
    reference_first = np.gradient(reference, radii, axis=1)
    candidate_first = np.gradient(candidate, radii, axis=1)
    reference_second = np.gradient(reference_first, radii, axis=1)
    candidate_second = np.gradient(candidate_first, radii, axis=1)
    rows: list[dict[str, object]] = []
    for column, radius in enumerate(radii):
        reference_vector = reference[:, column]
        candidate_vector = candidate[:, column]
        delta = difference[:, column]
        reference_norm = np.linalg.norm(reference_vector)
        candidate_norm = np.linalg.norm(candidate_vector)
        projected = projection_difference[:, column]
        rows.append(
            {
                "R": float(radius),
                "relative_coefficient_error": (
                    float(np.linalg.norm(delta) / reference_norm)
                    if reference_norm > 0.0
                    else float("nan")
                ),
                "coefficient_cosine_similarity": (
                    float(
                        np.dot(candidate_vector, reference_vector)
                        / (candidate_norm * reference_norm)
                    )
                    if candidate_norm > 0.0 and reference_norm > 0.0
                    else float("nan")
                ),
                "projection_deviation_rmse": float(
                    np.sqrt(np.mean(projected**2))
                ),
                "maximum_projection_deviation": float(np.max(np.abs(projected))),
                "relative_first_derivative_error": (
                    float(
                        np.linalg.norm(
                            candidate_first[:, column] - reference_first[:, column]
                        )
                        / np.linalg.norm(reference_first[:, column])
                    )
                    if np.linalg.norm(reference_first[:, column]) > 0.0
                    else float("nan")
                ),
                "relative_second_derivative_error": (
                    float(
                        np.linalg.norm(
                            candidate_second[:, column] - reference_second[:, column]
                        )
                        / np.linalg.norm(reference_second[:, column])
                    )
                    if np.linalg.norm(reference_second[:, column]) > 0.0
                    else float("nan")
                ),
            }
        )
    per_radius = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "global_relative_coefficient_error": float(
                    np.linalg.norm(difference) / np.linalg.norm(reference)
                ),
                "median_per_R_coefficient_error": float(
                    per_radius["relative_coefficient_error"].median()
                ),
                "maximum_per_R_coefficient_error": float(
                    per_radius["relative_coefficient_error"].max()
                ),
                "minimum_cosine_similarity": float(
                    per_radius["coefficient_cosine_similarity"].min()
                ),
                "maximum_projection_deviation_rmse": float(
                    per_radius["projection_deviation_rmse"].max()
                ),
                "projection_limiting_R": float(
                    per_radius.loc[
                        per_radius["projection_deviation_rmse"].idxmax(), "R"
                    ]
                ),
                "maximum_pointwise_projection_deviation": float(
                    per_radius["maximum_projection_deviation"].max()
                ),
                "global_relative_first_derivative_error": float(
                    np.linalg.norm(candidate_first - reference_first)
                    / np.linalg.norm(reference_first)
                ),
                "global_relative_second_derivative_error": float(
                    np.linalg.norm(candidate_second - reference_second)
                    / np.linalg.norm(reference_second)
                ),
            }
        ]
    )
    return per_radius, summary


def leave_one_radius_out_diagnostic(
    radii: NDArray[np.float64],
    truth: NDArray[np.float64],
    design: NDArray[np.float64],
    coefficient_sources: dict[str, NDArray[np.float64]],
    *,
    minimum_R: float = 4.4,
    maximum_R: float = 6.6,
    evaluation_spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate diagnostic cubic-spline interpolation at withheld radial nodes."""

    radii = np.asarray(radii, dtype=float)
    truth = np.asarray(truth, dtype=float)
    holdouts = radii[(radii > minimum_R) & (radii < maximum_R)]
    detail_rows: list[dict[str, object]] = []
    for method, coefficients in coefficient_sources.items():
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (design.shape[1], len(radii)):
            raise ValueError(f"{method} coefficient matrix has an invalid shape.")
        for radius in holdouts:
            column = int(np.flatnonzero(np.isclose(radii, radius))[0])
            keep = np.ones(len(radii), dtype=bool)
            keep[column] = False
            spline = CubicSpline(radii[keep], coefficients[:, keep], axis=1)
            prediction = design @ spline(radius)
            target = truth[:, column]
            for band, mask in energy_band_masks(target, evaluation_spec):
                if band == "all" or band.startswith("guardrail_") or not mask.any():
                    continue
                absolute = np.abs(prediction[mask] - target[mask])
                normalized = absolute / hybrid_tolerance(target[mask], evaluation_spec)
                detail_rows.append(
                    {
                        "method": method,
                        "heldout_R": float(radius),
                        "energy_band": band,
                        "n_points": int(mask.sum()),
                        "mae": float(absolute.mean()),
                        "rmse": float(np.sqrt(np.mean(absolute**2))),
                        "p95_hybrid_normalized_error": float(
                            np.percentile(normalized, 95)
                        ),
                        "maximum_hybrid_normalized_error": float(normalized.max()),
                        "within_tolerance_percent": float(
                            100.0 * np.mean(normalized <= 1.0)
                        ),
                    }
                )
    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["method", "heldout_R"], as_index=False)
        .agg(
            worst_band_mae=("mae", "max"),
            worst_band_p95_normalized_error=(
                "p95_hybrid_normalized_error",
                "max",
            ),
            worst_individual_normalized_error=(
                "maximum_hybrid_normalized_error",
                "max",
            ),
            minimum_band_within_tolerance_percent=(
                "within_tolerance_percent",
                "min",
            ),
        )
    )
    return detail, summary


def _read_run(path: Path) -> tuple[dict[str, object], NDArray[np.float64], NDArray[np.float64]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    radii = np.asarray(manifest["input"]["radii_bohr"], dtype=float)
    coefficients = coefficient_table_to_matrix(
        pd.read_csv(path / "coefficients.csv"),
        radii,
        basis=CANDIDATE_BASIS_V1,
    )
    return manifest, radii, coefficients


def run_radial_diagnostics(
    data_path: str | Path,
    reference_directory: str | Path,
    candidate_directory: str | Path,
    output_directory: str | Path,
    *,
    minimum_R: float = 4.4,
    maximum_R: float = 6.6,
    evaluation_spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> Path:
    """Write coefficient smoothness and leave-one-R-out diagnostic artifacts."""

    source = Path(data_path).resolve()
    reference_path = Path(reference_directory).resolve()
    candidate_path = Path(candidate_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Radial diagnostic directory exists: {output}")
    frame = load_ab_initio_data(source)
    orientations = reference_orientations(frame)
    data_radii, truth = radial_potential_matrix(frame)
    design = build_design_matrix(*angles_in_radians(orientations))
    reference_manifest, reference_radii, reference = _read_run(reference_path)
    candidate_manifest, candidate_radii, candidate = _read_run(candidate_path)
    if reference_manifest["basis"]["id"] != candidate_manifest["basis"]["id"]:
        raise ValueError("Radial diagnostics require a shared basis contract.")
    np.testing.assert_allclose(data_radii, reference_radii, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(data_radii, candidate_radii, rtol=0.0, atol=0.0)
    if reference_manifest["input"]["sha256"] != sha256_file(source):
        raise ValueError("Reference run does not identify the supplied dataset.")
    if candidate_manifest["input"]["sha256"] != sha256_file(source):
        raise ValueError("Candidate run does not identify the supplied dataset.")

    coefficient_by_R, coefficient_summary = coefficient_curve_diagnostics(
        data_radii, reference, candidate, design
    )
    holdout_detail, holdout_summary = leave_one_radius_out_diagnostic(
        data_radii,
        truth,
        design,
        {
            str(reference_manifest["method"]["id"]): reference,
            str(candidate_manifest["method"]["id"]): candidate,
        },
        minimum_R=minimum_R,
        maximum_R=maximum_R,
        evaluation_spec=evaluation_spec,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        coefficient_by_R.to_csv(
            temporary / "coefficient_curve_by_R.csv",
            index=False,
            float_format="%.17g",
        )
        coefficient_summary.to_csv(
            temporary / "coefficient_curve_summary.csv",
            index=False,
            float_format="%.17g",
        )
        holdout_detail.to_csv(
            temporary / "radial_holdout_detail.csv",
            index=False,
            float_format="%.17g",
        )
        holdout_summary.to_csv(
            temporary / "radial_holdout_summary.csv",
            index=False,
            float_format="%.17g",
        )
        files = {
            "coefficient_by_R": "coefficient_curve_by_R.csv",
            "coefficient_summary": "coefficient_curve_summary.csv",
            "holdout_detail": "radial_holdout_detail.csv",
            "holdout_summary": "radial_holdout_summary.csv",
        }
        write_json(
            temporary / "manifest.json",
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "artifact_id": RADIAL_DIAGNOSTIC_ID,
                "created_utc": utc_timestamp(),
                "evidence_type": (
                    "diagnostic numerical spline; not production radial interpolation"
                ),
                "input_dataset_sha256": sha256_file(source),
                "reference_method": reference_manifest["method"]["id"],
                "candidate_method": candidate_manifest["method"]["id"],
                "reference_manifest_sha256": sha256_file(
                    reference_path / "manifest.json"
                ),
                "candidate_manifest_sha256": sha256_file(
                    candidate_path / "manifest.json"
                ),
                "basis_id": reference_manifest["basis"]["id"],
                "parameters": {
                    "holdout_minimum_R_bohr": minimum_R,
                    "holdout_maximum_R_bohr": maximum_R,
                    "interpolator": "scipy.interpolate.CubicSpline default not-a-knot",
                    "coefficient_derivatives": "numpy.gradient on supplied radial grid",
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
