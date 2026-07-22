"""Prospective independent-ab-initio request construction and scoring."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import qmc

from .artifacts import (
    RUN_SCHEMA_VERSION,
    current_git_revision,
    software_versions,
    utc_timestamp,
    write_json,
)
from .basis import CANDIDATE_BASIS_V1, build_design_matrix
from .data import sha256_file
from .evaluation import (
    DEFAULT_EVALUATION_SPEC,
    EvaluationSpec,
    energy_band_masks,
    hybrid_tolerance,
)
from .offgrid import (
    DEFAULT_MARGIN,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    OffGridStressResult,
    _read_run_coefficients,
    compare_coefficient_surfaces_offgrid,
)

VALIDATION_REQUEST_ID = "co2-h2-independent-request-v3"
VALIDATION_GEOMETRY_SHA256 = (
    "e36ac1070256ff72458abda2ee49eadb287cefaa07b87bb2a48c2fffdd870ec2"
)
REQUEST_SOBOL_SEED = 20260716
PRIMARY_VALIDATION_RADII = (4.4, 4.8, 5.0)
OPTIONAL_RADIAL_RADII = (4.7, 4.9)
EXPECTED_CATEGORIES = {
    "Distributional: coordinate-uniform": 12,
    "Distributional: solid-angle": 12,
    "Stress: energy disagreement": 8,
    "Stress: angular derivative": 8,
    "Stress: wall boundary": 4,
    "Stress: local extremum": 4,
}

# The L-BFGS-B search can return either of two basis-equivalent representatives
# for the symmetric minimum at theta1=theta2=90 degrees. SciPy/environment
# differences reproduced 47 v3 rows exactly and returned phi=0 for this one,
# whereas the executed v3 notebook froze the phi~=180 representative. Preserve
# the preregistered coordinate rather than letting an optimizer tie alter it.
V3_SYMMETRIC_MINIMUM = np.array([90.0, 89.999999, 179.999998])


@dataclass(frozen=True)
class ValidationRequest:
    """Frozen coordinate and calculation tables for independent validation."""

    orientations: pd.DataFrame
    primary_energies: pd.DataFrame
    optional_radial_energies: pd.DataFrame
    complete_energies: pd.DataFrame


def make_distributional_request(
    *,
    request_seed: int = REQUEST_SOBOL_SEED,
    margin: float = DEFAULT_MARGIN,
) -> pd.DataFrame:
    """Construct the 24 ordinary, label-free Sobol orientations."""

    rows: list[dict[str, object]] = []
    for mode in ("coordinate-uniform", "solid-angle"):
        seed_offset = 0 if mode == "coordinate-uniform" else 1
        unit = qmc.Sobol(
            d=3, scramble=True, seed=request_seed + seed_offset
        ).random_base2(4)[:12]
        unit = margin + (1.0 - 2.0 * margin) * unit
        if mode == "coordinate-uniform":
            theta1 = np.pi * unit[:, 0]
            theta2 = np.pi * unit[:, 1]
        else:
            theta1 = np.arccos(1.0 - 2.0 * unit[:, 0])
            theta2 = np.arccos(1.0 - 2.0 * unit[:, 1])
        angles = np.column_stack(
            [np.degrees(theta1), np.degrees(theta2), 180.0 * unit[:, 2]]
        )
        for theta_1, theta_2, phi in angles:
            rows.append(
                {
                    "Category": f"Distributional: {mode}",
                    "Selection detail": (
                        f"Fresh Sobol seed {request_seed}; label-free"
                    ),
                    "Theta_1": theta_1,
                    "Theta_2": theta_2,
                    "Phi": phi,
                }
            )
    return pd.DataFrame(rows)


def _signature(row: pd.Series, decimals: int = 10) -> tuple[float, ...]:
    radians = np.radians([row["Theta_1"], row["Theta_2"], row["Phi"]])
    design = build_design_matrix(
        np.array([radians[0]]),
        np.array([radians[1]]),
        np.array([radians[2]]),
    )[0]
    return tuple(np.round(design, decimals))


def _radial_index(radii: NDArray[np.float64], radius: float) -> int:
    matches = np.flatnonzero(np.isclose(radii, radius))
    if len(matches) != 1:
        raise ValueError(f"Expected one coefficient column at R={radius}.")
    return int(matches[0])


def build_validation_request(
    stress: OffGridStressResult,
    radii: NDArray[np.float64],
    reference_coefficients: NDArray[np.float64],
    candidate_coefficients: NDArray[np.float64],
    *,
    request_seed: int = REQUEST_SOBOL_SEED,
    primary_radii: tuple[float, ...] = PRIMARY_VALIDATION_RADII,
    optional_radii: tuple[float, ...] = OPTIONAL_RADIAL_RADII,
    margin: float = DEFAULT_MARGIN,
    evaluation_spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> ValidationRequest:
    """Build the notebook-v3 request without inspecting any requested label."""

    radii = np.asarray(radii, dtype=float)
    distributional = make_distributional_request(
        request_seed=request_seed, margin=margin
    )
    used_signatures = {_signature(row) for _, row in distributional.iterrows()}
    if len(used_signatures) != len(distributional):
        raise RuntimeError("Distributional request has symmetry-equivalent rows.")

    candidates = stress.candidates.rename(
        columns={
            "sampling_measure": "Sampling measure",
            "absolute_disagreement": "Absolute disagreement",
            "normalized_disagreement": "Normalized disagreement",
            "gradient_disagreement_cm1_per_rad": "Gradient disagreement",
            "reference_high_candidate_below_warning": "Severe opening",
            "reference_above_1000_candidate_below_1000": "1000 crossing",
        }
    )
    stress_pool = (
        candidates.groupby(
            ["Sampling measure", "Theta_1", "Theta_2", "Phi"], as_index=False
        )
        .agg(
            Source_R=("R", lambda values: ",".join(
                f"{value:.1f}" for value in sorted(set(values))
            )),
            Maximum_absolute_disagreement=("Absolute disagreement", "max"),
            Maximum_normalized_disagreement=("Normalized disagreement", "max"),
            Maximum_gradient_disagreement=("Gradient disagreement", "max"),
            Any_severe_opening=("Severe opening", "max"),
            Any_1000_crossing=("1000 crossing", "max"),
        )
    )
    selected: list[dict[str, object]] = []

    def add_unique(
        pool: pd.DataFrame,
        count: int,
        category: str,
        detail,
    ) -> None:
        added = 0
        for _, row in pool.iterrows():
            signature = _signature(row)
            if signature in used_signatures:
                continue
            used_signatures.add(signature)
            selected.append(
                {
                    "Category": category,
                    "Selection detail": detail(row),
                    "Theta_1": row["Theta_1"],
                    "Theta_2": row["Theta_2"],
                    "Phi": row["Phi"],
                }
            )
            added += 1
            if added == count:
                break
        if added != count:
            raise RuntimeError(f"Selected only {added} of {count} rows for {category}.")

    add_unique(
        stress_pool.sort_values("Maximum_normalized_disagreement", ascending=False),
        4,
        "Stress: energy disagreement",
        lambda row: (
            "largest normalized 180-vs-500 disagreement; "
            f"source R={row['Source_R']}"
        ),
    )
    add_unique(
        stress_pool.sort_values("Maximum_absolute_disagreement", ascending=False),
        4,
        "Stress: energy disagreement",
        lambda row: (
            "largest absolute 180-vs-500 disagreement; "
            f"source R={row['Source_R']}"
        ),
    )
    add_unique(
        stress_pool.sort_values("Maximum_gradient_disagreement", ascending=False),
        8,
        "Stress: angular derivative",
        lambda row: (
            "largest angular-gradient disagreement; "
            f"source R={row['Source_R']}"
        ),
    )

    threshold_values = np.array(
        [
            evaluation_spec.low_repulsive_upper_cm1,
            evaluation_spec.lower_wall_upper_cm1,
            evaluation_spec.physical_ceiling_cm1,
        ]
    )
    boundary_rows: list[dict[str, object]] = []
    extrema_rows: list[dict[str, object]] = []
    for mode, grid in stress.grids.items():
        degrees = np.column_stack(
            [
                np.degrees(grid.theta1),
                np.degrees(grid.theta2),
                np.degrees(grid.phi),
            ]
        )
        for radius in primary_radii:
            column = _radial_index(radii, radius)
            reference_values = grid.design @ reference_coefficients[:, column]
            candidate_values = grid.design @ candidate_coefficients[:, column]
            crossing = (
                (reference_values[:, None] - threshold_values[None, :])
                * (candidate_values[:, None] - threshold_values[None, :])
                < 0.0
            ).any(axis=1)
            distance = np.minimum(
                np.min(
                    np.abs(reference_values[:, None] - threshold_values[None, :]),
                    axis=1,
                ),
                np.min(
                    np.abs(candidate_values[:, None] - threshold_values[None, :]),
                    axis=1,
                ),
            )
            boundary_indices = np.unique(
                np.concatenate([np.flatnonzero(crossing), np.argsort(distance)[:20]])
            )
            for index in boundary_indices:
                boundary_rows.append(
                    {
                        "Sampling measure": mode,
                        "Source_R": radius,
                        "Theta_1": degrees[index, 0],
                        "Theta_2": degrees[index, 1],
                        "Phi": degrees[index, 2],
                        "Any threshold crossing": bool(crossing[index]),
                        "Boundary distance": distance[index],
                    }
                )
            for index in np.argsort(np.minimum(reference_values, candidate_values))[:20]:
                extrema_rows.append(
                    {
                        "Sampling measure": mode,
                        "Source_R": radius,
                        "Theta_1": degrees[index, 0],
                        "Theta_2": degrees[index, 1],
                        "Phi": degrees[index, 2],
                        "Minimum fitted energy": min(
                            reference_values[index], candidate_values[index]
                        ),
                    }
                )

    boundary_pool = (
        pd.DataFrame(boundary_rows)
        .groupby(
            ["Sampling measure", "Theta_1", "Theta_2", "Phi"], as_index=False
        )
        .agg(
            Source_R=("Source_R", lambda values: ",".join(
                f"{value:.1f}" for value in sorted(set(values))
            )),
            Any_threshold_crossing=("Any threshold crossing", "max"),
            Boundary_distance=("Boundary distance", "min"),
        )
        .sort_values(
            ["Any_threshold_crossing", "Boundary_distance"],
            ascending=[False, True],
        )
    )
    add_unique(
        boundary_pool,
        4,
        "Stress: wall boundary",
        lambda row: (
            f"threshold crossing={bool(row['Any_threshold_crossing'])}; "
            f"nearest fitted-boundary distance={row['Boundary_distance']:.4f}; "
            f"source R={row['Source_R']}"
        ),
    )

    optimized = stress.local_minima[
        ["R", "Theta_1", "Theta_2", "Phi", "V_candidate"]
    ].rename(columns={"R": "Source_R", "V_candidate": "Minimum fitted energy"})
    optimized["Sampling measure"] = "optimized"
    extrema_pool = pd.concat(
        [optimized, pd.DataFrame(extrema_rows)], ignore_index=True
    ).sort_values("Minimum fitted energy")
    add_unique(
        extrema_pool,
        4,
        "Stress: local extremum",
        lambda row: (
            f"low fitted-energy/local-minimum candidate; source R={row['Source_R']}"
        ),
    )

    stress_request = pd.DataFrame(selected)
    orientations = pd.concat([distributional, stress_request], ignore_index=True)
    orientations.insert(
        0, "Validation ID", [f"V{index:03d}" for index in range(1, 49)]
    )
    frozen_minimum_design = build_design_matrix(
        np.radians(np.array([V3_SYMMETRIC_MINIMUM[0]])),
        np.radians(np.array([V3_SYMMETRIC_MINIMUM[1]])),
        np.radians(np.array([V3_SYMMETRIC_MINIMUM[2]])),
    )[0]
    for row_index in orientations.index[
        orientations["Category"] == "Stress: local extremum"
    ]:
        row = orientations.loc[row_index]
        row_design = build_design_matrix(
            np.radians(np.array([row["Theta_1"]], dtype=float)),
            np.radians(np.array([row["Theta_2"]], dtype=float)),
            np.radians(np.array([row["Phi"]], dtype=float)),
        )[0]
        if np.max(np.abs(row_design - frozen_minimum_design)) < 1.0e-10:
            orientations.loc[
                row_index, ["Theta_1", "Theta_2", "Phi"]
            ] = V3_SYMMETRIC_MINIMUM
            break
    orientations[["Theta_1", "Theta_2", "Phi"]] = orientations[
        ["Theta_1", "Theta_2", "Phi"]
    ].round(6)

    primary = orientations.merge(pd.DataFrame({"R": primary_radii}), how="cross")
    primary.insert(0, "Request phase", "Primary off-grid")
    radial_counts = {
        "Distributional: coordinate-uniform": 2,
        "Distributional: solid-angle": 2,
        "Stress: energy disagreement": 3,
        "Stress: angular derivative": 3,
        "Stress: wall boundary": 1,
        "Stress: local extremum": 1,
    }
    radial_orientations = pd.concat(
        [
            orientations.loc[orientations["Category"] == category].head(count)
            for category, count in radial_counts.items()
        ],
        ignore_index=True,
    )
    optional = radial_orientations.merge(
        pd.DataFrame({"R": optional_radii}), how="cross"
    )
    optional.insert(0, "Request phase", "Optional radial midpoint")
    request = ValidationRequest(
        orientations=orientations,
        primary_energies=primary,
        optional_radial_energies=optional,
        complete_energies=pd.concat([primary, optional], ignore_index=True),
    )
    validate_request(request)
    return request


def validate_request(request: ValidationRequest) -> None:
    """Enforce the frozen counts, uniqueness, domains, and basis-row identity."""

    orientations = request.orientations
    if len(orientations) != 48 or orientations["Validation ID"].nunique() != 48:
        raise ValueError("The validation request must contain 48 unique IDs.")
    if orientations["Category"].value_counts().to_dict() != EXPECTED_CATEGORIES:
        raise ValueError("Validation category counts do not match the v3 contract.")
    angles = orientations[["Theta_1", "Theta_2", "Phi"]]
    if angles.duplicated().any():
        raise ValueError("Validation orientations must be coordinate-unique.")
    if not (
        angles["Theta_1"].between(0.0, 180.0).all()
        and angles["Theta_2"].between(0.0, 180.0).all()
        and angles["Phi"].between(0.0, 180.0).all()
    ):
        raise ValueError("Validation angles are outside the contracted domain.")
    design = build_design_matrix(
        np.radians(angles["Theta_1"].to_numpy()),
        np.radians(angles["Theta_2"].to_numpy()),
        np.radians(angles["Phi"].to_numpy()),
    )
    if len({tuple(np.round(row, 9)) for row in design}) != 48:
        raise ValueError("Validation orientations contain equivalent basis rows.")
    if len(request.primary_energies) != 144:
        raise ValueError("Primary request must contain 144 energies.")
    if len(request.optional_radial_energies) != 24:
        raise ValueError("Optional radial request must contain 24 energies.")
    if len(request.complete_energies) != 168:
        raise ValueError("Complete request must contain 168 energies.")
    if request.complete_energies[["R", "Theta_1", "Theta_2", "Phi"]].duplicated().any():
        raise ValueError("Complete request contains duplicate geometries.")


def run_validation_request(
    reference_directory: str | Path,
    candidate_directory: str | Path,
    output_directory: str | Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    offgrid_seed: int = DEFAULT_SEED,
    request_seed: int = REQUEST_SOBOL_SEED,
    evaluation_spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> Path:
    """Generate and write the immutable prospective request artifact."""

    reference_path = Path(reference_directory).resolve()
    candidate_path = Path(candidate_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Validation-request directory exists: {output}")
    reference_manifest, radii, reference = _read_run_coefficients(reference_path)
    candidate_manifest, candidate_radii, candidate = _read_run_coefficients(
        candidate_path
    )
    if reference_manifest["basis"]["id"] != candidate_manifest["basis"]["id"]:
        raise ValueError("Validation request requires a shared basis contract.")
    np.testing.assert_allclose(radii, candidate_radii, rtol=0.0, atol=0.0)
    stress = compare_coefficient_surfaces_offgrid(
        radii,
        reference,
        candidate,
        sample_size=sample_size,
        seed=offgrid_seed,
        evaluation_spec=evaluation_spec,
    )
    request = build_validation_request(
        stress,
        radii,
        reference,
        candidate,
        request_seed=request_seed,
        evaluation_spec=evaluation_spec,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    filenames = {
        "orientations": "validation_orientations_48.csv",
        "primary": "primary_validation_energies_144.csv",
        "optional_radial": "optional_radial_energies_24.csv",
        "complete": "complete_ab_initio_request_168.csv",
    }
    try:
        request.orientations.to_csv(temporary / filenames["orientations"], index=False)
        request.primary_energies.to_csv(temporary / filenames["primary"], index=False)
        request.optional_radial_energies.to_csv(
            temporary / filenames["optional_radial"], index=False
        )
        request.complete_energies.to_csv(temporary / filenames["complete"], index=False)
        file_hashes = {
            key: sha256_file(temporary / filename) for key, filename in filenames.items()
        }
        write_json(
            temporary / "manifest.json",
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "artifact_id": VALIDATION_REQUEST_ID,
                "created_utc": utc_timestamp(),
                "status": "frozen-before-independent-labels",
                "evidence_boundary": (
                    "coordinates use existing model diagnostics; requested energies are unknown"
                ),
                "reference_method": reference_manifest["method"]["id"],
                "candidate_method": candidate_manifest["method"]["id"],
                "reference_manifest_sha256": sha256_file(
                    reference_path / "manifest.json"
                ),
                "candidate_manifest_sha256": sha256_file(
                    candidate_path / "manifest.json"
                ),
                "input_dataset_sha256": reference_manifest["input"]["sha256"],
                "basis_id": reference_manifest["basis"]["id"],
                "parameters": {
                    "offgrid_diagnostic_id": "sobol-model-disagreement-v1",
                    "offgrid_sample_size_per_measure": sample_size,
                    "offgrid_seed": offgrid_seed,
                    "request_seed": request_seed,
                    "primary_radii_bohr": list(PRIMARY_VALIDATION_RADII),
                    "optional_radial_radii_bohr": list(OPTIONAL_RADIAL_RADII),
                    "category_counts": EXPECTED_CATEGORIES,
                },
                "decision_rule": {
                    "evaluation_spec": evaluation_spec.to_manifest(),
                    "strict_failure": (
                        "any V_true < physical_ceiling with normalized error > 1"
                    ),
                    "wall_failure": (
                        "any V_true >= physical_ceiling reconstructed below guardrail warning"
                    ),
                    "shared_failure": (
                        "if both fits fail the same geometry, investigate basis or radial representation"
                    ),
                    "after_failure": (
                        "augment or reoptimize, then reserve a fresh secondary holdout"
                    ),
                },
                "counts": {
                    "orientations": 48,
                    "primary_energies": 144,
                    "optional_radial_energies": 24,
                    "complete_energies": 168,
                },
                "code": {
                    "git_revision_at_invocation": current_git_revision(
                        Path(__file__).resolve().parents[2]
                    ),
                    "software": software_versions(),
                },
                "files": filenames,
                "file_sha256": file_hashes,
            },
        )
        temporary.rename(output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def score_returned_validation_energies(
    returned: pd.DataFrame,
    radii: NDArray[np.float64],
    method_coefficients: dict[str, NDArray[np.float64]],
    *,
    evaluation_spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Apply the preregistered rule after independent labels are returned."""

    required = {"Validation ID", "R", "Theta_1", "Theta_2", "Phi", "V"}
    missing = required.difference(returned.columns)
    if missing:
        raise ValueError(f"Returned validation table is missing {sorted(missing)}.")
    if returned[list(required)].isna().any().any():
        raise ValueError("Returned validation table contains missing values.")
    numeric_columns = ["R", "Theta_1", "Theta_2", "Phi", "V"]
    if not np.isfinite(returned[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Returned validation table contains non-finite values.")
    if len(returned) != 144:
        raise ValueError("A final primary-validation score requires all 144 rows.")
    if returned[["Validation ID", "R"]].duplicated().any():
        raise ValueError("Returned validation table contains duplicate ID/radius rows.")
    expected_radii = set(PRIMARY_VALIDATION_RADII)
    for validation_id, rows_for_id in returned.groupby("Validation ID"):
        if len(rows_for_id) != 3 or set(rows_for_id["R"].astype(float)) != expected_radii:
            raise ValueError(
                f"{validation_id} must contain exactly the three primary radii."
            )
        if len(rows_for_id[["Theta_1", "Theta_2", "Phi"]].drop_duplicates()) != 1:
            raise ValueError(f"{validation_id} changes angles between radii.")
    orientations = (
        returned[["Validation ID", "Theta_1", "Theta_2", "Phi"]]
        .drop_duplicates()
        .sort_values("Validation ID")
    )
    geometry_text = "".join(
        f"{row['Validation ID']},{row['Theta_1']:.6f},"
        f"{row['Theta_2']:.6f},{row['Phi']:.6f}\n"
        for _, row in orientations.iterrows()
    )
    geometry_sha256 = hashlib.sha256(geometry_text.encode("utf-8")).hexdigest()
    if geometry_sha256 != VALIDATION_GEOMETRY_SHA256:
        raise ValueError(
            "Returned geometries do not match the frozen v3 validation request."
        )
    rows: list[dict[str, object]] = []
    radii = np.asarray(radii, dtype=float)
    for _, geometry in returned.iterrows():
        column = _radial_index(radii, float(geometry["R"]))
        design = build_design_matrix(
            np.radians(np.array([geometry["Theta_1"]], dtype=float)),
            np.radians(np.array([geometry["Theta_2"]], dtype=float)),
            np.radians(np.array([geometry["Phi"]], dtype=float)),
        )[0]
        truth = float(geometry["V"])
        band = next(
            name
            for name, mask in energy_band_masks(np.array([truth]), evaluation_spec)
            if name != "all" and bool(mask[0])
        )
        for method, coefficients in method_coefficients.items():
            prediction = float(design @ coefficients[:, column])
            normalized = abs(prediction - truth) / float(
                hybrid_tolerance(np.array([truth]), evaluation_spec)[0]
            )
            rows.append(
                {
                    "method": method,
                    "Validation ID": geometry.get("Validation ID", pd.NA),
                    "R": float(geometry["R"]),
                    "Theta_1": float(geometry["Theta_1"]),
                    "Theta_2": float(geometry["Theta_2"]),
                    "Phi": float(geometry["Phi"]),
                    "V_true": truth,
                    "V_predicted": prediction,
                    "energy_band": band,
                    "absolute_error": abs(prediction - truth),
                    "hybrid_normalized_error": normalized,
                    "strict_failure": (
                        truth < evaluation_spec.physical_ceiling_cm1
                        and normalized > 1.0
                    ),
                    "wall_failure": (
                        truth >= evaluation_spec.physical_ceiling_cm1
                        and prediction < evaluation_spec.guardrail_warning_cm1
                    ),
                }
            )
    pointwise = pd.DataFrame(rows)
    summary = (
        pointwise.groupby(["method", "R", "energy_band"], as_index=False)
        .agg(
            n_points=("absolute_error", "size"),
            rmse=("absolute_error", lambda values: float(
                np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2))
            )),
            maximum_absolute_error=("absolute_error", "max"),
            maximum_hybrid_normalized_error=("hybrid_normalized_error", "max"),
            strict_failures=("strict_failure", "sum"),
            wall_failures=("wall_failure", "sum"),
        )
    )
    keys = ["R", "Theta_1", "Theta_2", "Phi"]
    failures = pointwise.assign(any_failure=lambda table: table["strict_failure"] | table["wall_failure"])
    shared = failures.groupby(keys)["any_failure"].all()
    method_outcomes = {
        str(method): {
            "pass": bool(
                not method_rows["strict_failure"].any()
                and not method_rows["wall_failure"].any()
            ),
            "strict_failure_count": int(method_rows["strict_failure"].sum()),
            "wall_failure_count": int(method_rows["wall_failure"].sum()),
        }
        for method, method_rows in failures.groupby("method")
    }
    decision = {
        "overall_pass": bool(all(row["pass"] for row in method_outcomes.values())),
        "method_outcomes": method_outcomes,
        "strict_failure_count": int(failures["strict_failure"].sum()),
        "wall_failure_count": int(failures["wall_failure"].sum()),
        "shared_failure_geometry_count": int(shared.sum()),
        "interpretation": (
            "Failures require investigation and a fresh secondary holdout after tuning; "
            "shared failures prioritize basis or radial representation diagnosis."
        ),
    }
    return pointwise, summary, decision


def run_validation_scoring(
    returned_energy_path: str | Path,
    reference_directory: str | Path,
    candidate_directory: str | Path,
    output_directory: str | Path,
    *,
    evaluation_spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> Path:
    """Write an immutable score artifact after primary labels are returned."""

    returned_path = Path(returned_energy_path).resolve()
    reference_path = Path(reference_directory).resolve()
    candidate_path = Path(candidate_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Validation score directory exists: {output}")
    reference_manifest, radii, reference = _read_run_coefficients(reference_path)
    candidate_manifest, candidate_radii, candidate = _read_run_coefficients(
        candidate_path
    )
    if reference_manifest["basis"]["id"] != candidate_manifest["basis"]["id"]:
        raise ValueError("Validation scoring requires a shared basis contract.")
    np.testing.assert_allclose(radii, candidate_radii, rtol=0.0, atol=0.0)
    returned = pd.read_csv(returned_path)
    pointwise, summary, decision = score_returned_validation_energies(
        returned,
        radii,
        {
            str(reference_manifest["method"]["id"]): reference,
            str(candidate_manifest["method"]["id"]): candidate,
        },
        evaluation_spec=evaluation_spec,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        pointwise.to_csv(
            temporary / "pointwise_scores.csv", index=False, float_format="%.17g"
        )
        summary.to_csv(
            temporary / "summary.csv", index=False, float_format="%.17g"
        )
        write_json(temporary / "decision.json", decision)
        files = {
            "pointwise": "pointwise_scores.csv",
            "summary": "summary.csv",
            "decision": "decision.json",
        }
        write_json(
            temporary / "manifest.json",
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "artifact_id": "co2-h2-independent-validation-score-v1",
                "created_utc": utc_timestamp(),
                "returned_energy_sha256": sha256_file(returned_path),
                "reference_method": reference_manifest["method"]["id"],
                "candidate_method": candidate_manifest["method"]["id"],
                "reference_manifest_sha256": sha256_file(
                    reference_path / "manifest.json"
                ),
                "candidate_manifest_sha256": sha256_file(
                    candidate_path / "manifest.json"
                ),
                "evaluation_spec": evaluation_spec.to_manifest(),
                "decision": decision,
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
