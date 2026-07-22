"""End-to-end conversion of an energy grid into a versioned run artifact."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import (
    RUN_SCHEMA_VERSION,
    basis_fingerprint,
    current_git_revision,
    selected_ids_fingerprint,
    software_versions,
    utc_timestamp,
    write_json,
)
from .basis import BASIS_ID, CANDIDATE_BASIS_V1, build_design_matrix
from .coefficients import coefficient_matrix_to_table, coefficient_table_to_matrix
from .data import (
    angles_in_radians,
    load_ab_initio_data,
    radial_potential_matrix,
    reference_orientations,
    sha256_file,
)
from .methods import (
    AngularFitMethod,
    FullGridLeastSquares,
    QRGreedyDOptimalLeastSquares,
)


def _evaluation_partitions(
    orientation_count: int,
    selected_orientation_ids: np.ndarray,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Build full, selected, and unselected masks for common-grid scoring."""

    selected = np.asarray(selected_orientation_ids, dtype=int)
    if selected.ndim != 1:
        raise ValueError("Selected orientation IDs must be one-dimensional.")
    if np.unique(selected).size != selected.size:
        raise ValueError("Selected orientation IDs must be unique.")
    if np.any(selected < 0) or np.any(selected >= orientation_count):
        raise ValueError("Selected orientation ID is outside the evaluation grid.")

    all_mask = np.ones(orientation_count, dtype=bool)
    if selected.size == orientation_count:
        return (("all", all_mask),)
    selected_mask = np.zeros(orientation_count, dtype=bool)
    selected_mask[selected] = True
    return (
        ("all", all_mask),
        ("selected", selected_mask),
        ("unselected", ~selected_mask),
    )


def _error_metrics(
    radii: np.ndarray,
    reference: np.ndarray,
    reconstructed: np.ndarray,
    selected_orientation_ids: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    bands = (
        ("all", lambda values: np.ones(values.shape, dtype=bool)),
        ("attractive_V<0", lambda values: values < 0.0),
        ("low_repulsive_0<=V<1000", lambda values: (values >= 0.0) & (values < 1000.0)),
        ("lower_wall_1000<=V<3000", lambda values: (values >= 1000.0) & (values < 3000.0)),
        ("upper_wall_3000<=V<5000", lambda values: (values >= 3000.0) & (values < 5000.0)),
        ("guardrail_V>=5000", lambda values: values >= 5000.0),
    )
    partitions = _evaluation_partitions(reference.shape[0], selected_orientation_ids)
    for column, radius in enumerate(radii):
        target = reference[:, column]
        residual = reconstructed[:, column] - target
        for subset_name, subset_mask in partitions:
            for band_name, selector in bands:
                mask = subset_mask & selector(target)
                if not mask.any():
                    continue
                values = residual[mask]
                absolute_error = np.abs(values)
                reference_rms = float(np.sqrt(np.mean(target[mask] ** 2)))
                rmse = float(np.sqrt(np.mean(values**2)))
                tolerance = np.maximum(1.0, 0.01 * np.abs(target[mask]))
                normalized_error = absolute_error / tolerance
                rows.append(
                    {
                        "R": float(radius),
                        "evaluation_subset": subset_name,
                        "energy_band": band_name,
                        "n_points": int(mask.sum()),
                        "rmse": rmse,
                        "mae": float(np.mean(absolute_error)),
                        "median_abs_error": float(np.median(absolute_error)),
                        "max_abs_error": float(np.max(absolute_error)),
                        "reference_rms": reference_rms,
                        "normalized_rmse": (
                            rmse / reference_rms if reference_rms else 0.0
                        ),
                        "p95_hybrid_normalized_error": float(
                            np.percentile(normalized_error, 95)
                        ),
                        "max_hybrid_normalized_error": float(
                            np.max(normalized_error)
                        ),
                        "within_hybrid_tolerance_percent": float(
                            100.0 * np.mean(normalized_error <= 1.0)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _guardrail_metrics(
    radii: np.ndarray,
    reference: np.ndarray,
    reconstructed: np.ndarray,
    selected_orientation_ids: np.ndarray,
) -> pd.DataFrame:
    """Score pathological openings where the true wall exceeds 5000 cm^-1."""

    rows: list[dict[str, float | int | str]] = []
    partitions = _evaluation_partitions(reference.shape[0], selected_orientation_ids)
    for column, radius in enumerate(radii):
        target = reference[:, column]
        prediction = reconstructed[:, column]
        for subset_name, subset_mask in partitions:
            mask = subset_mask & (target >= 5000.0)
            if not mask.any():
                continue
            values = prediction[mask]
            rows.append(
                {
                    "R": float(radius),
                    "evaluation_subset": subset_name,
                    "n_points": int(mask.sum()),
                    "minimum_true_potential": float(np.min(target[mask])),
                    "minimum_predicted_potential": float(np.min(values)),
                    "p05_predicted_potential": float(np.percentile(values, 5)),
                    "predicted_below_3000_percent": float(
                        100.0 * np.mean(values < 3000.0)
                    ),
                    "predicted_below_1000_percent": float(
                        100.0 * np.mean(values < 1000.0)
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "R",
            "evaluation_subset",
            "n_points",
            "minimum_true_potential",
            "minimum_predicted_potential",
            "p05_predicted_potential",
            "predicted_below_3000_percent",
            "predicted_below_1000_percent",
        ],
    )


def _v000_table(radii: np.ndarray, coefficients: np.ndarray) -> pd.DataFrame:
    index = CANDIDATE_BASIS_V1.index((0, 0, 0))
    a000 = 1.0 / (4.0 * np.sqrt(np.pi))
    contribution = coefficients[index] * a000
    if radii.size >= 3:
        derivative = np.gradient(contribution, radii, edge_order=2)
    elif radii.size == 2:
        derivative = np.gradient(contribution, radii, edge_order=1)
    else:
        derivative = np.full(contribution.shape, np.nan)
    return pd.DataFrame(
        {
            "R": radii,
            "coefficient_c000": coefficients[index],
            "A000": np.full(radii.shape, a000),
            "isotropic_contribution": contribution,
            "d_isotropic_dR": derivative,
        }
    )


def _plot_v000(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.axhline(0.0, color="0.45", linewidth=0.8)
    axis.plot(
        table["R"],
        table["isotropic_contribution"],
        marker="o",
        markersize=3.0,
    )
    axis.set_xlabel("R (Bohr)")
    axis.set_ylabel(r"$c_{000}A_{000}$ (cm$^{-1}$)")
    axis.set_title("Isotropic contribution diagnostic")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_fit(
    data_path: str | Path,
    output_directory: str | Path,
    method: AngularFitMethod,
    *,
    expected_points_per_radius: int | None = 500,
) -> Path:
    """Execute a method and atomically write its immutable run directory."""

    data_path = Path(data_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Run directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))

    try:
        frame = load_ab_initio_data(
            data_path,
            expected_points_per_radius=expected_points_per_radius,
        )
        orientations = reference_orientations(frame)
        radii, potentials = radial_potential_matrix(frame)
        design = build_design_matrix(*angles_in_radians(orientations))
        result = method.fit(design, potentials)
        if result.coefficients.shape != (len(CANDIDATE_BASIS_V1), radii.size):
            raise ValueError("Method returned an incompatible coefficient matrix.")
        if not np.isfinite(result.coefficients).all():
            raise ValueError("Method returned NaN or infinite coefficients.")
        _evaluation_partitions(
            design.shape[0],
            result.selected_orientation_ids,
        )

        reconstructed = design @ result.coefficients
        coefficient_table = coefficient_matrix_to_table(
            result.method_id,
            result.coefficients,
            radii,
            basis=CANDIDATE_BASIS_V1,
        )
        coefficient_path = temporary / "coefficients.csv"
        coefficient_table.to_csv(coefficient_path, index=False, float_format="%.17g")

        reloaded_table = pd.read_csv(coefficient_path)
        reloaded_coefficients = coefficient_table_to_matrix(
            reloaded_table,
            radii,
            basis=CANDIDATE_BASIS_V1,
        )
        reload_delta = float(
            np.max(np.abs((design @ reloaded_coefficients) - reconstructed))
        )
        tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(reconstructed))))
        if reload_delta > tolerance:
            raise RuntimeError(
                "Serialized coefficients do not reproduce the fitted potential: "
                f"{reload_delta} > {tolerance}."
            )

        metrics = _error_metrics(
            radii,
            potentials,
            reconstructed,
            result.selected_orientation_ids,
        )
        metrics.to_csv(temporary / "metrics.csv", index=False, float_format="%.17g")
        guardrail = _guardrail_metrics(
            radii,
            potentials,
            reconstructed,
            result.selected_orientation_ids,
        )
        guardrail.to_csv(
            temporary / "guardrail.csv",
            index=False,
            float_format="%.17g",
        )

        v000 = _v000_table(radii, result.coefficients)
        v000.to_csv(temporary / "v000.csv", index=False, float_format="%.17g")
        _plot_v000(v000, temporary / "v000.png")

        selected = orientations.iloc[result.selected_orientation_ids][
            ["orientation_id", "Theta_1", "Theta_2", "Phi"]
        ].copy()
        selected.insert(
            0,
            "design_position",
            np.arange(1, len(selected) + 1, dtype=int),
        )
        selected.to_csv(temporary / "selected_orientations.csv", index=False)
        orientations[["orientation_id", "Theta_1", "Theta_2", "Phi"]].to_csv(
            temporary / "evaluation_orientations.csv",
            index=False,
        )

        dataset_hash = sha256_file(data_path)
        last_v000 = float(v000.iloc[-1]["isotropic_contribution"])
        evaluation_singular_values = np.linalg.svd(design, compute_uv=False)
        evaluation_rank = int(np.linalg.matrix_rank(design))
        evaluation_condition = (
            float(evaluation_singular_values[0] / evaluation_singular_values[-1])
            if evaluation_singular_values.size
            and evaluation_singular_values[-1] > 0.0
            else float("inf")
        )
        warnings: list[str] = []
        if last_v000 >= 0.0:
            warnings.append(
                "The isotropic contribution at the largest supplied R is not "
                "attractive; inspect the ab initio offset and long-range replacement."
            )
        if result.rank < len(CANDIDATE_BASIS_V1):
            warnings.append(
                "The fit design is rank deficient for the candidate basis; "
                "coefficient values are not uniquely identified."
            )
        manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "created_utc": utc_timestamp(),
            "method": {
                "id": result.method_id,
                "parameters": result.parameters,
                "selected_orientation_count": int(result.selected_orientation_ids.size),
                "selected_orientation_sha256": selected_ids_fingerprint(
                    result.selected_orientation_ids
                ),
                "target": "raw_potential",
            },
            "input": {
                "filename": data_path.name,
                "sha256": dataset_hash,
                "bytes": data_path.stat().st_size,
                "rows": int(len(frame)),
                "radial_count": int(radii.size),
                "radii_bohr": radii.tolist(),
                "orientation_count": int(len(orientations)),
                "shared_angular_grid": True,
                "units": {"R": "Bohr", "angles": "degree", "V": "cm^-1"},
            },
            "basis": {
                "id": BASIS_ID,
                "status": "candidate-until-production-term-list-validation",
                "term_count": len(CANDIDATE_BASIS_V1),
                "ordered_tuple_sha256": basis_fingerprint(CANDIDATE_BASIS_V1),
                "coefficient_units": "cm^-1",
            },
            "design_matrix": {
                "evaluation_shape": list(design.shape),
                "evaluation_rank": evaluation_rank,
                "evaluation_condition_number": (
                    evaluation_condition
                    if np.isfinite(evaluation_condition)
                    else None
                ),
                "fit_shape": [
                    int(result.selected_orientation_ids.size),
                    int(design.shape[1]),
                ],
                "fit_rank": result.rank,
                "fit_condition_number": (
                    result.condition_number
                    if np.isfinite(result.condition_number)
                    else None
                ),
                "fit_largest_singular_value": float(result.singular_values[0]),
                "fit_smallest_singular_value": float(result.singular_values[-1]),
            },
            "validation": {
                "coefficient_rows": int(len(coefficient_table)),
                "unique_tuple_radius_rows": int(
                    coefficient_table.drop_duplicates(["R", "l1", "l2", "L"]).shape[0]
                ),
                "reconstruction_reload_max_abs_delta": reload_delta,
                "reconstruction_reload_tolerance": tolerance,
                "largest_R_isotropic_contribution_cm^-1": last_v000,
                "warnings": warnings,
            },
            "code": {
                "git_revision": current_git_revision(Path(__file__).resolve().parents[2]),
                "software": software_versions(),
            },
            "files": {
                "coefficients": "coefficients.csv",
                "metrics": "metrics.csv",
                "guardrail": "guardrail.csv",
                "evaluation_orientations": "evaluation_orientations.csv",
                "selected_orientations": "selected_orientations.csv",
                "v000": "v000.csv",
                "v000_plot": "v000.png",
            },
        }
        write_json(temporary / "manifest.json", manifest)
        temporary.rename(output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_full_grid_reference(
    data_path: str | Path,
    output_directory: str | Path,
    *,
    rcond: float | None = None,
    expected_points_per_radius: int | None = 500,
) -> Path:
    """Run the neutral raw-potential full-grid reference method."""

    return run_fit(
        data_path,
        output_directory,
        FullGridLeastSquares(rcond=rcond),
        expected_points_per_radius=expected_points_per_radius,
    )


def run_doptimal_candidate(
    data_path: str | Path,
    output_directory: str | Path,
    *,
    point_count: int = 180,
    rcond: float | None = None,
    recompute_interval: int = 50,
    expected_points_per_radius: int | None = 500,
) -> Path:
    """Run the deterministic energy-blind D-optimal-style candidate method."""

    return run_fit(
        data_path,
        output_directory,
        QRGreedyDOptimalLeastSquares(
            point_count=point_count,
            rcond=rcond,
            recompute_interval=recompute_interval,
        ),
        expected_points_per_radius=expected_points_per_radius,
    )
