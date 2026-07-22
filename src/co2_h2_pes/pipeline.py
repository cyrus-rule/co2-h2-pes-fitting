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
from .methods import AngularFitMethod, FullGridLeastSquares


def _error_metrics(
    radii: np.ndarray,
    reference: np.ndarray,
    reconstructed: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    bands = (
        ("all", lambda values: np.ones(values.shape, dtype=bool)),
        ("V<=1000", lambda values: values <= 1000.0),
        ("1000<V<=5000", lambda values: (values > 1000.0) & (values <= 5000.0)),
        ("V>5000", lambda values: values > 5000.0),
    )
    for column, radius in enumerate(radii):
        target = reference[:, column]
        residual = reconstructed[:, column] - target
        for band_name, selector in bands:
            mask = selector(target)
            if not mask.any():
                continue
            values = residual[mask]
            reference_rms = float(np.sqrt(np.mean(target[mask] ** 2)))
            rmse = float(np.sqrt(np.mean(values**2)))
            rows.append(
                {
                    "R": float(radius),
                    "energy_band": band_name,
                    "n_points": int(mask.sum()),
                    "rmse": rmse,
                    "mae": float(np.mean(np.abs(values))),
                    "median_abs_error": float(np.median(np.abs(values))),
                    "max_abs_error": float(np.max(np.abs(values))),
                    "reference_rms": reference_rms,
                    "normalized_rmse": rmse / reference_rms if reference_rms else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _v000_table(radii: np.ndarray, coefficients: np.ndarray) -> pd.DataFrame:
    index = CANDIDATE_BASIS_V1.index((0, 0, 0))
    a000 = 1.0 / (4.0 * np.sqrt(np.pi))
    contribution = coefficients[index] * a000
    derivative = np.gradient(contribution, radii, edge_order=2)
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

        metrics = _error_metrics(radii, potentials, reconstructed)
        metrics.to_csv(temporary / "metrics.csv", index=False, float_format="%.17g")

        v000 = _v000_table(radii, result.coefficients)
        v000.to_csv(temporary / "v000.csv", index=False, float_format="%.17g")
        _plot_v000(v000, temporary / "v000.png")

        selected = orientations.iloc[result.selected_orientation_ids][
            ["orientation_id", "Theta_1", "Theta_2", "Phi"]
        ]
        selected.to_csv(temporary / "selected_orientations.csv", index=False)
        orientations[["orientation_id", "Theta_1", "Theta_2", "Phi"]].to_csv(
            temporary / "evaluation_orientations.csv",
            index=False,
        )

        dataset_hash = sha256_file(data_path)
        last_v000 = float(v000.iloc[-1]["isotropic_contribution"])
        warnings: list[str] = []
        if last_v000 >= 0.0:
            warnings.append(
                "The isotropic contribution at the largest supplied R is not "
                "attractive; inspect the ab initio offset and long-range replacement."
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
                "shape": list(design.shape),
                "rank": result.rank,
                "condition_number": (
                    result.condition_number
                    if np.isfinite(result.condition_number)
                    else None
                ),
                "largest_singular_value": float(result.singular_values[0]),
                "smallest_singular_value": float(result.singular_values[-1]),
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
