"""Tuple-keyed coefficient and reconstructed-potential comparisons."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import RUN_SCHEMA_VERSION, utc_timestamp, write_json
from .basis import CANDIDATE_BASIS_V1, build_design_matrix
from .coefficients import coefficient_table_to_matrix
from .data import sha256_file


def _comparison_summary_markdown(
    left_method: str,
    right_method: str,
    summary: pd.DataFrame,
) -> str:
    """Render the most decision-relevant comparison extrema."""

    worst_rmse = summary.loc[summary["rmse_difference"].idxmax()]
    worst_point = summary.loc[summary["max_abs_difference"].idxmax()]
    worst_v000 = summary.loc[summary["abs_v000_delta"].idxmax()]
    return (
        f"# {left_method} vs {right_method}\n\n"
        "Both reconstructed surfaces were evaluated on the shared full "
        "orientation grid. Coefficients were aligned by `(R, l1, l2, L)`.\n\n"
        f"- Largest potential RMSE difference: "
        f"{worst_rmse['rmse_difference']:.8g} cm^-1 at "
        f"R = {worst_rmse['R']:.8g} Bohr.\n"
        f"- Largest pointwise absolute potential difference: "
        f"{worst_point['max_abs_difference']:.8g} cm^-1 at "
        f"R = {worst_point['R']:.8g} Bohr.\n"
        f"- Largest absolute isotropic-contribution difference: "
        f"{worst_v000['abs_v000_delta']:.8g} cm^-1 at "
        f"R = {worst_v000['R']:.8g} Bohr.\n\n"
        "These are finite-grid differences, not independent ab initio "
        "validation errors. See `summary.csv` for every radius.\n"
    )


def _read_run(path: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    coefficients = pd.read_csv(path / "coefficients.csv")
    evaluation_path = path / "evaluation_orientations.csv"
    orientations = pd.read_csv(
        evaluation_path if evaluation_path.exists() else path / "selected_orientations.csv"
    )
    return manifest, coefficients, orientations


def compare_runs(
    left_directory: str | Path,
    right_directory: str | Path,
    output_directory: str | Path,
) -> Path:
    """Compare two standardized runs without relying on coefficient position."""

    left_path = Path(left_directory).resolve()
    right_path = Path(right_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Comparison directory already exists: {output}")

    left_manifest, left, left_angles = _read_run(left_path)
    right_manifest, right, right_angles = _read_run(right_path)
    if left_manifest["basis"]["id"] != right_manifest["basis"]["id"]:
        raise ValueError("Runs use different basis contracts.")

    angle_columns = ["orientation_id", "Theta_1", "Theta_2", "Phi"]
    left_angles = left_angles[angle_columns].sort_values("orientation_id")
    right_angles = right_angles[angle_columns].sort_values("orientation_id")
    if not left_angles.reset_index(drop=True).equals(right_angles.reset_index(drop=True)):
        raise ValueError("Runs do not use the same comparison orientation grid.")

    keys = ["R", "l1", "l2", "L"]
    left_method = str(left_manifest["method"]["id"])
    right_method = str(right_manifest["method"]["id"])
    merged = left[keys + ["coefficient"]].merge(
        right[keys + ["coefficient"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_left", "_right"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[merged["_merge"] != "both", keys + ["_merge"]]
        raise ValueError(f"Coefficient universes differ; first mismatch: {missing.iloc[0].to_dict()}")
    merged = merged.drop(columns="_merge")
    merged["delta"] = merged["coefficient_right"] - merged["coefficient_left"]
    merged["abs_delta"] = merged["delta"].abs()
    scale = merged[["coefficient_left", "coefficient_right"]].abs().max(axis=1)
    merged["symmetric_relative_delta"] = np.where(
        scale > 0.0,
        merged["abs_delta"] / scale,
        0.0,
    )
    merged.insert(0, "right_method", right_method)
    merged.insert(0, "left_method", left_method)

    radii = np.array(sorted(merged["R"].unique()), dtype=float)
    left_matrix = coefficient_table_to_matrix(left, radii, basis=CANDIDATE_BASIS_V1)
    right_matrix = coefficient_table_to_matrix(right, radii, basis=CANDIDATE_BASIS_V1)
    design = build_design_matrix(
        *(
            np.radians(left_angles[column].to_numpy(dtype=float))
            for column in ("Theta_1", "Theta_2", "Phi")
        )
    )
    potential_delta = design @ (right_matrix - left_matrix)
    potential_rows = []
    for column, radius in enumerate(radii):
        values = potential_delta[:, column]
        potential_rows.append(
            {
                "R": float(radius),
                "n_orientations": int(values.size),
                "rmse_difference": float(np.sqrt(np.mean(values**2))),
                "mae_difference": float(np.mean(np.abs(values))),
                "max_abs_difference": float(np.max(np.abs(values))),
            }
        )

    potential_summary = pd.DataFrame(potential_rows)
    coefficient_summary = (
        merged.groupby("R", as_index=False)
        .agg(
            rms_coefficient_delta=(
                "delta",
                lambda values: float(np.sqrt(np.mean(np.asarray(values) ** 2))),
            ),
            median_abs_coefficient_delta=("abs_delta", "median"),
            max_abs_coefficient_delta=("abs_delta", "max"),
            max_symmetric_relative_delta=("symmetric_relative_delta", "max"),
        )
        .sort_values("R")
    )

    zero_index = CANDIDATE_BASIS_V1.index((0, 0, 0))
    a000 = 1.0 / (4.0 * np.sqrt(np.pi))
    v000 = pd.DataFrame(
        {
            "R": radii,
            "left_isotropic_contribution": left_matrix[zero_index] * a000,
            "right_isotropic_contribution": right_matrix[zero_index] * a000,
        }
    )
    v000["delta"] = (
        v000["right_isotropic_contribution"]
        - v000["left_isotropic_contribution"]
    )
    summary = coefficient_summary.merge(
        potential_summary,
        on="R",
        how="inner",
        validate="one_to_one",
    ).merge(
        v000[["R", "delta"]].rename(columns={"delta": "v000_delta"}),
        on="R",
        how="inner",
        validate="one_to_one",
    )
    summary["abs_v000_delta"] = summary["v000_delta"].abs()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        merged.to_csv(
            temporary / "coefficient_differences.csv",
            index=False,
            float_format="%.17g",
        )
        potential_summary.to_csv(
            temporary / "potential_differences.csv",
            index=False,
            float_format="%.17g",
        )
        v000.to_csv(temporary / "v000_differences.csv", index=False, float_format="%.17g")
        summary.to_csv(
            temporary / "summary.csv",
            index=False,
            float_format="%.17g",
        )
        (temporary / "summary.md").write_text(
            _comparison_summary_markdown(left_method, right_method, summary),
            encoding="utf-8",
        )
        write_json(
            temporary / "manifest.json",
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "created_utc": utc_timestamp(),
                "left_run": left_path.name,
                "right_run": right_path.name,
                "left_manifest_sha256": sha256_file(left_path / "manifest.json"),
                "right_manifest_sha256": sha256_file(right_path / "manifest.json"),
                "left_method": left_method,
                "right_method": right_method,
                "basis_id": left_manifest["basis"]["id"],
                "radii_bohr": radii.tolist(),
                "orientation_count": int(len(left_angles)),
                "relative_delta_definition": (
                    "abs(right-left)/max(abs(left),abs(right)); zero when both are zero"
                ),
                "files": {
                    "coefficients": "coefficient_differences.csv",
                    "potentials": "potential_differences.csv",
                    "v000": "v000_differences.csv",
                    "summary": "summary.csv",
                    "summary_markdown": "summary.md",
                },
            },
        )
        temporary.rename(output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
