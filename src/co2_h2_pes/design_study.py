"""Reproducible evidence supporting the provisional 180-point design choice."""

from __future__ import annotations

import shutil
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .artifacts import (
    RUN_SCHEMA_VERSION,
    current_git_revision,
    software_versions,
    utc_timestamp,
    write_json,
)
from .basis import BASIS_ID, CANDIDATE_BASIS_V1, build_design_matrix
from .data import (
    angles_in_radians,
    load_ab_initio_data,
    radial_potential_matrix,
    reference_orientations,
    sha256_file,
)
from .evaluation import DEFAULT_EVALUATION_SPEC, EvaluationSpec, hybrid_tolerance
from .methods import qr_seeded_greedy_d_optimal_order, whiten_candidate_design

DESIGN_STUDY_ID = "doptimal-count-robustness-v1"
DEFAULT_COUNTS = tuple(range(len(CANDIDATE_BASIS_V1), 201))
SUBSTITUTION_COUNTS = (161, 175, 180, 182)
SENSITIVITY_COUNTS = (161, 175, 180, 182, 200)


def evaluate_fixed_subset(
    design: NDArray[np.float64],
    truth: NDArray[np.float64],
    radii: NDArray[np.float64],
    selected_ids: NDArray[np.int64],
    *,
    evaluation_spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
    rcond: float | None = None,
) -> dict[str, object]:
    """Apply the strict finite-grid rule to one shared angular subset."""

    design = np.asarray(design, dtype=float)
    truth = np.asarray(truth, dtype=float)
    radii = np.asarray(radii, dtype=float)
    selected = np.asarray(selected_ids, dtype=int)
    if design.ndim != 2 or truth.ndim != 2:
        raise ValueError("Design and truth must be matrices.")
    if design.shape[0] != truth.shape[0] or truth.shape[1] != len(radii):
        raise ValueError("Design, truth, and radii have incompatible shapes.")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Selected orientation IDs must be unique.")
    if len(selected) == 0 or np.any(selected < 0) or np.any(selected >= len(design)):
        raise ValueError("Selected orientation IDs are outside the design matrix.")

    selected_mask = np.zeros(design.shape[0], dtype=bool)
    selected_mask[selected] = True
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design[selected], truth[selected], rcond=rcond
    )
    prediction = design @ coefficients
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    radial_rows: list[dict[str, object]] = []
    for column, radius in enumerate(radii):
        target = truth[:, column]
        estimated = prediction[:, column]
        relevant = target < evaluation_spec.physical_ceiling_cm1
        normalized = np.abs(estimated - target) / hybrid_tolerance(
            target, evaluation_spec
        )
        complement_relevant = relevant & ~selected_mask
        guardrail = target >= evaluation_spec.physical_ceiling_cm1
        if guardrail.any():
            minimum_guardrail = float(estimated[guardrail].min())
            guardrail_failure = bool(
                np.any(
                    estimated[guardrail] < evaluation_spec.guardrail_warning_cm1
                )
            )
        else:
            minimum_guardrail = float("nan")
            guardrail_failure = False
        radial_rows.append(
            {
                "R": float(radius),
                "full_maximum_normalized_error": float(normalized[relevant].max()),
                "complement_maximum_normalized_error": (
                    float(normalized[complement_relevant].max())
                    if complement_relevant.any()
                    else 0.0
                ),
                "minimum_guardrail_prediction": minimum_guardrail,
                "guardrail_failure": guardrail_failure,
            }
        )
    radial = pd.DataFrame(radial_rows)
    limiting = radial.loc[radial["full_maximum_normalized_error"].idxmax()]
    worst = float(limiting["full_maximum_normalized_error"])
    finite_guardrail = radial["minimum_guardrail_prediction"].dropna()
    guardrail_failure = bool(radial["guardrail_failure"].any())
    return {
        "retained_rank": int(rank),
        "condition_number": condition,
        "worst_normalized_error": worst,
        "tolerance_headroom": 1.0 - worst,
        "tolerance_safety_factor": 1.0 / worst if worst > 0.0 else float("inf"),
        "limiting_R": float(limiting["R"]),
        "minimum_guardrail_prediction": (
            float(finite_guardrail.min()) if len(finite_guardrail) else float("nan")
        ),
        "any_guardrail_below_warning": guardrail_failure,
        "strict_pass": bool(
            rank == design.shape[1] and worst <= 1.0 and not guardrail_failure
        ),
        "coefficients": coefficients,
        "prediction": prediction,
        "radial_diagnostics": radial,
    }


def _public_metrics(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"coefficients", "prediction", "radial_diagnostics"}
    }


def _summary_markdown(
    count_sweep: pd.DataFrame,
    substitutions: pd.DataFrame,
    reoptimized: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> str:
    passing = count_sweep.loc[count_sweep["strict_pass"], "training_points"]
    first_pass = int(passing.min()) if len(passing) else None
    row_180 = count_sweep.loc[count_sweep["training_points"] == 180].iloc[0]
    substitution_180 = substitutions.loc[
        substitutions["training_points"] == 180
    ].iloc[0]
    sensitivity_180 = sensitivity.loc[sensitivity["training_points"] == 180].iloc[0]
    recovered = int(reoptimized["reoptimized_strict_pass"].sum()) if len(reoptimized) else 0
    return (
        "# D-optimal count and robustness study\n\n"
        f"- First strict known-grid pass: {first_pass} orientations.\n"
        f"- The 180-point baseline safety factor is "
        f"{row_180['tolerance_safety_factor']:.6g}.\n"
        f"- 180-point single-substitution pass rate: "
        f"{substitution_180['strict_pass_rate_percent']:.6g}%.\n"
        f"- Targeted label-free reoptimization recovered {recovered} of "
        f"{len(reoptimized)} observed 175--182 failures.\n"
        f"- The minimum 180-point safety factor over "
        f"{int(sensitivity_180['specifications'])} scoring specifications is "
        f"{sensitivity_180['minimum_safety_factor']:.6g}.\n\n"
        "These are retrospective finite-grid results. They support 180 as a "
        "robust planning choice, not as a mathematically optimal or independently "
        "validated ab initio budget.\n"
    )


def run_design_study(
    data_path: str | Path,
    output_directory: str | Path,
    *,
    minimum_R: float = 4.4,
    maximum_R: float = 6.6,
    counts: tuple[int, ...] = DEFAULT_COUNTS,
    substitution_counts: tuple[int, ...] = SUBSTITUTION_COUNTS,
    sensitivity_counts: tuple[int, ...] = SENSITIVITY_COUNTS,
    rcond: float | None = None,
    evaluation_spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> Path:
    """Write the count sweep, removal robustness, and criterion sensitivity."""

    source = Path(data_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Design-study directory exists: {output}")
    frame = load_ab_initio_data(source)
    orientations = reference_orientations(frame)
    all_radii, all_truth = radial_potential_matrix(frame)
    radial_mask = (all_radii >= minimum_R) & (all_radii <= maximum_R)
    radii = all_radii[radial_mask]
    truth = all_truth[:, radial_mask]
    design = build_design_matrix(*angles_in_radians(orientations))
    order = qr_seeded_greedy_d_optimal_order(design)
    leverage = np.sum(whiten_candidate_design(design) ** 2, axis=1)

    count_rows: list[dict[str, object]] = []
    baseline_results: dict[int, dict[str, object]] = {}
    for count in counts:
        if not design.shape[1] <= count <= design.shape[0]:
            raise ValueError(f"Invalid design-study count: {count}")
        result = evaluate_fixed_subset(
            design,
            truth,
            radii,
            order[:count],
            evaluation_spec=evaluation_spec,
            rcond=rcond,
        )
        baseline_results[count] = result
        count_rows.append({"training_points": count, **_public_metrics(result)})
    count_sweep = pd.DataFrame(count_rows)

    substitution_rows: list[dict[str, object]] = []
    for count in substitution_counts:
        selected = np.asarray(order[:count], dtype=int)
        replacement = int(order[count])
        for position in range(count):
            modified = selected.copy()
            removed = int(modified[position])
            modified[position] = replacement
            result = evaluate_fixed_subset(
                design,
                truth,
                radii,
                modified,
                evaluation_spec=evaluation_spec,
                rcond=rcond,
            )
            substitution_rows.append(
                {
                    "training_points": count,
                    "removed_position": position + 1,
                    "removed_orientation": removed,
                    "removed_leverage": leverage[removed],
                    "replacement_orientation": replacement,
                    "replacement_leverage": leverage[replacement],
                    **_public_metrics(result),
                }
            )
    substitution_detail = pd.DataFrame(substitution_rows)
    substitution_summary_rows: list[dict[str, object]] = []
    for count in substitution_counts:
        rows = substitution_detail.loc[
            substitution_detail["training_points"] == count
        ]
        worst = rows.loc[rows["worst_normalized_error"].idxmax()]
        substitution_summary_rows.append(
            {
                "training_points": count,
                "variants_tested": len(rows),
                "strict_pass_rate_percent": 100.0 * rows["strict_pass"].mean(),
                "median_worst_normalized_error": rows[
                    "worst_normalized_error"
                ].median(),
                "p95_worst_normalized_error": np.percentile(
                    rows["worst_normalized_error"], 95
                ),
                "worst_normalized_error": worst["worst_normalized_error"],
                "worst_removed_position": int(worst["removed_position"]),
                "worst_removed_orientation": int(worst["removed_orientation"]),
                "worst_limiting_R": worst["limiting_R"],
                "rank_deficient_variants": int(
                    (rows["retained_rank"] < design.shape[1]).sum()
                ),
                "worst_condition_number": rows["condition_number"].max(),
                "minimum_guardrail_prediction": rows[
                    "minimum_guardrail_prediction"
                ].min(),
            }
        )
    substitution_summary = pd.DataFrame(substitution_summary_rows)

    failures = substitution_detail.loc[
        substitution_detail["training_points"].isin([175, 180, 182])
        & ~substitution_detail["strict_pass"]
    ].copy()
    reoptimized_orders: dict[int, NDArray[np.int64]] = {}
    for unavailable in sorted(failures["removed_orientation"].unique()):
        candidate_ids = np.setdiff1d(
            np.arange(design.shape[0]), np.array([unavailable]), assume_unique=True
        )
        local_order = qr_seeded_greedy_d_optimal_order(design[candidate_ids])
        reoptimized_orders[int(unavailable)] = candidate_ids[local_order]
    reoptimized_rows: list[dict[str, object]] = []
    for _, failure in failures.iterrows():
        count = int(failure["training_points"])
        unavailable = int(failure["removed_orientation"])
        selected = reoptimized_orders[unavailable][:count]
        result = evaluate_fixed_subset(
            design,
            truth,
            radii,
            selected,
            evaluation_spec=evaluation_spec,
            rcond=rcond,
        )
        retained = len(set(order[:count]).intersection(selected))
        reoptimized_rows.append(
            {
                "training_points": count,
                "unavailable_orientation": unavailable,
                "unavailable_leverage": leverage[unavailable],
                "simple_replacement_error": failure["worst_normalized_error"],
                "reoptimized_error": result["worst_normalized_error"],
                "reoptimized_safety_factor": result["tolerance_safety_factor"],
                "reoptimized_limiting_R": result["limiting_R"],
                "reoptimized_condition_number": result["condition_number"],
                "original_selected_points_retained": retained,
                "selected_points_changed": count - retained,
                "reoptimized_strict_pass": result["strict_pass"],
                "minimum_guardrail_prediction": result[
                    "minimum_guardrail_prediction"
                ],
            }
        )
    reoptimized = pd.DataFrame(reoptimized_rows)

    sensitivity_rows: list[dict[str, object]] = []
    for absolute_floor, relative_fraction, ceiling, warning in product(
        (0.5, 1.0, 2.0),
        (0.005, 0.01, 0.02),
        (4000.0, 5000.0),
        (3000.0, 4000.0),
    ):
        specification = EvaluationSpec(
            id=(
                f"sensitivity-floor{absolute_floor:g}-rel{relative_fraction:g}-"
                f"ceiling{ceiling:g}-warning{warning:g}"
            ),
            absolute_tolerance_floor_cm1=absolute_floor,
            relative_tolerance_fraction=relative_fraction,
            attractive_upper_cm1=0.0,
            low_repulsive_upper_cm1=1000.0,
            lower_wall_upper_cm1=min(3000.0, ceiling - 1.0),
            physical_ceiling_cm1=ceiling,
            guardrail_warning_cm1=warning,
            guardrail_dangerous_cm1=1000.0,
        )
        for count in sensitivity_counts:
            result = evaluate_fixed_subset(
                design,
                truth,
                radii,
                order[:count],
                evaluation_spec=specification,
                rcond=rcond,
            )
            sensitivity_rows.append(
                {
                    "absolute_floor": absolute_floor,
                    "relative_fraction": relative_fraction,
                    "physical_ceiling": ceiling,
                    "guardrail_warning": warning,
                    "training_points": count,
                    **_public_metrics(result),
                }
            )
    sensitivity_detail = pd.DataFrame(sensitivity_rows)
    sensitivity_summary = (
        sensitivity_detail.groupby("training_points", as_index=False)
        .agg(
            specifications=("strict_pass", "size"),
            specifications_passed=("strict_pass", "sum"),
            pass_rate_percent=("strict_pass", lambda values: 100.0 * values.mean()),
            minimum_safety_factor=("tolerance_safety_factor", "min"),
            median_safety_factor=("tolerance_safety_factor", "median"),
            maximum_safety_factor=("tolerance_safety_factor", "max"),
        )
    )

    order_table = orientations.iloc[order][
        ["orientation_id", "Theta_1", "Theta_2", "Phi"]
    ].copy()
    order_table.insert(0, "design_position", np.arange(1, len(order) + 1))
    order_table["leverage"] = leverage[order]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        tables = {
            "doptimal_order.csv": order_table,
            "count_sweep.csv": count_sweep,
            "single_substitution_detail.csv": substitution_detail,
            "single_substitution_summary.csv": substitution_summary,
            "reoptimized_recovery.csv": reoptimized,
            "criterion_sensitivity_detail.csv": sensitivity_detail,
            "criterion_sensitivity_summary.csv": sensitivity_summary,
        }
        for filename, table in tables.items():
            table.to_csv(temporary / filename, index=False, float_format="%.17g")
        (temporary / "summary.md").write_text(
            _summary_markdown(
                count_sweep, substitution_summary, reoptimized, sensitivity_summary
            ),
            encoding="utf-8",
        )
        passing = count_sweep.loc[count_sweep["strict_pass"], "training_points"]
        files = {
            **{filename.removesuffix(".csv"): filename for filename in tables},
            "summary": "summary.md",
        }
        write_json(
            temporary / "manifest.json",
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "artifact_id": DESIGN_STUDY_ID,
                "created_utc": utc_timestamp(),
                "evidence_type": "retrospective finite-grid design study",
                "input": {
                    "filename": source.name,
                    "sha256": sha256_file(source),
                    "rows": int(len(frame)),
                    "radii_bohr": radii.tolist(),
                    "orientation_count": int(len(orientations)),
                },
                "basis_id": BASIS_ID,
                "selection_algorithm": "qr_seeded_greedy_doptimal_style_v1",
                "selection_uses_potential_energies": False,
                "rcond": rcond,
                "evaluation_spec": evaluation_spec.to_manifest(),
                "counts": list(counts),
                "substitution_counts": list(substitution_counts),
                "sensitivity_counts": list(sensitivity_counts),
                "result": {
                    "first_strict_known_grid_pass": (
                        int(passing.min()) if len(passing) else None
                    ),
                    "planning_count": 180,
                    "planning_interpretation": (
                        "robust provisional choice, not a discovered optimum"
                    ),
                },
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
