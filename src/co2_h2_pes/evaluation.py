"""Versioned physical scoring rules shared by every fitted method."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class EvaluationSpec:
    """Complete, serializable definition of a PES evaluation rule."""

    id: str
    absolute_tolerance_floor_cm1: float
    relative_tolerance_fraction: float
    attractive_upper_cm1: float
    low_repulsive_upper_cm1: float
    lower_wall_upper_cm1: float
    physical_ceiling_cm1: float
    guardrail_warning_cm1: float
    guardrail_dangerous_cm1: float

    def __post_init__(self) -> None:
        boundaries = (
            self.attractive_upper_cm1,
            self.low_repulsive_upper_cm1,
            self.lower_wall_upper_cm1,
            self.physical_ceiling_cm1,
        )
        if self.absolute_tolerance_floor_cm1 <= 0.0:
            raise ValueError("The absolute tolerance floor must be positive.")
        if self.relative_tolerance_fraction <= 0.0:
            raise ValueError("The relative tolerance fraction must be positive.")
        if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("Energy-band boundaries must be strictly increasing.")
        if not (
            self.guardrail_dangerous_cm1
            < self.guardrail_warning_cm1
            <= self.physical_ceiling_cm1
        ):
            raise ValueError(
                "Guardrail dangerous < warning <= physical ceiling is required."
            )

    def to_manifest(self) -> dict[str, object]:
        """Return a JSON-safe mapping with the scoring equations made explicit."""

        payload: dict[str, object] = asdict(self)
        payload["hybrid_tolerance"] = (
            "max(absolute_tolerance_floor_cm1, "
            "relative_tolerance_fraction * abs(V_true))"
        )
        payload["guardrail_definition"] = (
            "V_true >= physical_ceiling_cm1; flag predictions below "
            "guardrail_warning_cm1 and guardrail_dangerous_cm1"
        )
        return payload


DEFAULT_EVALUATION_SPEC = EvaluationSpec(
    id="co2-h2-hybrid-1cm-1pct-wall-v1",
    absolute_tolerance_floor_cm1=1.0,
    relative_tolerance_fraction=0.01,
    attractive_upper_cm1=0.0,
    low_repulsive_upper_cm1=1000.0,
    lower_wall_upper_cm1=3000.0,
    physical_ceiling_cm1=5000.0,
    guardrail_warning_cm1=3000.0,
    guardrail_dangerous_cm1=1000.0,
)


def hybrid_tolerance(
    potential: ArrayLike,
    spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> NDArray[np.float64]:
    """Return the versioned pointwise tolerance in inverse centimetres."""

    values = np.asarray(potential, dtype=float)
    return np.maximum(
        spec.absolute_tolerance_floor_cm1,
        spec.relative_tolerance_fraction * np.abs(values),
    )


def energy_band_masks(
    potential: ArrayLike,
    spec: EvaluationSpec = DEFAULT_EVALUATION_SPEC,
) -> tuple[tuple[str, NDArray[np.bool_]], ...]:
    """Return stable names and masks for every physical scoring band."""

    values = np.asarray(potential, dtype=float)
    attractive = f"{spec.attractive_upper_cm1:g}"
    low = f"{spec.low_repulsive_upper_cm1:g}"
    lower_wall = f"{spec.lower_wall_upper_cm1:g}"
    ceiling = f"{spec.physical_ceiling_cm1:g}"
    return (
        ("all", np.ones(values.shape, dtype=bool)),
        (f"attractive_V<{attractive}", values < spec.attractive_upper_cm1),
        (
            f"low_repulsive_{attractive}<=V<{low}",
            (values >= spec.attractive_upper_cm1)
            & (values < spec.low_repulsive_upper_cm1),
        ),
        (
            f"lower_wall_{low}<=V<{lower_wall}",
            (values >= spec.low_repulsive_upper_cm1)
            & (values < spec.lower_wall_upper_cm1),
        ),
        (
            f"upper_wall_{lower_wall}<=V<{ceiling}",
            (values >= spec.lower_wall_upper_cm1)
            & (values < spec.physical_ceiling_cm1),
        ),
        (f"guardrail_V>={ceiling}", values >= spec.physical_ceiling_cm1),
    )
