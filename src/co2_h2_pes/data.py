"""Loading and structural validation for the ab initio grid."""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

ANGLE_COLUMNS = ("Theta_1", "Theta_2", "Phi")
DATA_COLUMNS = ("R", *ANGLE_COLUMNS, "V")


def load_ab_initio_data(
    path: str | Path,
    *,
    expected_points_per_radius: int | None = 500,
    require_shared_angular_grid: bool = True,
) -> pd.DataFrame:
    """Load the whitespace-delimited PES data and assign stable orientation IDs."""

    path = Path(path)
    frame = pd.read_csv(path, sep=r"\s+", names=DATA_COLUMNS)

    if frame[list(DATA_COLUMNS)].isna().any().any():
        raise ValueError(f"{path} contains missing values.")
    if not np.isfinite(frame[list(DATA_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains NaN or infinity.")
    if frame.duplicated(subset=["R", *ANGLE_COLUMNS]).any():
        raise ValueError(f"{path} contains duplicate radial/angular geometries.")

    angle_index = pd.MultiIndex.from_frame(frame[list(ANGLE_COLUMNS)].round(8))
    frame["orientation_id"] = pd.factorize(angle_index, sort=True)[0]

    points_per_radius = frame.groupby("R").size()
    if expected_points_per_radius is not None and not points_per_radius.eq(
        expected_points_per_radius
    ).all():
        raise ValueError(
            "Expected "
            f"{expected_points_per_radius} points at every R; observed range "
            f"{points_per_radius.min()} to {points_per_radius.max()}."
        )

    if require_shared_angular_grid:
        expected_orientations = frame["orientation_id"].nunique()
        shared = (
            frame.groupby("R")["orientation_id"]
            .nunique()
            .eq(expected_orientations)
            .all()
        )
        if not shared:
            raise ValueError("The angular orientation grid is not shared across R.")

    return frame


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 fingerprint for a research input."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def radial_potential_matrix(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted radii and an ``(orientation, R)`` potential matrix."""

    required = {"R", "V", "orientation_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    pivot = (
        frame.pivot(index="orientation_id", columns="R", values="V")
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.isna().any().any():
        raise ValueError("The radial/angular grid is incomplete.")
    return pivot.columns.to_numpy(dtype=float), pivot.to_numpy(dtype=float)


def reference_orientations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per orientation in canonical orientation-ID order."""

    required = {"orientation_id", *ANGLE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    reference = (
        frame.drop_duplicates(subset="orientation_id")
        .sort_values("orientation_id")
        .reset_index(drop=True)
    )
    expected_ids = np.arange(len(reference))
    if not np.array_equal(reference["orientation_id"].to_numpy(), expected_ids):
        raise ValueError("Orientation IDs are not consecutive from zero.")
    return reference


def angles_in_radians(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``Theta_1``, ``Theta_2``, and ``Phi`` in radians."""

    return tuple(
        np.radians(frame[column].to_numpy(dtype=float)) for column in ANGLE_COLUMNS
    )

