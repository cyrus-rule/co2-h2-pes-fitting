"""Stable metadata and filesystem helpers for research run artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from .basis import BasisIndex

RUN_SCHEMA_VERSION = "1.0"


def basis_fingerprint(basis: Iterable[BasisIndex]) -> str:
    """Hash an ordered basis independently of CSV formatting."""

    payload = "".join(f"{l1},{l2},{L}\n" for l1, l2, L in basis).encode()
    return hashlib.sha256(payload).hexdigest()


def selected_ids_fingerprint(ids: Iterable[int]) -> str:
    """Hash an ordered orientation selection."""

    payload = "".join(f"{int(value)}\n" for value in ids).encode()
    return hashlib.sha256(payload).hexdigest()


def current_git_revision(start: str | Path) -> str | None:
    """Return the local commit when run inside a checkout, otherwise ``None``."""

    completed = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def software_versions() -> dict[str, str]:
    """Return the numerical environment relevant to a fit."""

    packages = ("numpy", "pandas", "scipy", "sympy", "matplotlib")
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: object) -> None:
    """Write deterministic, human-readable JSON."""

    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
