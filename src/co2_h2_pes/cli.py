"""Command-line entry points for reproducible fits and interfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .comparison import compare_runs
from .basis import CANDIDATE_BASIS_V1
from .design_study import run_design_study
from .offgrid import run_offgrid_comparison
from .pipeline import run_doptimal_candidate, run_full_grid_reference
from .radial import run_radial_diagnostics
from .validation import run_validation_request, run_validation_scoring
from .yumi import fill_yumi_template, parse_yumi_template


def _rcond(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="co2-h2-pes")
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit-full", help="run the raw full-grid reference fit")
    fit.add_argument("data", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--rcond", type=_rcond, default=None)

    doptimal = commands.add_parser(
        "fit-doptimal",
        help="run the energy-blind QR + greedy D-optimal-style candidate fit",
    )
    doptimal.add_argument("data", type=Path)
    doptimal.add_argument("output", type=Path)
    doptimal.add_argument("--points", type=int, default=180)
    doptimal.add_argument("--rcond", type=_rcond, default=None)
    doptimal.add_argument("--recompute-interval", type=int, default=50)

    compare = commands.add_parser("compare", help="compare two run directories")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("output", type=Path)

    offgrid = commands.add_parser(
        "offgrid-compare",
        help="compare two fitted surfaces on deterministic unlabelled Sobol grids",
    )
    offgrid.add_argument("reference", type=Path)
    offgrid.add_argument("candidate", type=Path)
    offgrid.add_argument("output", type=Path)
    offgrid.add_argument("--sample-size", type=int, default=4096)
    offgrid.add_argument("--seed", type=int, default=20260715)

    request = commands.add_parser(
        "validation-request",
        help="generate the frozen 48-orientation independent-ab-initio request",
    )
    request.add_argument("reference", type=Path)
    request.add_argument("candidate", type=Path)
    request.add_argument("output", type=Path)
    request.add_argument("--sample-size", type=int, default=4096)
    request.add_argument("--offgrid-seed", type=int, default=20260715)
    request.add_argument("--request-seed", type=int, default=20260716)

    score = commands.add_parser(
        "validation-score",
        help="apply the frozen rule to returned primary validation energies",
    )
    score.add_argument("returned_energies", type=Path)
    score.add_argument("reference", type=Path)
    score.add_argument("candidate", type=Path)
    score.add_argument("output", type=Path)

    study = commands.add_parser(
        "design-study",
        help="reproduce the D-optimal count and robustness evidence",
    )
    study.add_argument("data", type=Path)
    study.add_argument("output", type=Path)

    radial = commands.add_parser(
        "radial-diagnostics",
        help="compare coefficient curves and run leave-one-R-out spline tests",
    )
    radial.add_argument("data", type=Path)
    radial.add_argument("reference", type=Path)
    radial.add_argument("candidate", type=Path)
    radial.add_argument("output", type=Path)

    yumi = commands.add_parser("yumi-fill", help="fill a supplied YUMI template")
    yumi.add_argument("template", type=Path)
    yumi.add_argument("coefficients", type=Path)
    yumi.add_argument("output", type=Path)
    yumi.add_argument("--radial-count", type=int, required=True)
    yumi.add_argument("--header-line-count", type=int, default=1)
    yumi.add_argument("--method")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit-full":
        result = run_full_grid_reference(args.data, args.output, rcond=args.rcond)
    elif args.command == "fit-doptimal":
        result = run_doptimal_candidate(
            args.data,
            args.output,
            point_count=args.points,
            rcond=args.rcond,
            recompute_interval=args.recompute_interval,
        )
    elif args.command == "compare":
        result = compare_runs(args.left, args.right, args.output)
    elif args.command == "offgrid-compare":
        result = run_offgrid_comparison(
            args.reference,
            args.candidate,
            args.output,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    elif args.command == "validation-request":
        result = run_validation_request(
            args.reference,
            args.candidate,
            args.output,
            sample_size=args.sample_size,
            offgrid_seed=args.offgrid_seed,
            request_seed=args.request_seed,
        )
    elif args.command == "validation-score":
        result = run_validation_scoring(
            args.returned_energies,
            args.reference,
            args.candidate,
            args.output,
        )
    elif args.command == "design-study":
        result = run_design_study(args.data, args.output)
    elif args.command == "radial-diagnostics":
        result = run_radial_diagnostics(
            args.data,
            args.reference,
            args.candidate,
            args.output,
        )
    elif args.command == "yumi-fill":
        template = parse_yumi_template(
            args.template,
            radial_count=args.radial_count,
            header_line_count=args.header_line_count,
        )
        template.validate_terms(CANDIDATE_BASIS_V1)
        coefficients = pd.read_csv(args.coefficients)
        filled = fill_yumi_template(template, coefficients, method=args.method)
        args.output.write_text(filled.to_text(), encoding="utf-8")
        result = args.output
    else:  # pragma: no cover - argparse enforces the command universe.
        raise AssertionError(f"Unhandled command: {args.command}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
