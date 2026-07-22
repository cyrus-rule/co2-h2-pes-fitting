"""Command-line entry points for reproducible fits and interfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .comparison import compare_runs
from .basis import CANDIDATE_BASIS_V1
from .pipeline import run_doptimal_candidate, run_full_grid_reference
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
    else:
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
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
