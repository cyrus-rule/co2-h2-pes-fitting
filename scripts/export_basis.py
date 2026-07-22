"""Export the versioned candidate basis as a tuple-keyed CSV table."""

from pathlib import Path

from co2_h2_pes.coefficients import basis_table


def main() -> None:
    output = Path("data/reference/co2_h2_candidate_158_v1.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    basis_table().to_csv(output, index=False)
    print(f"Wrote {output}.")


if __name__ == "__main__":
    main()
