# CO₂–H₂ potential-energy-surface fitting

This repository contains Cyrus Rule's reproducible short-range fitting work for the CO₂–H₂ intermolecular potential-energy surface. The present calculation represents the angular dependence at each center-of-mass separation as

\[
V(R, \Omega) = \sum_q c_q(R) A_q(\Omega),
\qquad q = (\ell_1, \ell_2, L).
\]

The repository begins by preserving the two research notebooks and extracting their stable shared machinery into a small tested Python package. Exploratory experiments and scientific conclusions remain in the notebooks until they have been deliberately promoted into reusable code or documentation.

## Current contents

- `notebooks/co2_h2_short_range_methodology_v2_meeting.ipynb`: current complete research and meeting artifact.
- `notebooks/archive/short_range_fit_diagnostics.ipynb`: original diagnostic notebook, preserved for provenance.
- `src/co2_h2_pes/`: reusable basis, data, fitting, and coefficient-table functions.
- `tests/`: checks for the 158-term basis, switching function, and tuple-keyed coefficient round trips.
- `docs/basis_contract.md`: the current explicit convention for the angular basis.

## Data

The ab initio input `all_avcbs_uniform.dat` is intentionally not tracked. Place an authorized copy at the repository root before running the notebooks. The expected columns are

```text
R  Theta_1  Theta_2  Phi  V
```

The supplied research papers and internal meeting notes are also not committed. Repository visibility and data-release permission should be settled with the research group before any public release.

## Setup

Create an environment with Python 3.10 or newer, then install the project and development tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

Run the test suite:

```bash
python -m unittest discover -s tests
```

Export the canonical basis table:

```bash
python scripts/export_basis.py
```

The script writes `data/reference/co2_h2_canonical_158_term_basis.csv`, which can be compared with another implementation by the index tuple `(l1, l2, L)` rather than by column position.

## Reproducibility boundary

The full 500-orientation calculation is the reference fit within the available angular dataset. Reduced-point fits are prospective efficiency experiments. Neither should be treated as an independently validated production surface until the agreed coefficient and downstream-scattering checks are complete.
