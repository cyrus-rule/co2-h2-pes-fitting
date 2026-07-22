# CO₂–H₂ PES fitting benchmark and integration layer

This repository turns an angular ab initio energy grid into explicit,
reproducible fitting artifacts that can be compared across methods and passed to
YUMI. It is the working memory of the fitting project; the notebooks are
provenance records, not the source of scientific truth.

The present model is

\[
V(R,\Omega)=\sum_q c_q(R)A_q(\Omega),
\qquad \Omega=(\theta_1,\theta_2,\phi),
\qquad q=(\ell_1,\ell_2,L).
\]

## What works now

- A validated loader for the 30-radius, shared-orientation energy grid.
- A versioned 158-term *candidate* angular basis and tuple-keyed coefficient
  interchange.
- A method interface with a neutral full-grid raw-potential reference fit.
- A deterministic, energy-blind QR-seeded greedy D-optimal-style subset fit,
  promoted as the explicitly provisional `doptimal180_candidate`.
- A versioned evaluation specification whose tolerance, bands, and wall
  thresholds are recorded in every run manifest.
- Immutable run directories containing inputs, solver choices, coefficients,
  selected/unselected errors, wall guardrails, selected orientations,
  conditioning, and the \(V_{000}\) diagnostic.
- One-to-one comparison of any two standardized runs.
- A structural YUMI template parser/writer that maps
  `(l1, 0, l2, L) <-> (l1, l2, L)`, writes omitted terms as `0.0`, and rejects
  NaN or infinity.
- A reproducible 158--200 point design study, deterministic off-grid
  model--model stress tests, coefficient/radial holdouts, and a frozen
  48-orientation independent-ab-initio request.

## Deliberate scientific limits

- `co2-h2-candidate-158-v1` reproduces the July 2026 notebook convention, but
  is not called canonical until its exact omission list is checked against a
  trusted production/YUMI term list.
- `full500_raw_reference` is the reference within the supplied angular dataset,
  not an independently validated production PES.
- `doptimal180_candidate` is evidence about reconstruction on the known
  500-orientation candidate grid. It is not yet evidence that 180 new ab initio
  orientations are sufficient away from that grid.
- The radial short/intermediate/long-range join and the downstream
  pressure-broadening benchmark are not implemented yet.
- The historical high-energy switch is preserved as a labelled negative-result
  method component; it is not the default target.

See [`docs/basis_contract.md`](docs/basis_contract.md),
[`docs/artifact_contract.md`](docs/artifact_contract.md), and
[`docs/evaluation_contract.md`](docs/evaluation_contract.md) before comparing
results. The YUMI boundary is documented in
[`docs/yumi_contract.md`](docs/yumi_contract.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests
```

The ab initio input is intentionally untracked. Place an authorized copy at
`all_avcbs_uniform.dat`; its expected public metadata is recorded in
`data/manifests/all_avcbs_uniform.json`.

## Produce the reference artifact

```bash
co2-h2-pes fit-full all_avcbs_uniform.dat outputs/local/full500_raw_reference
```

The run refuses to overwrite an existing directory. It writes:

```text
manifest.json
coefficients.csv
metrics.csv
guardrail.csv
evaluation_orientations.csv
selected_orientations.csv
v000.csv
v000.png
```

`manifest.json` records the input SHA-256, basis version and hash, target,
solver, explicit `rcond`, separate evaluation- and fit-design shapes,
conditioning, units, software versions, and reconstruction round-trip check.

## Produce the reduced-sampling candidate

```bash
co2-h2-pes fit-doptimal \
  all_avcbs_uniform.dat \
  outputs/local/doptimal180_candidate \
  --points 180
```

The orientation order is selected from the angular design matrix alone. The
potential energies do not influence selection. The first 158 rows are a
full-rank pivoted-QR seed; the remaining rows greedily maximize immediate
information-determinant gain. See
[`docs/methods/doptimal180_candidate.md`](docs/methods/doptimal180_candidate.md).

## Compare methods

Every method must emit the same artifact contract. Once a second run exists:

```bash
co2-h2-pes compare \
  outputs/local/full500_raw_reference \
  outputs/local/another_method \
  outputs/local/comparisons/reference-vs-another
```

The comparison aligns by `(R, l1, l2, L)`, never by array position, evaluates
both surfaces on the common full orientation grid, and reports coefficient,
reconstructed-potential, and \(V_{000}\) differences.
It also writes `summary.csv` and a short `summary.md` containing the worst
finite-grid discrepancies and their radii.

## Reproduce the evidence behind 180 points

```bash
co2-h2-pes design-study \
  all_avcbs_uniform.dat \
  outputs/local/doptimal_design_study_v1
```

This reproduces the complete 158--200 count sweep, every single substitution
at 161/175/180/182 points, targeted label-free reoptimization, and the 36-rule
criterion-sensitivity grid. The curated result is under
`data/reference/doptimal_design_study_v1/`.

## Stress the fitted surfaces away from the known grid

```bash
co2-h2-pes offgrid-compare \
  outputs/local/full500_raw_reference \
  outputs/local/doptimal180_candidate \
  outputs/local/offgrid_full500-vs-doptimal180
```

These Sobol and local-minimum results are explicitly model--model diagnostics:
there is no independent truth at those angles. The curated real-data artifact
is under `data/reference/offgrid_diagnostics_v1/`.

## Generate the frozen independent-validation request

```bash
co2-h2-pes validation-request \
  outputs/local/full500_raw_reference \
  outputs/local/doptimal180_candidate \
  outputs/local/validation_request_v3
```

The authoritative coordinate files are versioned under
`data/reference/validation_request_v3/`: 48 new orientations, 144 primary
energies at 4.4/4.8/5.0 Bohr, and an optional 24-energy radial challenge at
4.7/4.9 Bohr. Returned primary labels can be scored without changing the rule:

```bash
co2-h2-pes validation-score \
  returned_primary_energies.csv \
  outputs/local/full500_raw_reference \
  outputs/local/doptimal180_candidate \
  outputs/local/independent_validation_score
```

## Diagnose radial behavior

```bash
co2-h2-pes radial-diagnostics \
  all_avcbs_uniform.dat \
  outputs/local/full500_raw_reference \
  outputs/local/doptimal180_candidate \
  outputs/local/radial_diagnostics_v1
```

The cubic spline here is a leave-one-node-out diagnostic, not the production
short/intermediate/long-range joining algorithm.

## Fill a supplied YUMI template

```bash
co2-h2-pes yumi-fill \
  CO2-H2.txt \
  outputs/local/full500_raw_reference/coefficients.csv \
  outputs/local/full500_raw_reference/yumi_input.txt \
  --radial-count 30
```

The template defines the required terms and preserves the header verbatim. The
header's `68` semantics and whether the current production interface expects 69
blocks or a larger maximum list still require group confirmation.

## Repository map

- `src/co2_h2_pes/`: basis, methods, artifact pipeline, comparisons, and YUMI I/O.
- `docs/decisions/`: durable records of project pivots and negative results.
- `docs/methods/`: experimental design and status of each promoted method.
- `docs/protocols/`: preregistered comparisons that await external inputs.
- `notebooks/`: preserved exploratory provenance.
- `data/reference/`: small versioned contracts, never private raw energies.
- `outputs/reference/`: only group-approved, reviewable reference artifacts.
- `tests/`: synthetic unit and pipeline tests that require no private dataset.
