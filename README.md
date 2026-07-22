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
- Immutable run directories containing inputs, solver choices, coefficients,
  errors, selected orientations, conditioning, and the \(V_{000}\) diagnostic.
- One-to-one comparison of any two standardized runs.
- A structural YUMI template parser/writer that maps
  `(l1, 0, l2, L) <-> (l1, l2, L)`, writes omitted terms as `0.0`, and rejects
  NaN or infinity.

## Deliberate scientific limits

- `co2-h2-candidate-158-v1` reproduces the July 2026 notebook convention, but
  is not called canonical until its exact omission list is checked against a
  trusted production/YUMI term list.
- `full500_raw_reference` is the reference within the supplied angular dataset,
  not an independently validated production PES.
- The radial short/intermediate/long-range join and the downstream
  pressure-broadening benchmark are not implemented yet.
- The historical high-energy switch is preserved as a labelled negative-result
  method component; it is not the default target.

See [`docs/basis_contract.md`](docs/basis_contract.md),
[`docs/artifact_contract.md`](docs/artifact_contract.md), and
[`docs/yumi_contract.md`](docs/yumi_contract.md) before comparing results.

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
evaluation_orientations.csv
selected_orientations.csv
v000.csv
v000.png
```

`manifest.json` records the input SHA-256, basis version and hash, target,
solver, explicit `rcond`, rank, singular values, condition number, units,
software versions, and reconstruction round-trip check.

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
- `notebooks/`: preserved exploratory provenance.
- `data/reference/`: small versioned contracts, never private raw energies.
- `outputs/reference/`: only group-approved, reviewable reference artifacts.
- `tests/`: synthetic unit and pipeline tests that require no private dataset.
