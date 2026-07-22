# Standardized run artifact contract

A fitting method is scientifically comparable only after it produces one
immutable run directory. The directory is the unit of review, exchange, and
downstream benchmarking.

## Required files

| File | Role |
|---|---|
| `manifest.json` | Input identity, method, solver policy, basis, units, conditioning, validation, and environment |
| `coefficients.csv` | One tuple-keyed row per `(method, R, l1, l2, L)` |
| `metrics.csv` | Errors by radius, energy band, and full/selected/unselected evaluation subset |
| `guardrail.csv` | Wall-opening diagnostics where the true potential is at least 5000 cm^-1 |
| `selected_orientations.csv` | Ordered design position, stable ID, and angles used by the method |
| `evaluation_orientations.csv` | Common full grid used to compare reconstructed surfaces |
| `v000.csv` | `c000`, `A000`, isotropic contribution, and radial derivative |
| `v000.png` | Immediate sign and smoothness diagnostic |

## Invariants

- Coefficients are finite and uniquely keyed by `(R, l1, l2, L)`.
- A full candidate-basis run has exactly `158 * n_radii` coefficient rows.
- Reloading `coefficients.csv` and reconstructing `Xc` agrees with the in-memory
  fit to the recorded tolerance.
- `rcond` is always present. `null` means NumPy's machine-precision default; a
  number is an explicit truncated-SVD policy.
- Evaluation-grid shape/rank/conditioning and fit-subset
  shape/rank/conditioning are recorded separately. They must not be conflated
  for reduced-point methods.
- Reduced-point metrics include the unselected complement. Errors on selected
  rows alone are training residuals, not evidence of reconstruction quality.
- Hybrid normalized error uses
  `abs(prediction - truth) / max(1 cm^-1, 0.01 * abs(truth))`.
- The input is identified by SHA-256, byte count, row count, radii, and units.
- The basis is identified by a versioned name and ordered-tuple hash.
- Existing run directories are never overwritten.

## Method interface

Methods implement `AngularFitMethod.fit(design_matrix, radial_potentials)` and
return `FitResult`. Artifact generation is shared, so methods cannot silently
change units, tuple conventions, metrics, or serialization.

The promoted methods are `full500_raw_reference` and the provisional
`doptimal180_candidate`. Future methods should use descriptive IDs such as
`rohan_reduced_basis_v1`.

## Versioning and publication

Local runs belong under `outputs/local/`. A small artifact may move to
`outputs/reference/` only after data-release and group approval. The manifest
must remain with every derived file.
