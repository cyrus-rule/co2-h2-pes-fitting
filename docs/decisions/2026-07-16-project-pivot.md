# 2026-07-16: repository becomes the project product

## Decision

Organize the work around a program that accepts ab initio potentials and emits
comparable coefficient artifacts and YUMI inputs. Preserve notebooks as
provenance rather than maintaining a presentation narrative as the core product.

## Meeting requirements captured

- Record approaches, successes, failures, and applicability.
- Compare methods coefficient by coefficient.
- Test whether fitting differences change pressure broadening.
- Supply YUMI's maximum required coefficient structure with exact zeros.
- Plot and inspect `V000`, especially sign and radial smoothness.
- Eventually join short, intermediate, and analytic long-range behavior.
- Generalize later to harder molecular systems and adaptive point requests.

## Consequence

All promoted methods use the artifact contract. YUMI and downstream scattering
are consumers of those artifacts, not ad hoc notebook exports.
