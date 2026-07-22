# Promote the 180-point D-optimal-style design as a candidate method

**Status:** accepted as an executable comparison method; not accepted as a
production sampling budget.

## Context

The meeting notebook found that a deterministic 175--182 point design could
reconstruct the known 500-orientation grid substantially better than uniform
random subsets. Those results were trapped in notebook state and could not be
reproduced through the repository's standardized artifact interface.

## Decision

Implement the QR-seeded greedy ordering as package code and promote the
180-point prefix as `doptimal180_candidate`.

The method must:

- use only angular basis values for selection;
- retain the raw potential target;
- record the ordered stable orientation IDs;
- distinguish the 180-by-158 fit design from the 500-by-158 evaluation design;
- report selected and unselected-complement metrics separately;
- serialize all 158 coefficients, even though only 180 energies are used.

## Consequences

The full-500 and reduced-180 branches can now be compared coefficient by
coefficient and as reconstructed surfaces. The code can also produce an exact
orientation request for external calculation.

No statement about a generally sufficient 180-point ab initio budget follows
until independent off-grid energies are evaluated. The candidate basis and
YUMI maximum-term semantics remain separate unresolved contracts.
