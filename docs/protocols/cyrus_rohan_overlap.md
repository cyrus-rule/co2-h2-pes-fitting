# Cyrus--Rohan overlap protocol

Use 6.4--8.0 Bohr as a comparison window, not as a preselected splice. The
existing dataset contains nodes at 6.4, 6.6, 6.8, 7.0, 7.25, 7.5, 7.75, and
8.0 Bohr, so the first comparison needs no new ab initio labels.

Cyrus supplies the full-500 raw/candidate-basis reference artifact. Rohan's
method must emit the same artifact contract or an adapter must map its stored
`(L, l1, l2)` tuple into repository `(l1, l2, L)` keys. Positional coefficient
comparison is forbidden.

At every overlap node report:

- tuple membership and omitted channels;
- coefficient values, vector norms, and vector-angle similarity;
- reconstructed energies by physical band on the common 500-angle grid;
- first radial derivatives from each method's actual intended interpolation;
- wall guardrails and any basis-hysteresis changes.

A boundary may be proposed only after agreement persists over more than one
radial interval. This protocol cannot currently be executed because no
standardized Rohan artifact has been supplied.
