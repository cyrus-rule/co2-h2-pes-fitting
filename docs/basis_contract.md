# Working angular-basis contract

The fitting model is

\[
V(R,\Omega)=\sum_q c_q(R)A_q(\Omega),
\qquad
\Omega=(\theta_1,\theta_2,\phi),
\qquad
q=(\ell_1,\ell_2,L).
\]

Here, (R) is the center-of-mass separation and (Omega) specifies the relative molecular orientations. The (A_q) are coupled angular functions. At each tabulated (R), the present workflow estimates the coefficients (c_q(R)) through a linear least-squares problem.

## Canonical 158-term universe

The current implementation generates the fitting universe by applying all of the following rules:

1. (ell_1 \in \{0,2,4,\ldots,24\}).
2. (ell_2 \in \{0,2,4,6\}).
3. If (ell_1 \ge 22), then (ell_2=0).
4. (|\ell_1-\ell_2|\le L\le\ell_1+\ell_2).
5. (ell_1+\ell_2+L) is even.

The even monomer indices encode the axis-reversal symmetry of the centrosymmetric linear monomers. The triangle range follows from angular-momentum coupling. The upper bounds and the extra high-order rule are numerical truncations, not fundamental symmetry statements.

This contract separates three choices that must not be conflated:

- **Representation:** the canonical set of angular functions permitted in the comparison.
- **Fitting method:** which of those functions a reduced method retains at a particular radius.
- **Serialization:** the index order and subset required by a downstream file format.

All coefficient exchange should use explicit `(l1, l2, L)` keys. A reduced method should export omitted members of the canonical universe explicitly as zero when a complete vector is required.

## Index conventions under discussion

The current notebook uses `(l1, l2, L)`. Other code or file formats may permute or pad these indices. Any mapping to another convention must be written explicitly and tested before coefficient values are compared.

This document records the current implementation and remains subject to confirmation with the research group.

