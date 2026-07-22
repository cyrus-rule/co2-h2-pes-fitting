# Angular-basis contract: `co2-h2-candidate-158-v1`

## Model and units

\[
V(R,\Omega)=\sum_q c_q(R)A_q(\Omega),
\qquad \Omega=(\theta_1,\theta_2,\phi),
\qquad q=(\ell_1,\ell_2,L).
\]

- `R`: center-of-mass separation in Bohr.
- `theta1`, `theta2`, `phi`: input in degrees and converted to radians before
  basis evaluation.
- `V` and `c_q`: inverse centimetres.
- `A_q`: dimensionless coupled angular function.

The associated Legendre functions use the Appendix C7 normalization implemented
in `basis.normalized_associated_legendre`. SciPy's `lpmv` already includes the
Condon–Shortley phase used by this implementation. For the isotropic term,

\[
A_{000}=\frac{1}{4\sqrt{\pi}},
\]

so both `c000` and the physical contribution `c000*A000` are exported.

## Candidate 158-term universe

The current code applies:

1. \(\ell_1\in\{0,2,\ldots,24\}\).
2. \(\ell_2\in\{0,2,4,6\}\).
3. If \(\ell_1\ge22\), then \(\ell_2=0\).
4. \(|\ell_1-\ell_2|\le L\le\ell_1+\ell_2\).
5. \(\ell_1+\ell_2+L\) is even.

This produces 158 unique tuples and a full-rank `500 x 158` matrix on the July
2026 grid. The even monomer indices encode axis-reversal symmetry; the triangle
and parity conditions come from angular coupling. The bounds and high-order
omission are numerical truncations.

The published methodology states that some permitted terms were omitted but
does not list the entire omission set in prose. Therefore the repository calls
this basis *candidate*, not canonical. The ordered golden table is
`data/reference/co2_h2_candidate_158_v1.csv`. External validation must compare
named tuples to a trusted production fixture and then create a new basis ID; it
must not silently relabel this version.

## Representation is not method

- **Representation:** the agreed universe of physically permitted functions.
- **Method:** the fitting or selection algorithm operating within that universe.
- **Sampling:** the ab initio orientations supplied to the method.
- **Serialization:** the tuple order and maximum set required downstream.

A reduced method selects from the agreed universe and exports omitted required
terms as exact zero. It does not generate a new first-`N` basis by ordering.

## Index mappings

| Context | Stored tuple |
|---|---|
| Repository interchange | `(l1, l2, L)` |
| Rohan's July 2026 notebook | `(L, l1, l2)` |
| Observed YUMI template | `(l1, 0, l2, L)` |

All comparisons must map to `(l1, l2, L)` before joining values. Column position
alone has no scientific meaning.

## Promotion checklist

Before declaring a production basis:

1. Obtain the authoritative term list or unmasked production file.
2. Compare exact tuple membership and order.
3. Confirm representative `A_q(Omega)` values against an independent program.
4. Record phase, monomer, index-order, and normalization conventions.
5. Add the source fixture and its hash without replacing this version.
