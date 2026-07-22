# Diagnostic: `sobol-model-disagreement-v1`

This diagnostic evaluates full-500 and reduced-180 coefficient surfaces on two
deterministic 4096-angle scrambled Sobol grids:

- coordinate-uniform in `(theta1, theta2, phi)`;
- solid-angle in the two polar coordinates and uniform in `phi`.

At 4.4, 4.8, 5.0, 5.8, and 6.6 Bohr it records absolute and hybrid-normalized
energy disagreement, centered finite-difference angular-gradient disagreement,
and wall-threshold crossings. A multistart bounded search probes local minima at
4.4, 4.8, and 5.0 Bohr.

The v3/real-data result has a maximum 180-versus-500 disagreement of roughly
108 cm^-1 at 4.4 Bohr, no reference-above-5000/candidate-below-3000 opening,
one solid-angle 1000-threshold crossing, and corresponding fitted minima within
about 1.6 cm^-1.

The complete curated tables and their hashes are stored in
`data/reference/offgrid_diagnostics_v1/`.

There are no new electronic-structure labels at these angles. The outputs are
model--model stress evidence used to select independent calculations, not
validation errors.
