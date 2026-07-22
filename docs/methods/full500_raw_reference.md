# Method: `full500_raw_reference`

- **Input:** every available orientation at every supplied radius.
- **Target:** raw ab initio potential in inverse centimetres.
- **Representation:** `co2-h2-candidate-158-v1`.
- **Solver:** `numpy.linalg.lstsq`.
- **Default `rcond`:** `None`; no project-specific truncated-SVD cutoff.
- **Role:** reference within the available angular grid.

This method is the denominator for reduced-point and reduced-basis experiments.
It is not proof that the angular grid, radial grid, basis omission list, or
long-range electronic energies are production-correct.
