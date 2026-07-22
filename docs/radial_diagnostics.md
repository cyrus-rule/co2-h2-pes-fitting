# Radial diagnostic status

`co2-h2-pes radial-diagnostics` compares full-500 and reduced-180 coefficient
curves and withholds each interior 4.4--6.6 Bohr radial node from a diagnostic
cubic spline.

The real-data result in `data/reference/radial_diagnostics_v1/` finds a global
relative coefficient difference of about `2.84e-4`, relative first-derivative
difference `6.60e-4`, and relative second-derivative difference `1.41e-3`.
The largest finite-grid surface deviation remains at 4.4 Bohr: RMSE about
13.75 cm^-1 and maximum about 91.72 cm^-1.

Both angular designs fail the same 4.8 Bohr radial holdout under the strict
pointwise rule:

| Angular fit | Worst normalized error | Minimum band within tolerance |
|---|---:|---:|
| Full 500 | 1.287 | 96.30% |
| D-optimal 180 | 1.423 | 96.30% |

The shared failure points toward radial resolution/interpolation rather than
angular compression. The diagnostic spline is not the planned production
cubic-spline/exponential/inverse-power join.
