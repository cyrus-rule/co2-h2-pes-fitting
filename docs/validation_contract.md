# Frozen independent-ab-initio validation request

The authoritative request is `co2-h2-independent-request-v3`, stored under
`data/reference/validation_request_v3/` with hashes and a generation manifest.

It contains:

- 12 coordinate-uniform Sobol orientations;
- 12 solid-angle Sobol orientations;
- 8 energy-disagreement stress orientations;
- 8 angular-gradient stress orientations;
- 4 wall-boundary orientations;
- 4 local-extremum orientations.

The 48 orientations are requested at 4.4, 4.8, and 5.0 Bohr: 144 primary new
energies. A 12-orientation subset at 4.7 and 4.9 Bohr supplies an optional
24-energy radial challenge. Coordinates were frozen before their requested
energies were known.

## Decision rule

Both `full500_raw_reference` and `doptimal180_candidate` are scored against the
new truth using the primary evaluation contract. Any pointwise failure below
the 5000 ceiling or confirmed false wall opening requires investigation. If
both fits fail at the same geometry, diagnose basis or radial representation
before assigning the problem to angular compression. If a fit is augmented or
reoptimized after failure, the changed method needs a fresh secondary holdout;
the failed points cannot be relabelled as independent validation.

The scoring command accepts a final verdict only for all 144 primary rows. It
checks the three radii for every validation ID and verifies the complete frozen
geometry fingerprint, preventing a partial or altered subset from being
reported as the preregistered validation result.

The optional 4.7/4.9 labels cannot be scored through the angular-only run
artifacts until a production radial interpolation is specified. They are held
separate for that reason.
