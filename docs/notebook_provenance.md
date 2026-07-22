# Notebook provenance

The notebooks are preserved as dated research records. They are not the current
entry point and their saved outputs are not automatically promoted as results.

- `notebooks/archive/short_range_fit_diagnostics.ipynb`: original diagnostic
  exploration.
- `notebooks/co2_h2_short_range_methodology_v2_meeting.ipynb`: comprehensive
  July meeting/audit artifact. Its saved Section 18 terminates with a
  `RuntimeError`, so downstream saved cells must not be read as a clean
  chronological execution.
- `notebooks/archive/co2_h2_short_range_presentation_v3.ipynb`: streamlined
  July 16 presentation companion. It repairs the v2 Section 18 request
  construction and freezes the 48-orientation independent-validation request.
  The executable generator, exact coordinate files, and decision rule have now
  been promoted to package code and `data/reference/validation_request_v3/`.
  Its preserved-file SHA-256 is
  `3e4c09b60d242f3cafd99b668b133f2d26e2b7f157279945398d8026e18bb103`.

Reusable claims move out of a notebook only when their experimental design,
dependencies, method ID, input hash, and tests are represented in the package or
a standardized run directory.

V3 remains provenance rather than the main interface. Its saved optimizer
returned the `(90, 89.999999, 179.999998)` representative of a symmetric local
minimum. A clean regeneration returned a basis-equivalent `phi=0`
representative in one environment (maximum basis-row difference
`2.81e-14`). The package deliberately canonicalizes that tie to the coordinate
already frozen by v3, rather than allowing an optimizer tie to revise a
preregistered request.
