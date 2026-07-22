# Evaluation contract: `co2-h2-hybrid-1cm-1pct-wall-v1`

Every fit and diagnostic records the scoring contract in its manifest. The
current primary rule is

\[
\tau(V)=\max(1\ \mathrm{cm}^{-1},\;0.01|V|).
\]

Pointwise normalized error is `abs(V_pred - V_true) / tau(V_true)`. Energy
bands are defined from the raw truth:

| Band | Interval (cm^-1) |
|---|---:|
| Attractive | `V < 0` |
| Low repulsive | `0 <= V < 1000` |
| Lower wall | `1000 <= V < 3000` |
| Upper wall | `3000 <= V < 5000` |
| Guardrail | `V >= 5000` |

The guardrail asks whether a fitted surface creates a false low-energy path
through a known high wall. A prediction below 3000 where truth is at least 5000
is the primary wall failure; 1000 is also recorded as the dangerous threshold.

These are working physical choices, not calibrated electronic-structure error
bars. The design study holds fitted surfaces fixed while testing 36 combinations
of absolute floor, relative fraction, physical ceiling, and warning threshold.
Changing the primary rule requires a new evaluation ID; it must not silently
rewrite an old manifest.
