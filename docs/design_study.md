# Why the candidate uses 180 orientations

`co2-h2-pes design-study` reproduces the evidence in
`data/reference/doptimal_design_study_v1/`. Selection uses basis rows only;
energies enter only after the ordered subsets are frozen for retrospective
scoring on the known 500-angle grid.

Key results are:

| Count | Baseline safety factor | Single-substitution pass rate |
|---:|---:|---:|
| 161 | 2.126 | 65.84% |
| 175 | 2.747 | 96.57% |
| 180 | 3.028 | 97.22% |
| 182 | 3.265 | 97.25% |

All counts 161--200 pass the baseline strict screen, but 161 is a fragile
boundary. Label-free reoptimization recovered all 16 observed failed simple
substitutions in the 175--182 designs. Across 36 evaluation specifications,
the minimum safety factors were approximately 1.06, 1.37, 1.51, 1.63, and 1.05
for 161, 175, 180, 182, and 200 points respectively.

Therefore 180 is a round, robust prospective pilot inside the supported
175--182 range. It is not a global D-optimum, a statistically estimated true
minimum, or an independently validated total calculation budget.
