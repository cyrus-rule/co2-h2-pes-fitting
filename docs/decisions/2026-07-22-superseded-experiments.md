# Superseded exploratory interventions

The archived diagnostic and methodology notebooks tested several interventions
that are not promoted as production methods.

| Intervention | Purpose | Why not promoted |
|---|---|---|
| 46-term truncation | Reduce variance in sparse fits | Changed representation and lost important anisotropy; later full-basis D-optimal designs addressed sampling directly |
| Log-weighted fitting | Emphasize low energies | Did not supply a consistent raw-potential improvement and changed the regression objective |
| Ridge/Tikhonov regularization | Stabilize sparse coefficients | Exploratory penalty was not physically calibrated; better-conditioned geometry-only designs retained an unpenalized full-rank solution |
| 165-point sparse random fits | Minimize calculations | Unstable angular forces and incomplete numerical rank appeared in sparse subsets |
| Uniform-random saturation | Establish a point budget | Performance improved around 200--250 points but remained inefficient and variable at the hardest radii |
| Leverage-weighted random sampling | Target informative rows | Better than uniform in some pilots but less stable and less reproducible than the QR-seeded greedy ordering |
| Legacy switched target | Protect against extreme wall labels | Deliberate target distortion dominated raw-reference error; retained separately as a documented negative result |

These results are method history, not universal impossibility claims. Exact
historical reproduction remains available in the archived notebooks; no CLI
surface is added unless a future scientific question requires one.
