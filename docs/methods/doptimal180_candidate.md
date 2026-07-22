# Method: `doptimal180_candidate`

- **Candidate pool:** the complete shared 500-orientation angular grid.
- **Selection input:** angular design matrix only; no potential energies.
- **Target:** raw ab initio potential in inverse centimetres.
- **Representation:** `co2-h2-candidate-158-v1`.
- **Solver:** `numpy.linalg.lstsq` on the 180 selected rows.
- **Default `rcond`:** `None`; no project-specific truncated-SVD cutoff.
- **Status:** prospective reduced-sampling candidate, not a production PES.

The complete candidate design is first whitened so its information matrix is
the identity. Column-pivoted QR of the whitened transpose selects 158 rows,
one for each fitted coefficient, as a full-rank seed. Each of the next 22 rows
maximizes

\[
x_i^\mathsf{T}(X_S^\mathsf{T}X_S)^{-1}x_i.
\]

By the matrix determinant lemma, this is the candidate with the greatest
immediate information-determinant gain. It is therefore a deterministic,
QR-seeded greedy D-optimal-*style* design, not a claim of a globally exact
D-optimal solution.

## Required interpretation

The method must be scored on the complete 500-point evaluation grid and, in
particular, on the 320 unselected orientations. Passing this finite-grid test
does not establish that 180 points are sufficient for a new ab initio grid.
That claim requires the independent off-grid calculations proposed after the
July 9 meeting, especially at the shortest radii.

The selected orientation CSV is ordered by design position so another
implementation can reproduce and compare the exact design rather than merely
the unordered set.

The numerical reason for using 180 rather than the fragile first-pass count of
161 is recorded in [`../design_study.md`](../design_study.md) and the frozen
`data/reference/doptimal_design_study_v1/` artifact. The count remains a
planning decision inside a supported range, not an optimized scalar truth.
