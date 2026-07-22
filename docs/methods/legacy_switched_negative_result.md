# Negative result: legacy switched target

The notebook experimented with bounding high repulsive energies between 1000
and 5000 inverse centimetres. The mapping is continuous but its derivative jumps
at the lower threshold, so it is not `C1` there.

The transform is retained as `apply_legacy_switch` only to reproduce and explain
the experiment. It is not a neutral preprocessing step and is not used by
`full500_raw_reference`. Any future run using it must have a separate method ID
and record both thresholds.

## Evidence retained from the notebooks

The comprehensive methodology notebook compared raw and switched targets at
165, 175, 250, and 400 training orientations under paired uniform-random and
fold-conditioned D-optimal sampling at every short-range radius. Predictions
were always scored against the raw potential, while regression error relative
to each method's own target and deliberate target distortion were recorded
separately.

The switch did not improve the global raw-reference result in any tested count
or sampling regime. Its 1000--5000 inverse-centimetre target distortion
dominated the apparent numerical benefit, and regression spillover also harmed
some points below 1000 where the transform itself is the identity. The
`R=6.6` identity-dominated control did not supply evidence for promoting the
transform.

This is evidence against this particular transform for the present raw-fidelity
objective. It is not a proof that every identity-preserving wall treatment is
useless, nor a downstream pressure-broadening comparison.
