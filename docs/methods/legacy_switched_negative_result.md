# Negative result: legacy switched target

The notebook experimented with bounding high repulsive energies between 1000
and 5000 inverse centimetres. The mapping is continuous but its derivative jumps
at the lower threshold, so it is not `C1` there.

The transform is retained as `apply_legacy_switch` only to reproduce and explain
the experiment. It is not a neutral preprocessing step and is not used by
`full500_raw_reference`. Any future run using it must have a separate method ID
and record both thresholds.
