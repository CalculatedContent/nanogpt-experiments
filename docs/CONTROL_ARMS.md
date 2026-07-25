# Optional WW-PGD control arms

The publication's canonical command remains the six arms defined by
`CANONICAL_TRIAL_ARMS`. Controls are opt-in through `run-multiseed --extensions`;
they are deliberately not added to `run-canonical-trials`.

For a base optimizer, the required control comparison is:

1. `none` (the unmodified base optimizer),
2. `measurement_only`,
3. `norm_matched_sham`, and
4. `wwpgd` (the real target-alpha intervention).

For example:

```bash
wwgpt run-multiseed \
  --level 0 --data-root data --results-root results --token-multiplier 20 \
  --optimizer adamw \
  --extensions none,measurement_only,norm_matched_sham,wwpgd
```

All four arms use the single `wwpgd.target_alpha` value in the selected config.
There is no control-specific target alpha or external rank-exponent option, and
`q` remains rejected as a configuration key.

`measurement_only` runs the non-mutating WeightWatcher event measurement at the
WW-PGD cadence, but never creates a stock candidate or changes a weight.

`norm_matched_sham` first creates the same stock WW-PGD candidate as the real
arm. For each selected layer it deterministically constructs an arm-seeded
random displacement, removes the component parallel to the real candidate,
and norm-matches it after controller hardness and before the shared trust-region
cap. Projection logs include `real_candidate_displacement_cosine`,
`displacement_kind`, and `sham_seed`.

The optional `delayed_onset` arm uses the same target-directed intervention but
requires `wwpgd.delayed_onset_step` to be declared before execution. No
validation or test metric is consulted to choose that step.
