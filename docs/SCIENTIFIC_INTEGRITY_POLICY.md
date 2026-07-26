# Scientific Integrity Policy

WW-PGD scientific outputs must contain only real WeightWatcher measurements or documented derived values. If WeightWatcher fails, omits a field, or reports an invalid fit, outputs must preserve the missing/error status and use NaN for dependent derived fields.

## Legacy immediate-alpha issue

Earlier strength-scan code generated `wwpgd_projection_spectral.csv` with fabricated immediate alpha values and placeholder fit-quality fields. Those files are invalid for scientific use. The retired scan has no dedicated analysis or audit command; rerun a currently supported experiment rather than interpreting nominal strength as a projector parameter.

## Spectral target ownership

`target_alpha` is the only researcher-facing spectral target and must be finite and greater than one. The external WW-PGD dependency's private rank exponent is derived only inside the adapter as `1 / (target_alpha - 1)`; it is not a configurable, scannable, or tunable research parameter. Manifests record the target, derived value, formula, and external parameter name without exposing a second target control.
