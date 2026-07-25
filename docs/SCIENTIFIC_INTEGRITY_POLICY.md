# Scientific Integrity Policy

WW-PGD scientific outputs must contain only real WeightWatcher measurements or documented derived values. If WeightWatcher fails, omits a field, or reports an invalid fit, outputs must preserve the missing/error status and use NaN for dependent derived fields.

## Legacy immediate-alpha issue

Earlier strength-scan code generated `wwpgd_projection_spectral.csv` with fabricated immediate alpha values and placeholder fit-quality fields. Those files are invalid for scientific use unless they contain `immediate_spectral_source=weightwatcher_measured` and measured-provenance fields. Old data are not deleted automatically; audit old scans with `wwgpt audit-strength-scan --scan-root PATH` and rerun invalid arms.

## Spectral target ownership

`target_alpha` is the only researcher-facing spectral target and must be finite and greater than one. The external WW-PGD dependency's private rank exponent is derived only inside the adapter as `1 / (target_alpha - 1)`; it is not a configurable, scannable, or tunable research parameter. Manifests record the target, derived value, formula, and external parameter name without exposing a second target control.
