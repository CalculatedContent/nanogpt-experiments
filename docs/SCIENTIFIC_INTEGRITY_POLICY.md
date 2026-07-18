# Scientific Integrity Policy

WW-PGD scientific outputs must contain only real WeightWatcher measurements or documented derived values. If WeightWatcher fails, omits a field, or reports an invalid fit, outputs must preserve the missing/error status and use NaN for dependent derived fields.

## Legacy immediate-alpha issue

Earlier strength-scan code generated `wwpgd_projection_spectral.csv` with fabricated immediate alpha values and placeholder fit-quality fields. Those files are invalid for scientific use unless they contain `immediate_spectral_source=weightwatcher_measured` and measured-provenance fields. Old data are not deleted automatically; audit old scans with `wwgpt audit-strength-scan --scan-root PATH` and rerun invalid arms.
