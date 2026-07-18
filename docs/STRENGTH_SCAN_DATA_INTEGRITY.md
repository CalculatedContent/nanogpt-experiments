# Strength Scan Data Integrity

Older WW-PGD strength-scan versions created unverified synthetic `wwpgd_projection_spectral.csv` files for immediate alpha diagnostics. Those files can contain plausible-looking alpha, KS-D, and evaluation-count values that were not produced by WeightWatcher. They must not be used for scientific immediate-alpha analysis.

New immediate spectral files must include:

- `immediate_spectral_source = "weightwatcher_measured"`
- `measurement_valid_for_science = true`
- WeightWatcher version and configuration provenance

Files lacking these fields are classified as `legacy_fabricated_or_unverified` and are excluded from alpha analysis. User files are not deleted automatically.

`metrics.csv` loss/accuracy results and periodic `spectral.csv` WeightWatcher outputs require separate provenance checks. A scan can have usable loss and accuracy while its immediate-alpha file is invalid.

Run the integrity audit:

```bash
wwgpt audit-strength-scan --scan-root PATH
```

The audit writes:

- `analysis/strength_scan_integrity_audit.csv`
- `analysis/strength_scan_integrity_summary.json`

Fixture or toy-budget scans are invalid for scientific claims. Test fixtures may only be used with explicit test-only controls such as `WWGPT_ALLOW_TEST_FIXTURE=1` in notebooks.
