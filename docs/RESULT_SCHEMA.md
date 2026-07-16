# Document

This document describes the nanoGPT AdamW vs WW-PGD framework.

- Scaling rationale: use exact instantiated parameter counts; total parameters are default, non-embedding parameters are also recorded. The 20-token-per-parameter rule is an experimental extrapolation for these tiny GPTs.
- Dataset split methodology: stream documents, normalize text, SHA-256 hash normalized content, assign duplicates to the same split, keep validation documents out of tokenizer training unless explicitly configured, and refuse insufficient unique training tokens.
- Paired-run design: one initialization is saved and reused for AdamW and AdamW+WW-PGD, with identical data order, model configuration, tokenizer hash, corpus hash, and realized tokens.
- Uncertainty methodology: compute confidence intervals across independent seeds, not across layers or time points. Paired comparisons are WW-PGD minus AdamW.
- Statistical limitation: five seeds provide useful diagnostics but limited power.
- WW-PGD definition: after AdamW, selected matrices receive the local projection implemented in this repository toward target alpha 2.0. It is not a standard WeightWatcher operation.
- WeightWatcher interpretation: raw per-layer records are retained; layer variation is not treated as independent experimental replication.
- Valid runs: complete, non-wrapped, non-overlapping train/validation, matching paired configs, matching initialization, matching tokenizer/corpus hashes, matching token budgets, and sufficient corpus coverage.
- Troubleshooting: ensure `wwgpt` is installed with `python -m pip install -e .`; use CPU for portability, MPS defaults to fp32, CUDA uses bf16 where supported, TPU/XLA is recorded when available, WeightWatcher failures should be fixed rather than silently replaced for scientific runs, disk pressure requires choosing a larger storage root, interrupted preparation/training can be resumed from manifests and checkpoints, missing seeds are reported by analysis, and invalid scaling runs are excluded.

## Scientific schema version 2

New valid runs include these manifest fields: `spectral_estimator`, `spectral_estimator_version`, `wwpgd_implementation`, `wwpgd_commit`, `projection_schedule`, `validation_probe_hash`, `training_probe_hash`, and `scientific_schema_version`.

Valid spectral rows must have `spectral_estimator == "weightwatcher"`. Rows labeled `fallback_non_scientific`, or legacy rows without `spectral_estimator`, are invalid for WeightWatcher alpha analysis.
