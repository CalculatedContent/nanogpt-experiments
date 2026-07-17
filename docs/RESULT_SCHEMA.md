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

## Scientific schema version 2 analysis contract

Schema-v2 scientific optimizer arms are `adamw` and `adamw_wwpgd_reference`. The legacy raw optimizer name `adamw_wwpgd` is retained only for audit/backward compatibility and is not loaded by default for scientific comparisons. Analysis records preserve `optimizer_raw` and also derive `optimizer_family` (`adamw` or `wwpgd`) and publication labels (`AdamW`, `AdamW + WW-PGD`).

Every scientific run manifest should include `scientific_schema_version >= 2`, `valid_for_science`, `seed`, pair identifiers, `realized_tokens`, initialization/data-or-corpus/tokenizer hashes, and fixed `validation_probe_hash` and `training_probe_hash` values. Canonical duplicate-pair selection is by seed: a valid pair must contain completed AdamW and `adamw_wwpgd_reference` runs with `manifest.json`, `metrics.csv`, and `run_complete.json`; matched hashes and token budgets; and schema version at least 2. If multiple valid complete pairs exist for a seed, the newest complete pair is selected deterministically and older duplicate, incomplete, failed, or legacy pairs are recorded in an audit table with an exclusion reason.

Current metrics files may use `tokens_processed`, `val_loss`, `elapsed_time`, and `projection_overhead`. The shared analysis layer preserves those original columns and adds aliases `tokens_seen`, `validation_loss`, `elapsed_seconds`, and `projection_seconds`.

Current WW-PGD projection files use `projection_event`, `scheduled_token_fraction`, `actual_step`, `actual_tokens_seen`, `layer_name`, `hardness`, `projection_runtime`, `changed`, `skip_reason`, `relative_frobenius_change`, `relative_frobenius_weight_change`, `xmin`, `detX_num`, `tail_size`, `TraceLog_before`, `TraceLog_after`, `wwpgd_implementation`, and `wwpgd_commit`. Analysis preserves these fields and adds aliases `step`, `tokens_seen`, `trace_log_before`, and `trace_log_after`.

Current WeightWatcher spectral files may identify layers with `longname` or `name` rather than `layer_name`. Scientific WeightWatcher analysis requires `spectral_estimator == "weightwatcher"` and preserves available fields including `alpha`, `weighted_alpha`, `xmin`, `xmax`, `D`/KS statistic, `num_evals`, `detX_num`, `detX_val`, `spectral_norm`, `stable_rank`, `mp_softrank`, `num_spikes`, `status`, `warning`, and `weightwatcher_version`. Projected transformer matrices are analyzed separately from embeddings, positional embeddings, `lm_head`, and other unprojected matrices.
