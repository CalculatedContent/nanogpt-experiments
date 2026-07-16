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

## Methodology repair: WeightWatcher and WW-PGD validity

Scientific spectral records now require the real WeightWatcher analyzer: `WeightWatcher(model=model).analyze(detX=True, randomize=False, plot=False)`. The legacy local SVD rank-regression estimator is quarantined for smoke tests only and is labeled `fallback_non_scientific`.

The corrected WW-PGD arm is named `adamw_wwpgd_reference` and is pinned to `CalculatedContent/WW_PGD` commit `bf970cb6b73e977f8374114c442ae5b0589eccaa`. Projection is sparse and token-progress based with default thresholds `0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.82, 0.92`. Event index maps to hardness by the reference warmup/ramp schedule: events before warmup have zero hardness; ramp events increase linearly as `(event - warmup + 1) / ramp_events`; subsequent events use full hardness.

Eligible projected layers are only transformer matrices named `blocks.*.attn.c_attn`, `blocks.*.attn.c_proj`, `blocks.*.mlp.0`, and `blocks.*.mlp.2`. Token embeddings, position embeddings, tied output weights, LayerNorm parameters, and biases are excluded.

Evaluation uses fixed held-out validation and fixed training probes, identified by hashes in metrics and manifests. Validation readers never concatenate training tokens.
