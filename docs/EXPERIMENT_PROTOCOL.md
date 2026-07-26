# Document

This document describes the nanoGPT AdamW vs WW-PGD framework.

- Budget rationale: use exact instantiated parameter counts with `transformer_body` as the default selected convention; total and trainable counts are also recorded. A 20-token multiplier is an experimental budget convention, not a fitted scaling-law result or claimed optimum.
- Dataset split methodology: stream documents, normalize text, SHA-256 hash normalized content, assign duplicates to the same split, keep validation documents out of tokenizer training unless explicitly configured, and refuse insufficient unique training tokens.
- Paired-run design: one initialization is saved and reused for AdamW and AdamW+WW-PGD, with identical data order, model configuration, tokenizer hash, corpus hash, and realized tokens.
- Uncertainty methodology: compute confidence intervals across independent seeds, not across layers or time points. Paired comparisons are WW-PGD minus AdamW.
- Statistical limitation: five seeds provide useful diagnostics but limited power.
- WW-PGD definition: after AdamW, selected matrices receive the local projection implemented in this repository toward target alpha 2.0. It is not a standard WeightWatcher operation.
- WeightWatcher interpretation: raw per-layer records are retained; layer variation is not treated as independent experimental replication.
- Valid runs: complete, non-wrapped, non-overlapping train/validation, matching paired configs, matching initialization, matching tokenizer/corpus hashes, matching token budgets, and sufficient corpus coverage.
- Troubleshooting: ensure `wwgpt` is installed with `python -m pip install -e .`; use CPU for portability, MPS defaults to fp32, CUDA uses bf16 where supported, TPU/XLA is recorded when available, WeightWatcher failures should be fixed rather than silently replaced for scientific runs, disk pressure requires choosing a larger storage root, interrupted preparation/training can be resumed from manifests and checkpoints, missing seeds are reported by analysis, and invalid scaling runs are excluded.
