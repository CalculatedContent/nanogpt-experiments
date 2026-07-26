# nanoGPT WW-PGD Experiments

This repository runs append-only, paired nanoGPT experiments comparing base optimizers with and without the repository's WW-PGD extension. It provides experiment infrastructure and descriptive analysis; it does **not** ship a scaling-law fit, acceleration conclusion, statistical-significance test, or alpha-generalization result.

## Quick start

```bash
./scripts/setup_environment.sh
./scripts/download_data.sh 0 /path/to/data 20
./scripts/run_five_seeds.sh 0 /path/to/data /path/to/results 20
./scripts/analyze_five_seeds.sh /path/to/results
```

A smoke test checks infrastructure only and is invalid for scientific conclusions:

```bash
./scripts/run_smoke_test.sh /tmp
```

## Pilot commands

Prepare the pinned FineWeb-Edu data before launching these pilots (the quick-start
`download_data.sh` command above does this). Each command runs one paired AdamW
baseline/WW-PGD seed with the level-specific, `target_alpha: 2.0` cached-endpoint
configuration. Replace the example data and results paths as needed.

**Level 0 pilot:**

```bash
wwgpt run-multiseed --level 0 --config configs/level0_adaptive_alpha.yaml --analysis-plan configs/analysis_plan.yaml --data-root /path/to/data --results-root /path/to/results --token-multiplier 20 --seeds 1337 --extensions none,wwpgd
```

**Level 1 pilot:**

```bash
wwgpt run-multiseed --level 1 --config configs/level1_adaptive_alpha.yaml --data-root /path/to/data --results-root /path/to/results --token-multiplier 20 --seeds 1337 --extensions none,wwpgd
```

**Level 2 pilot:**

```bash
wwgpt run-multiseed --level 2 --config configs/level2_adaptive_alpha.yaml --data-root /path/to/data --results-root /path/to/results --token-multiplier 20 --seeds 1337 --extensions none,wwpgd
```

Run the preregistered paired acceleration and alpha-distance analyses only after
both arms have completed:

```bash
wwgpt analyze-results /path/to/results --profile scaling --analysis-plan configs/analysis_plan.yaml
```

Audit the same isolated experiment root before interpreting any output:

```bash
wwgpt audit-experiment --experiment-root /path/to/results
```

These commands make the repository capable of testing whether WW-PGD accelerates
generalization. They do **not** establish that it does; that claim requires analysis
of actual eligible paired results.

## Public experiment interface

`wwgpt prepare-data`, `wwgpt run-multiseed`, and `wwgpt run-canonical-trials` accept `--dry-run` and print the resolved configuration, trial and arm counts, seeds, token budget, estimated optimizer steps, and output location. `run-strength-scan` is retired and is not a CLI command: its nominal strength was not a scientifically defined projector parameter. It has not been replaced by a `q` scan or target-alpha scan.

The only researcher-facing spectral target is `wwpgd.target_alpha` (default `2.0`). It must be finite and greater than one. At the external adapter boundary the dependency's required rank exponent is derived as `1 / (target_alpha - 1)`; it is not independently configurable or scannable. Functional controller-dose controls, including apply/measurement intervals, start step, maximum per-step gain, and relative-Frobenius trust-region caps, can be set on ordinary explicit ablation runs. There is currently no public dose-scan runner.

Profiles resolve to pinned YAML configurations:

- `scaling` → `configs/default.yaml`;
- `reproduction_tiny` → `configs/reproduction_tiny.yaml`;
- `reproduction_fineweb` → `configs/reproduction_fineweb.yaml`.

Supplying both a profile and a conflicting config is an error. CLI arguments, not undocumented environment overrides, are the supported general interface.

## Actual default configuration

The source of truth is `configs/default.yaml`:

- **Head setting:** token embeddings and the bias-free LM head are tied (`model.tie_weights: true`). Transformer attention uses separate bias-free key, query, value, and projection matrices.
- **Weight decay and clipping:** weight decay is `0.1`; gradient clipping is enabled at `1.0`.
- **Model ladder:** level 0 is `(layers=1, heads=1, width=64, block=256)`. Levels 1–4 are `(2,2,128)`, `(4,3,192)`, `(6,4,256)`, and `(8,5,320)`, preserving 64-dimensional heads.
- **Data mode:** the scaling default is `fineweb_custom_bpe_scaling` over pinned `HuggingFaceFW/fineweb-edu` `sample-10BT`. Preparation deterministically splits normalized documents, keeps duplicates in one split, trains the custom BPE only on training documents, and records dataset/tokenizer identities. Local or synthetic text is for tests and smoke runs only.
- **Token convention:** budgets use `parameter_count_convention: transformer_body`, not total or trainable parameter count. The selected count and realized tokens are recorded in each manifest. A multiplier such as 20 is an experiment budget convention, not a claimed optimum.
- **Measurement cadence:** evaluation, alpha measurement, and trap diagnostics default to every 10 optimizer steps; checkpoints default to every 50 steps. Composite spectral analysis is disabled by default. WW-PGD event cadence is a separate optimizer-step schedule (default every step), and cached-endpoint mode separates expensive measurements from fast endpoint relaxation.
- **Training:** batch size 16, gradient accumulation 1, dropout 0, flat layer learning rates, and warmup-cosine scheduling are the defaults.

Resolved manifests, rather than this summary, are authoritative for any individual run.

## Arms and intervention

Schema v3 separates a base optimizer from an extension. Canonical trials contain AdamW, Muon, and StableAdamW, each paired with its WW-PGD extension arm. Paired arms share initialization, sampling identity, tokenizer/corpus hashes, probes, and token budgets. WW-PGD runs after a successful base-optimizer step and acts only on eligible transformer matrices; it does not target embeddings, the tied LM head, LayerNorm values, or biases.

Adaptive and cached-endpoint modes are explicit ablations, not evidence of improved loss or generalization. Controller decisions are recorded separately from actual projection or relaxation moves, including requested/applied hardness, endpoint state, and trust-region clipping.

## Available analysis

`wwgpt analyze-results RESULTS_ROOT` discovers completed runs, inventories them, selects auditable paired arms, computes per-seed terminal differences and descriptive uncertainty summaries, and analyzes measured alpha trajectories. It writes a `scaling_fit_results.csv` marker with status `not_fit`; no scaling fit or hypothesis test is implemented. Acceleration-analysis outputs are produced only when an explicit analysis plan is supplied and its prerequisites are met; their existence should not be described as a result without inspecting the generated artifacts.

Retired nominal-strength artifacts are not accepted by a dedicated public analysis or audit interface. They must not be interpreted as controller-dose, target-alpha, or rank-exponent scans.

## Resume and append-only behavior

New runs use unique timestamped directories and do not overwrite prior results. With `--resume`, the runner resumes a compatible incomplete run from its latest atomically written checkpoint; it does not silently reuse a completed run as new evidence. Resume validation checks configuration, data/tokenizer identity, initialization, schema, and code provenance. Incompatible continuation is refused unless the specifically supported code-version override is supplied, in which case an audit artifact is written and publication eligibility follows the recorded override policy. Controller and cached-endpoint state are checkpointed so continuation preserves scheduling decisions.

## Audit behavior

`wwgpt audit-experiment --experiment-root PATH` checks artifact completeness, arm identity, pair invariants, measurement schedules, finite metrics, schema/provenance fields, and recorded overrides. It writes machine-readable and Markdown reports under the analysis directory. Audit reports eligibility and exclusion reasons; it does not infer efficacy, significance, acceleration, scaling behavior, or generalization. `generate-reproducibility-report` summarizes recorded provenance but likewise makes no scientific outcome claim.

See `docs/SCHEMA_V3.md`, `docs/EXPERIMENT_RESUME_PROTOCOL.md`, `docs/RESULT_SCHEMA.md`, and `docs/SCIENTIFIC_INTEGRITY_POLICY.md` for schema and integrity details.
