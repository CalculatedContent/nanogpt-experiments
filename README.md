# nanoGPT WW-PGD Experiments

This repository runs append-only, paired nanoGPT experiments comparing base optimizers with and without the repository's WW-PGD extension. It provides experiment infrastructure and descriptive analysis; it does **not** ship a scaling-law fit, acceleration conclusion, statistical-significance test, or alpha-generalization result.

## Quick start on a local MacBook

Run the repository from its checked-out `main` branch and keep the prepared corpus and results under `/tmp`:

```bash
./scripts/setup_environment.sh
./scripts/run_one_pair.sh \
  0 \
  /tmp/nanogpt-experiments/data \
  /tmp/nanogpt-experiments/results-level0 \
  20 \
  mps \
  1337
```

`run_one_pair.sh` prepares or reuses the exact level-specific data identity, runs the paired AdamW and AdamW+WWPGD arms, checks run health, analyzes the result, audits the experiment, and writes the reproducibility report.

Run one bounded paired pilot at each of Levels 0, 1, and 2 with:

```bash
WWGPT_SEEDS=1337 WWGPT_RUN_NOTEBOOKS=0 \
  ./scripts/run_level0_2_experiment.sh \
  /tmp/nanogpt-experiments/data \
  /tmp/nanogpt-experiments/results-level0-2 \
  20 \
  mps
```

Set `WWGPT_SEEDS=1337,2027,4099` only after the one-seed pilots complete and pass health and audit checks.

### Fixed-corpus Level 0 overtraining pilot

The nominal multiplier-20 corpus can be reused for an explicitly labeled
overtraining protocol. This is opt-in: without `allow_overtraining`,
`max_steps` remains a cap and cannot extend the token-budget horizon. The pilot
configuration uses a fixed validation probe, revisits the immutable training
corpus through random-window sampling, excludes the run from scaling-law fits,
and records both nominal and actual token counts.

```bash
./scripts/run_level0_overtraining_pilot.sh \
  /tmp/nanogpt-experiments/data \
  /tmp/nanogpt-experiments/results-level0-overtraining \
  mps \
  1337,2027,4099
```

The supplied pilot extends Level 0 from 242 to 1,000 optimizer steps and uses a
late, lower-dose, below-target-only WW-PGD intervention. It is a development
protocol, not evidence that WW-PGD improves generalization.

A smoke test checks infrastructure only and is invalid for scientific conclusions:

```bash
./scripts/run_smoke_test.sh /tmp
```

## Pilot commands

Prepare the pinned FineWeb-Edu data before launching these pilots. Each command runs a paired AdamW baseline/WW-PGD experiment with the level-specific, `target_alpha: 2.0` cached-endpoint configuration. Replace the example data and results paths as needed.

**Level 0 pilot:**

```bash
wwgpt run-multiseed --level 0 --config configs/level0_adaptive_alpha.yaml --analysis-plan configs/analysis_plan_exploratory.yaml --data-root /path/to/data --results-root /path/to/results --token-multiplier 20 --seeds 1337 --extensions none,wwpgd
```

**Level 1 pilot:**

```bash
wwgpt run-multiseed --level 1 --config configs/level1_adaptive_alpha.yaml --analysis-plan configs/analysis_plan_exploratory.yaml --data-root /path/to/data --results-root /path/to/results --token-multiplier 20 --seeds 1337 --extensions none,wwpgd
```

**Level 2 pilot:**

```bash
wwgpt run-multiseed --level 2 --config configs/level2_adaptive_alpha.yaml --analysis-plan configs/analysis_plan_exploratory.yaml --data-root /path/to/data --results-root /path/to/results --token-multiplier 20 --seeds 1337 --extensions none,wwpgd
```

Run the paired acceleration and alpha-distance analyses only after both arms have completed:

```bash
wwgpt analyze-results /path/to/results --profile scaling --analysis-plan configs/analysis_plan_exploratory.yaml
```

Audit the same isolated experiment root before interpreting any output:

```bash
wwgpt audit-experiment --experiment-root /path/to/results
```

These commands make the repository capable of testing whether WW-PGD improves generalization. They do **not** establish that it does; that claim requires analysis of actual eligible paired results.

## WW-PGD installation policy

`python -m pip install -e .` installs WW-PGD through pip from the current default branch of `CalculatedContent/WW_PGD`. The dependency has no commit, tag, or branch pin. Each run records the package version and, when pip supplies PEP 610 VCS provenance, the commit that happened to be installed. That recorded commit is runtime metadata, not an installation requirement.

The nanoGPT adapter targets the public `ww_pgd` API. When the installed package exposes a native `diagnostic_logs` sink, those internal rows are retained. When it exposes only its established `ww_logs` output, the repository installs a compatibility wrapper that preserves normal WW-PGD execution and records the available pre-projection WeightWatcher fields plus adapter-observed candidate movement. Exact internal midpoint, Cayley-ratio, and TraceLog-retraction fields remain explicitly unsupported in that case; they are never invented and their absence does not prevent the experiment from running.

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
- **Measurement cadence:** evaluation, alpha measurement, and trap diagnostics default to every 10 optimizer steps; checkpoints default to every 50 steps. Composite spectral analysis is disabled by default. WW-PGD event cadence is a separate optimizer-step schedule, and cached-endpoint mode separates expensive measurements from fast endpoint relaxation.
- **Training:** batch size 16, gradient accumulation 1, dropout 0, flat layer learning rates, and warmup-cosine scheduling are the defaults.

Resolved manifests, rather than this summary, are authoritative for any individual run.

## Arms and intervention

Schema v3 separates a base optimizer from an extension. Canonical trials contain AdamW, Muon, and StableAdamW, each paired with its WW-PGD extension arm. Paired arms share initialization, sampling identity, tokenizer/corpus hashes, probes, and token budgets. WW-PGD runs after a successful base-optimizer step and acts only on eligible transformer matrices; it does not target embeddings, the tied LM head, LayerNorm values, or biases.

Adaptive and cached-endpoint modes are explicit ablations, not evidence of improved loss or generalization. Controller decisions are recorded separately from actual projection or relaxation moves, including requested/applied hardness, endpoint state, and trust-region clipping.

`wwpgd_internal_diagnostics.csv` records the diagnostics exposed by the installed package at fresh stock-candidate events. Native rows may include tail midpoint, Cayley, and TraceLog quantities. Compatibility rows are labeled `native_internal_diagnostics: false`, preserve available WeightWatcher and candidate-movement data, and list unsupported internal fields explicitly. Cached fast moves remain in `wwpgd_endpoint_relaxation.csv` and never fabricate fresh SVD diagnostics.

## Available analysis

`wwgpt analyze-results RESULTS_ROOT` discovers completed runs, inventories them, selects auditable paired arms, computes per-seed terminal differences and descriptive uncertainty summaries, and analyzes measured alpha trajectories. It writes a `scaling_fit_results.csv` marker with status `not_fit`; no scaling fit or efficacy conclusion is implied. Acceleration-analysis outputs are produced only when an explicit analysis plan is supplied and its prerequisites are met; their existence should not be described as a result without inspecting the generated artifacts.

Retired nominal-strength artifacts are not accepted by a dedicated public analysis or audit interface. They must not be interpreted as controller-dose, target-alpha, or rank-exponent scans.

## Resume and append-only behavior

New runs use unique timestamped directories and do not overwrite prior results. With `--resume`, the runner resumes a compatible incomplete run from its latest atomically written checkpoint; it does not silently reuse a completed run as new evidence. Resume validation checks configuration, data/tokenizer identity, initialization, schema, and code provenance. Incompatible continuation is refused unless the specifically supported code-version override is supplied, in which case an audit artifact is written and publication eligibility follows the recorded override policy. Controller and cached-endpoint state are checkpointed so continuation preserves scheduling decisions.

## Audit behavior

`wwgpt audit-experiment --experiment-root PATH` checks artifact completeness, arm identity, pair invariants, measurement schedules, finite metrics, schema/provenance fields, and recorded overrides. It writes machine-readable and Markdown reports under the analysis directory. Audit reports eligibility and exclusion reasons; it does not infer efficacy, significance, acceleration, scaling behavior, or generalization. `generate-reproducibility-report` summarizes recorded provenance but likewise makes no scientific outcome claim.

See `docs/SCHEMA_V3.md`, `docs/EXPERIMENT_RESUME_PROTOCOL.md`, `docs/RESULT_SCHEMA.md`, and `docs/SCIENTIFIC_INTEGRITY_POLICY.md` for schema and integrity details.
# Notebook workflow

The seven schema-v3 analysis notebooks consume completed runs and **never retrain
models**. Selected test metrics always come from validation-selected checkpoints;
next-token accuracy is not downstream classification accuracy.

```bash
export WWGPT_RESULTS_ROOT="$HOME/wwgpt-results-level0"
export WWGPT_NOTEBOOK_OUTPUT_DIR="$HOME/wwgpt-notebooks-level0"
export WWGPT_ANALYSIS_PLAN="configs/analysis_plan_exploratory.yaml"
export WWGPT_LEVEL=0
export WWGPT_TOKEN_MULTIPLIER=20
export WWGPT_BASE_OPTIMIZER=adamw
./scripts/run_analysis_notebooks.sh
```

Direct Papermill execution uses the same parameter precedence:

```bash
papermill notebooks/07_wwpgd_diagnostics.ipynb \
  "$WWGPT_NOTEBOOK_OUTPUT_DIR/executed/07_wwpgd_diagnostics.ipynb" \
  -p RESULTS_ROOT "$WWGPT_RESULTS_ROOT" \
  -p OUTPUT_ROOT "$WWGPT_NOTEBOOK_OUTPUT_DIR" \
  -p LEVEL 0 -p TOKEN_MULTIPLIER 20
```

Compatibility diagnostics contain observable candidate movement, never invented
private midpoint/Cayley/TraceLog values. Level 0--2 endpoint dose is normalized by
the refresh interval and bounded both per step and cumulatively per refresh.
`target_alpha = 2.0` remains the only public spectral target.
