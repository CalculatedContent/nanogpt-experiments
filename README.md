# nanoGPT WW-PGD Scaling

This repository provides an append-only framework for controlled nanoGPT experiments comparing AdamW with AdamW plus a repository-defined WeightWatcher-informed projected-gradient step (WW-PGD). The scientific question is whether WW-PGD preserves or improves validation loss, perplexity, next-token accuracy, token error, generalization gap, overfitting behavior, and movement of matrix spectral exponents toward alpha near 2.

## Quick start

```bash
./scripts/setup_environment.sh
./scripts/download_data.sh 0 /path/to/data 20
./scripts/run_five_seeds.sh 0 /path/to/data /path/to/results 20
./scripts/analyze_five_seeds.sh /path/to/results
./scripts/run_full_experiment.sh 0 /path/to/data /path/to/results 20
```

Smoke test only, invalid for scientific conclusions:

```bash
./scripts/run_smoke_test.sh /tmp
```


## Schema v3 highlights

Scientific schema v3 separates base optimizers from extensions. Use `--optimizer {adamw,muon,stableadamw}` with `--extensions none,wwpgd` for paired runs, or `--extension {none,wwpgd}` for a single arm. The six canonical arms are AdamW, AdamW+WW-PGD, Muon, Muon+WW-PGD, StableAdamW, and StableAdamW+WW-PGD.

The default ladder is level 0 `(n_layer=1,n_head=1,n_embd=64,block_size=256)`, then levels 1-4 `(2,2,128)`, `(4,3,192)`, `(6,4,256)`, and `(8,5,320)`, always with 64-dimensional attention heads. Blocks use separate bias-free key/query/value/projection matrices, bias-free MLP linears, and an untied bias-free LM head.

Training defaults are batch size 16, gradient accumulation 1, dropout 0, weight decay 0.01, and no gradient clipping when `grad_clip=0.0`. Warmup-cosine scheduling updates every optimizer parameter group before each optimizer step. Level-aware LLRD derives gamma from `llrd_min_multiplier=0.50` unless explicitly supplied. Token budgets are based on actual trainable parameter counts: target tokens default to `20 * parameter_count_used`, with extended horizons such as 40, 80, and 160 accepted.

Evaluation uses random-per-evaluation train and validation batches from deterministic SHA-256-derived streams; paired arms share hashes at the same evaluation index, and validation data is never concatenated with training data. WW-PGD runs after all base optimizer steps every `wwpgd_interval` optimizer steps (default `eval_interval`). Raw and composite WeightWatcher diagnostics use correct PyTorch linear conventions, including `OV=sum_h W_O,h @ W_V,h` and `VO=W_V @ W_O`. Schema-v2 and schema-v3 results are readable but not comparable as pooled paired statistics. See `docs/SCHEMA_V3.md`.

## Design

The AdamW baseline uses configurable learning rate, betas, epsilon, weight decay, warmup, cosine scheduling hooks, gradient clipping, batch size, gradient accumulation, dropout, evaluation interval, and checkpoint interval. WW-PGD first applies the normal AdamW step, then projects selected matrix-valued transformer layers with a documented local projection. WW-PGD is not standard WeightWatcher.

Default scientific data is `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, revision pinned in YAML. Data preparation streams documents, assigns deterministic train/validation splits by normalized-content SHA-256, keeps duplicate documents in the same split, trains the BPE tokenizer only on training-assigned documents, writes manifests and hashes, and refuses to repeat tokens. Tiny Shakespeare is not used for scientific experiments; local synthetic text is used only for infrastructure smoke tests.

Model ladder levels 0-4 are `(1,1,64)`, `(2,2,128)`, `(4,3,192)`, `(6,4,256)`, and `(8,5,320)` with block size 256. Actual instantiated parameter counts are the source of truth and reports include total, trainable, token embedding, position embedding, output head, embedding, and non-embedding parameters.

Scaling uses an experimental Chinchilla-style extrapolation `D = kN` with arbitrary positive token multipliers, including 20, 40, 80, and 160. Applying 20 tokens per parameter to tiny GPTs is an extrapolation, not a proven optimum. Valid scaling analysis requires a non-collinear grid over model size and token multiplier.

Five default paired seeds are 1337, 2027, 4099, 7919, and 104729. Paired arms share initialization, token order, model config, tokenizer and corpus hashes, and token budgets. Results are immutable run directories beneath `level_XX/pair_<id>/<optimizer>/run_<timestamp>_<suffix>`.

Analysis discovers completed runs, ignores incomplete runs, verifies pairs, computes seed-level uncertainty, paired WW-PGD minus AdamW differences, WeightWatcher layer trajectories, model-level spectral summaries, target-alpha distances, and plots. With five seeds, the repository reports limited statistical power and avoids strong claims.

Notebooks in `notebooks/` validate repository integrity, compare a single level, analyze WeightWatcher trajectories, inspect scaling laws, examine overfitting/generalization, and generate a summary report. Plotting helpers live in the package and use matplotlib.

Append-only behavior: scripts create new timestamped directories, preserve partial outputs, never recycle result directories, and never treat smoke tests as scaling-law evidence.

## Level-0 WW-PGD 0.5 from scratch

For a fully scripted level-0 run with five seeds, environment-variable setup, notebook execution, and a per-layer alpha-error confirmation for WW-PGD strength `0.5`, see [`docs/LEVEL0_WWPGD_0P5_FROM_SCRATCH.md`](docs/LEVEL0_WWPGD_0P5_FROM_SCRATCH.md). The entry point is:

```bash
DATA_ROOT=/tmp/wwpgd_v2/data RESULTS_ROOT=/tmp/wwpgd_level0_wwpgd_0p5 ./scripts/run_level0_wwpgd_0p5_from_scratch.sh
```

## WW-PGD Strength Scan

The strength scan is a secondary ablation for AdamW + WW-PGD. It is not run by default and does not change `wwgpt run-multiseed` or the default WW-PGD strength of `0.02`. The runner creates one shared initialization per seed, runs one immutable AdamW control, and reuses that control for every fixed WW-PGD strength arm. Every arm resets the deterministic token reader and fixed probes so only `wwpgd.strength` differs.

Run a scan:

```bash
wwgpt run-strength-scan --level 0 --data-root /tmp/wwpgd_v2/data --results-root /tmp/wwpgd_strength_scan --token-multiplier 20 --seeds 1337 --strengths 0.02,0.1,0.25,0.5,1.0 --device mps --eval-interval 25 --spectral-interval 100 --checkpoint-interval 500 --immediate-projection-spectral --resume
```

Analyze without notebooks:

```bash
wwgpt analyze-strength-scan --scan-root /tmp/wwpgd_strength_scan
```

Outputs are under `experiments/strength_scan/level_<NN>/multiplier_<M>/scan_<timestamp>_*`, with `seeds/`, one `adamw_control/`, per-strength run directories, and `analysis/strength_scan_*.csv`. Resume skips compatible completed arms and writes new append-only run directories for reruns. Immediate alpha before/after logging is written to `wwpgd_projection_spectral.csv`; it matches WeightWatcher layer names by `longname`/`name` and reports alpha-error changes, projection norms, and WeightWatcher overhead.

Open notebooks after setting `WWGPT_STRENGTH_SCAN_ROOT` to either a scan directory or parent results directory:

```bash
WWGPT_STRENGTH_SCAN_ROOT=/tmp/wwpgd_strength_scan jupyter lab notebooks/07_strength_scan_overview.ipynb notebooks/08_strength_scan_weightwatcher.ipynb
```
