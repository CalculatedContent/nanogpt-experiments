# Scientific schema v3

## Two-timescale endpoint records

The optional `cached_endpoint_relaxation` mode checkpoints endpoint and measurement
tensors, measurement step/index, raw and EMA alpha state, hardness/side/event
hardness, fit diagnostics, endpoint distances, active/invalidation state, last
application, cumulative movement, counters, controller version, and adapter mode.
`wwpgd_endpoint_measurements.csv` records slow selection/fit/cache decisions;
`wwpgd_endpoint_relaxation.csv` records requested/applied gain, before/after
distance, physical relative movement, trust-region scale, convergence, and
invalidation for fast actions. `num_evals` is never labeled as tail size.

`wwpgd_fast_control_steps.csv` is the append-only step-level schedule ledger. It
contains exactly one row for every scheduled cached-control attempt, including
measurement-step status, active endpoint count, changed layer count, whether any
change occurred, and an explicit skip reason. Schedule integrity uses this file,
not the optional per-layer relaxation rows. The run manifest records the actual
top-level `measurement.alpha_interval` used to audit slow measurements.

An analysis plan supplied to `run-multiseed` or `run-canonical-trials` is hashed
from its exact bytes before training. Its path, SHA-256, mode, paired-seed
requirement, thresholds, and primary outcomes are copied into pair/trial and arm
manifests. Confirmatory analysis writes `analysis_eligibility.json` and refuses
to proceed unless the supplied hash matches and every analyzed optimizer has the
required number of complete seed pairs. Exploratory analysis requires only one
positive complete pair.

Schema v3 composes a base optimizer (`adamw`, `muon`, `stableadamw`) with an extension (`none`, `wwpgd`). Canonical arms are `adamw`, `adamw_wwpgd`, `muon`, `muon_wwpgd`, `stableadamw`, and `stableadamw_wwpgd`; paired effects compare the same base optimizer with and without WW-PGD.

Base optimizers are authoritative and pairable. `adamw` uses standard `torch.optim.AdamW` with one documented parameter group per trainable parameter. `muon` uses the repository implementation matching the KellerJordan/modded-nanogpt Muon update with the Newton-Schulz coefficients recorded in `MUON_IMPLEMENTATION_VERSION`. `stableadamw` uses `optimi.StableAdamW` from the `torch-optimi` package, and run manifests record the installed package version. A requested optimizer construction failure is fatal; runs must not silently substitute AdamW for Muon or StableAdamW.

The model ladder uses one-layer level 0 `(1 layer, 1 head, width 64, block 256)` and levels `L>=1` use `n_layer=2L`, `n_head=L+1`, `n_embd=64(L+1)`, preserving 64-dimensional heads. Attention uses separate bias-free `key`, `query`, `value`, and `proj` matrices. MLP linear layers and the LM head are bias-free by default, and the default LM head is tied to token embeddings (`model.tie_weights: true`).

Training defaults are batch size 16, gradient accumulation 1, weight decay 0.1, dropout 0, and gradient clipping at 1.0. The default learning-rate policy is nanoGPT-compatible: one global normalized linear-warmup plus cosine-decay schedule is applied to every optimizer group, minimum LR is 10% of each group's peak LR, derived warmup is 1% of the resolved decay horizon, and the decay horizon defaults to the full optimizer-step training horizon. Layer learning rates are flat by default; LLRD is an explicit research ablation and manual layer multipliers are a legacy historical ablation.

Token budgets use the selected `transformer_body` parameter-count convention. With default `token_multiplier=20`, target tokens equal `20 * parameter_count_used`, steps are `ceil(target_tokens / (batch_size * block_size * gradient_accumulation))`, and realized tokens are the resulting full-step count times tokens per step. `max_steps` overrides `max_train_tokens`, which overrides token multiplier. Manifests record both the convention and selected count.

The default data mode is `fineweb_custom_bpe_scaling` with the pinned FineWeb-Edu source and custom training-only BPE. Evaluation, alpha measurement, and trap diagnostics default to every 10 optimizer steps; checkpointing defaults to every 50. Composite spectral analysis is disabled by default, and the WW-PGD apply cadence remains independent of measurement cadence.

Evaluation defaults to `random_per_eval`: new deterministic random train and validation windows are sampled at each evaluation event from independent SHA-256-derived streams. Paired arms share seeds and therefore evaluation hashes. Evaluation does not advance the training reader and restores train/eval mode.

WW-PGD is a post-step extension controlled by an optimizer-step interval. The canonical default `wwpgd_interval` is `1`, which projects after every successful base optimizer step; explicit positive intervals are supported ablations and are never derived from evaluation or logging cadence. Interval `N` applies WW-PGD at optimizer steps `N, 2N, 3N, ...`. A Level 1 interval ablation can use `--ww-interval 8`, preserving `blend_eta=0.5` while reducing projection events. It projects only raw eligible block matrices using WeightWatcher-selected large-eigenvalue tails from `xmin` and `detX_num`; embeddings, LayerNorms, biases, and the LM head are excluded by default.

Every schema-v3 run manifest records a normalized base `optimizer_fingerprint` covering optimizer type, parameter groups, learning rates, weight decay, betas or equivalent values, epsilon, and implementation versions. Within each baseline/WW-PGD pair, this fingerprint must match exactly; only extension metadata and extension outputs may differ.

Spectral diagnostics run independently of evaluation. Raw matrices include W_K, W_Q, W_V, W_O, W_MLP_IN, and W_MLP_OUT. Composite diagnostics include `KQ=W_K@W_Q`, `QK=W_Q@W_K`, `QK_effective=W_Q.T@W_K`, `KQ_effective=W_K.T@W_Q`, `OV=sum_h W_O,h@W_V,h`, `VO=W_V@W_O`, and `MLP_IO=W_MLP_OUT@W_MLP_IN`. Composite matrices are diagnostics only.

Schema-v2 and schema-v3 runs remain readable, but analysis must not pool them in statistical comparisons because optimizer arms, architecture, evaluation sampling, and projection schedules differ.

Schema-v3 reporting sources alpha trajectories from `alpha_measurements.csv`,
selected-checkpoint test loss/perplexity/accuracy from
`selected_checkpoint_metrics.json` (or its CSV twin), and normalized correlation
trap counts/fractions from `weightwatcher_aggregates.csv`. Missing WeightWatcher
fields stay missing; `detX_num` is not a correlation-trap metric.

## `wwpgd.adaptive`

Schema v3 supports an optional nested adaptive WW-PGD controller configuration. Defaults preserve prior behavior:

- `mode: uniform` applies layer hardness `1.0` to every otherwise eligible transformer matrix.
- `mode: alpha_linear` maps smoothed layer alpha above `wwpgd.target_alpha + deadband_above_target` to hardness in `[0, max_hardness]`, with `response_curve: linear` or `smoothstep`.
- `mode: alpha_piecewise` linearly interpolates an ordered list of `[alpha, hardness]` points and clamps outside the configured range.

Fields: `direction` (`above_target`), `response_curve`, `start_step`, `min_observations`, `alpha_ema_beta`, `deadband_above_target`, `full_strength_alpha`, `max_hardness`, `max_D`, `max_relative_frobenius_change`, `cooldown_events`, `piecewise_points`, `matrix_type_overrides`, and `layer_overrides`. `strength` remains a deprecated compatibility field and is not used as adaptive hardness.

The interval schedule controls only event timing. Adaptive hardness controls per-layer projection strength at an event. Run manifests record the resolved adaptive configuration, controller version, override precedence, expected projection steps, maximum `blend_eta`/`cayley_eta`, and trust-region setting. `run_complete.json` records layer-decision counts, skip-counts by reason, and aggregate applied-hardness and relative-Frobenius-change statistics.

### Adaptive WW-PGD metadata

Schema v3 manifests record whether adaptive WW-PGD is enabled, the adaptive mode and direction, above-target and below-target side settings, controller version, override precedence, expected scheduled event steps, maximum blend/Cayley eta, trust-region configuration, the fixed spectral-target policy, and the external WW_PGD commit/API version. `wwpgd_controller.csv` contains one decision row for every eligible layer at every scheduled event; `wwpgd_projection.csv` is reserved for actual changed projection rows. Requested and applied hardness/eta/displacement fields are separate so trust-region clipping is auditable.

### Stock WW_PGD candidate-displacement adapter

Schema v3 manifests record `wwpgd_commit: bf970cb6b73e977f8374114c442ae5b0589eccaa` and `wwpgd_adapter_mode: stock_candidate_displacement_scaling_v1`. This means the public WW_PGD API is used only as `ww_pgd_project(model, cfg, epoch=..., num_epochs=..., global_step=..., ww_logs=..., layer_selector=...)`; adaptive strength is implemented by `nanogpt-experiments` after stock candidate generation.

`wwpgd_controller.csv` has one row for every eligible layer at every scheduled event, including raw and smoothed alpha, alpha side, side-specific deadband/full-strength/max-hardness/response settings, requested controller hardness, global event hardness, requested and applied combined hardness, requested/applied relative-Frobenius changes, trust-region limit/scale, whether the stock candidate changed, and skip/projected state. `wwpgd_projection.csv` contains only actual applied rows whose applied relative-Frobenius change is positive. `num_evals` is recorded under that name and is not a selected-tail-size field.

Checkpoint state persists alpha observation counts, latest raw alpha, alpha EMA, last alpha side, signed error and distance, last applied projection event/hardness, accumulated controller decisions, scheduled event indexes, candidate-generation count, changed-event count, controller version, and adapter mode so resumed execution can reproduce uninterrupted controller decisions and scaled weights.
