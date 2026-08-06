# Level Zero optimizer baselines for nanoGPT

This folder is the canonical **baseline-only** Level Zero experiment. It compares three optimizers under a matched nanoGPT protocol:

1. SGD with Nesterov momentum;
2. AdamW;
3. Muon on hidden matrices with auxiliary AdamW on embeddings, the output head, normalization parameters, and other non-Muon parameters.

Each optimizer runs the same three seeds (`1337`, `2027`, `4099`), model, tokenizer, data splits, effective batch, and training-token budget. Only optimizer-specific hyperparameters and schedules differ.

## Confirmatory protocol

| Component | Fixed value |
|---|---:|
| Dataset | pinned FineWeb-Edu `sample-10BT` stream |
| Tokenizer | GPT-2 BPE, vocabulary 50,257 |
| Train / validation / test | 10M / 1M / 1M tokens, document-disjoint |
| Context | 256 tokens |
| Model | 4 transformer blocks, 4 heads, width 128 |
| Effective batch | 4 micro-batches × 8 accumulation steps = 32 sequences |
| Tokens per optimizer step | 8,192 |
| Training | 6,104 optimizer steps = 50,003,968 tokens ≈ 5 passes over train |
| Seeds | 1337, 2027, 4099 |
| Precision | float32 |
| Primary device | Apple MPS, with CPU fallback |

The test split is not used to draw learning curves. It is evaluated only at the final checkpoint and at the checkpoint selected by validation loss. Training and validation trajectories use fixed probes whose RNG streams are independent of minibatch sampling.

## Optimizer profiles

The exact profiles live in `configs/level0.yaml`.

| Optimizer | Peak learning rate | Floor | Warm-up | Decay | Other settings |
|---|---:|---:|---:|---|---|
| SGD + Nesterov | 0.05 | 0.005 | 10% | cosine | momentum 0.90, weight decay 0.01 |
| AdamW | 6e-4 | 6e-5 | 1% | cosine | betas (0.90, 0.95), weight decay 0.10 |
| Muon hidden weights | 0.02 | 0.002 | 5% | cosine | momentum 0.95, 5 Newton-Schulz steps, weight decay 0.01 |
| Muon auxiliary AdamW | 3e-4 | 3e-5 | 5% | cosine | betas (0.90, 0.95), weight decay 0.01 |

`HYPERPARAMETERS.md` records the rationale and source trail. These are strong, literature-aligned baseline settings, not a claim that one fixed learning rate is universally optimal for every Mac, dataset realization, or model scale.

## Metrics

Every run writes:

- train and validation cross-entropy, perplexity, and exact next-token top-1 accuracy;
- final and validation-selected test loss, perplexity, and accuracy;
- validation/test generalization gaps;
- pre-clip and post-clip gradient norms and clipping incidence;
- model weight norm, interval update norm, and update-to-weight ratio;
- elapsed time, token throughput, learning rates, and MPS memory counters;
- WeightWatcher layer details and aggregate trajectories, including `alpha`, `alpha_weighted`, `ERG_gap`, fit distance `D`, stable rank, MP soft rank, spectral/log norms, entropy, spike counts, and any additional raw columns returned by the installed WeightWatcher version.

WeightWatcher is called with `ERG=True`. Missing alpha or ERG values remain missing; the code does not manufacture fallback estimates.

## Error bars and plots

Trajectory plots use an across-seed **Bollinger-style envelope**: mean ± 2 sample standard deviations at each common evaluation step. Final test tables and bar charts use two-sided 95% Student-t confidence intervals across the three seeds. The notebooks use a colorblind-safe Okabe-Ito palette consistently:

- blue: SGD + momentum;
- orange: AdamW;
- green: Muon.

## Mac setup

From the repository root:

```bash
bash level_0_baseline/scripts/setup_mac.sh
```

This creates `level_0_baseline/.venv-level0`, installs data/analysis/test dependencies, registers a Jupyter kernel, and reports whether PyTorch can see MPS. The scientific runs remain float32 and leave `torch.compile` disabled by default for MPS reliability.

A Conda specification is also provided:

```bash
cd level_0_baseline
conda env create -f environment-mac.yml
conda activate nanogpt-level0-baselines
python -m pip install -e '.[data,analysis,test]'
```

## Prepare the pinned corpus

```bash
bash level_0_baseline/scripts/prepare_data.sh
```

The default experiment root is:

```text
/tmp/nanogpt-level0-baselines
```

Override it without changing code:

```bash
export NANOGPT_LEVEL0_ROOT=/path/with/space/for/results
```

## Run all nine confirmatory jobs

```bash
bash level_0_baseline/scripts/run_all_baselines.sh
```

Runs are sequential, which avoids simultaneous MPS memory pressure. Completed runs are skipped; incomplete runs resume from `checkpoint_latest.pt`. The runner generates final-checkpoint samples after each successful run.

Run one optimizer across all seeds:

```bash
bash level_0_baseline/scripts/run_multiseed.sh sgd_momentum
bash level_0_baseline/scripts/run_multiseed.sh adamw
bash level_0_baseline/scripts/run_multiseed.sh muon
```

Run one seed:

```bash
bash level_0_baseline/scripts/run_one.sh muon 1337
```

## Four notebooks

Each optimizer notebook can execute its three runs independently and then analyze them:

- `notebooks/01_sgd_momentum_baseline.ipynb`
- `notebooks/02_adamw_baseline.ipynb`
- `notebooks/03_muon_baseline.ipynb`
- `notebooks/04_compare_baselines.ipynb`

The final cell in each optimizer notebook generates text from each seed's `checkpoint_final.pt`. The comparison notebook verifies protocol identity before overlaying curves.

## Output layout

```text
results/
  sgd_momentum/
    seed_1337/
      manifest.json
      metrics.csv
      checkpoint_latest.pt
      checkpoint_best.pt
      checkpoint_final.pt
      test_results.json
      run_complete.json
      generated_samples.json
      spectral/
        summary.csv
        layers.csv
        raw/weightwatcher_step_*.csv
  adamw/
  muon/
```

## Validation

The bounded tests exercise all three optimizer update paths, schedules, checkpoint/test policies, aggregation statistics, generation mechanics, and notebook structure without pretending to be scientific results:

```bash
bash level_0_baseline/scripts/smoke_test.sh
```

Full five-epoch, three-seed MPS runs must be executed on the target MacBook Pro. No precomputed baseline result is committed as if it came from that hardware.
