# Level 1: 50M-class paired AdamW versus AdamW + WWPGD

Level 1 is the first long-running scale experiment. It keeps the paired scientific design while increasing depth, width, context, and training horizon.

## Protocol

| quantity | Level 0 | Level 1 |
|---|---:|---:|
| layers | 4 | 8 |
| heads | 4 | 8 |
| embedding width | 128 | 512 |
| context | 256 | 512 |
| approximate parameters | 7.2M | 51M |
| optimizer steps | 2,000 | 10,000 |
| tokens per step | 8,192 | 8,192 |
| tokens processed per run | 16.4M | 81.9M |

The larger model uses a 3e-4 peak learning rate, 200-step warmup, and cosine decay to 3e-5. AdamW and AdamW+WWPGD remain identical in model, initialization seed, data, minibatch stream, optimizer schedule, evaluation probes, and checkpoints. WWPGD remains an event projection after every optimizer update with target alpha 2.0.

This is a compute-constrained scaling experiment, not a Chinchilla-optimal pretraining run. A roughly 51M-parameter model would require on the order of a billion unique training tokens for a fully compute-optimal run; that is intentionally outside the laptop budget. Here we test whether the Level 0 WWPGD effect survives a roughly sevenfold parameter increase and a fivefold longer optimization horizon.

## Before the long run

```bash
conda activate ww_prod310
cd ~/Desktop/work/nanoGPT/nanogpt-experiments
git switch experiment/level-1-scaleup
git pull --ff-only origin experiment/level-1-scaleup

jupyter lab level_1_wwpgd/notebooks/01_protocol_and_scale.ipynb
```

Run one matched smoke-test pair first:

```bash
caffeinate -dimsu env \
  NANOGPT_LEVEL1_PAIRED_SEEDS="1337" \
  NANOGPT_LEVEL1_DATA_ROOT="/tmp/nanogpt-level0-bpe/data" \
  NANOGPT_LEVEL1_DEVICE="mps" \
  bash scripts/run_isolated_level1_pair.sh all
```

The smoke test still runs the full 10,000 steps. For a short mechanical test only, use the existing environment overrides directly with the single-run wrappers rather than treating the result as scientific data.

## Main campaign

The compute-budgeted default is three matched seeds, which produces preliminary error bars and is the most plausible multi-day Mac run:

```bash
caffeinate -dimsu env \
  NANOGPT_LEVEL1_PAIRED_SEEDS="1337,2027,4099" \
  NANOGPT_LEVEL1_DATA_ROOT="/tmp/nanogpt-level0-bpe/data" \
  NANOGPT_LEVEL1_DEVICE="mps" \
  bash scripts/run_isolated_level1_pair.sh all
```

For a confirmatory five-seed campaign, use:

```bash
NANOGPT_LEVEL1_PAIRED_SEEDS="1337,2027,4099,7919,104729" \
  bash scripts/run_isolated_level1_pair.sh all
```

Runtime cannot be predicted reliably without a benchmark on the exact Mac. WWPGD performs CPU-side spectral work after every optimizer step, so the WWPGD arm will dominate elapsed time. The one-seed run is the calibration measurement; multiply its elapsed time by three for the default campaign.

## Verify and analyze

```bash
bash scripts/run_isolated_level1_pair.sh verify

PAIR_ROOT="$(cat /tmp/nanogpt-level1-current-pair)"
env \
  NANOGPT_LEVEL1_BASELINE_RESULTS_ROOT="$PAIR_ROOT/baseline/results" \
  NANOGPT_LEVEL1_WWPGD_RESULTS_ROOT="$PAIR_ROOT/wwpgd/results" \
  jupyter lab level_1_wwpgd/notebooks/02_compare_multiseed.ipynb
```

The comparison notebook plots validation loss as mean plus or minus one sample standard deviation, reports paired endpoint deltas, and plots every WeightWatcher matrix alpha with cross-seed standard-deviation bands.
