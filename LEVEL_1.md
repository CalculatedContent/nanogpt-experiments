# Level 1: scaled paired AdamW versus AdamW + WWPGD

Level 1 repeats the frozen Level 0 paired experiment while changing only model capacity.

## Scale change

| quantity | Level 0 | Level 1 |
|---|---:|---:|
| layers | 4 | 6 |
| heads | 4 | 6 |
| embedding width | 128 | 384 |
| context | 256 | 256 |
| approximate parameters | 7.2M | 30.0M |

Level 1 is approximately 4.2x larger by parameter count. The tokenizer, shared 10M/1M/1M FineWeb-Edu corpus, context length, effective batch, optimizer, learning-rate schedule, 2,000-step horizon, evaluation cadence, seeds, and WWPGD event projection remain unchanged.

## Run the paired experiment

```bash
conda activate ww_prod310
cd ~/Desktop/work/nanoGPT/nanogpt-experiments

caffeinate -dimsu env \
  NANOGPT_LEVEL1_PAIRED_SEEDS="1337,2027,4099,7919,104729" \
  NANOGPT_LEVEL1_DATA_ROOT="/tmp/nanogpt-level0-bpe/data" \
  NANOGPT_LEVEL1_DEVICE="mps" \
  bash scripts/run_isolated_level1_pair.sh all
```

The Level 1 runner deliberately reuses the prepared Level 0 token files so both scales see the same immutable corpus.

## Verify and compare

```bash
bash scripts/run_isolated_level1_pair.sh verify

PAIR_ROOT="$(cat /tmp/nanogpt-level1-current-pair)"
env \
  NANOGPT_LEVEL0_BASELINE_RESULTS_ROOT="$PAIR_ROOT/baseline/results" \
  NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT="$PAIR_ROOT/wwpgd/results" \
  jupyter lab level_0_wwpgd/notebooks/03_compare_baseline_wwpgd.ipynb
```

The comparison remains paired by seed. Report endpoint and best-validation differences as WWPGD minus baseline, together with the mean, standard deviation, and 95% confidence interval across the five seed pairs.

## Operational warning

WWPGD performs CPU-side spectral work after every AdamW update. At roughly 30M parameters this arm will be substantially slower than Level 0. Run one smoke-test seed before committing to all five when using a laptop:

```bash
NANOGPT_LEVEL1_PAIRED_SEEDS=1337 bash scripts/run_isolated_level1_pair.sh all
```
