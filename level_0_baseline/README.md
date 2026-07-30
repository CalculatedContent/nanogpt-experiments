# Isolated Level 0 nanoGPT Baseline

This subtree provides the AdamW control arm for the isolated FineWeb-Edu GPT-2-BPE Level 0 experiment. The matching intervention arm lives in `level_0_wwpgd/`.

## Scientific protocol

- pinned FineWeb-Edu `sample-10BT` stream;
- GPT-2 BPE tokenizer (`50,257` tokens), not raw bytes;
- fixed 10M-token training, 1M-token validation, and 1M-token test splits with document-disjoint boundaries;
- 4 transformer blocks, 4 heads, width 128, context 256;
- tied token embedding/output head;
- AdamW peak learning rate `6e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`;
- matrix-only weight decay `0.1`;
- micro-batch 4 with eight-step gradient accumulation;
- 20-step linear warmup followed by cosine decay to `6e-5`;
- gradient clipping at 1.0 and a flat learning rate across layers;
- fixed train/validation probes with RNG streams independent of training;
- test evaluation only at the final and validation-selected checkpoints;
- WeightWatcher analysis every 250 steps on transformer block matrices only.

The default 2,000-step run processes 16,384,000 training tokens. Held-out cross-entropy loss and perplexity are primary; exact BPE-token top-1 accuracy is secondary.

## Shell-safety contract

Run repository scripts with `bash`; do not source them. Each script refuses to run when sourced, and all strict shell options remain inside the child script rather than changing the interactive shell.

```bash
bash scripts/run_one.sh adamw 1337
bash scripts/run_multiseed.sh
```

The scripts are also tracked as executable, but the explicit `bash` form is portable even when a checkout or archive loses executable permission.

## Install with Conda

```bash
conda activate ww_prod310
cd ~/Desktop/work/nanoGPT/nanogpt-experiments
python -m pip install -e .
python -m pip install -e 'level_0_baseline[data,analysis,test]'
python -m pip install -e 'level_0_wwpgd[data,analysis,test]'
python -m pip check
```

## Prepare the pinned FineWeb-Edu corpus

The shared data identity is stored under `/tmp/nanogpt-level0-bpe/data` by default.

```bash
env NANOGPT_LEVEL0_DATA_ROOT=/tmp/nanogpt-level0-bpe/data \
  level0-prepare-data \
    --dataset fineweb-edu \
    --verbose \
    --log-interval-seconds 5 \
  2>&1 | tee /tmp/level0-bpe-prepare.log
```

The command writes `train.bin`, `val.bin`, `test.bin`, and `meta.json` as `uint16` GPT-2 BPE token IDs.

## Run the five-seed baseline with error bars

From the repository root, use the paired runner. It creates a fresh timestamped pair root under `/tmp`, runs the five AdamW seeds, records the pair root in `/tmp/nanogpt-level0-current-pair`, skips seeds that already have valid completion markers, and never modifies the parent shell.

```bash
cd ~/Desktop/work/nanoGPT/nanogpt-experiments
bash scripts/run_isolated_level0_pair.sh baseline
```

The canonical seeds are:

```text
1337, 2027, 4099, 7919, 104729
```

To continue the same pair after reopening a terminal:

```bash
PAIR_ROOT="$(cat /tmp/nanogpt-level0-current-pair)"
bash scripts/run_isolated_level0_pair.sh baseline "$PAIR_ROOT"
```

## Run a direct single seed

```bash
cd ~/Desktop/work/nanoGPT/nanogpt-experiments

env \
  NANOGPT_LEVEL0_DATA_ROOT=/tmp/nanogpt-level0-bpe/data \
  NANOGPT_LEVEL0_RESULTS_ROOT=/tmp/nanogpt-level0-bpe/results \
  NANOGPT_LEVEL0_DEVICE=mps \
  bash level_0_baseline/scripts/run_one.sh adamw 1337 \
  2>&1 | tee /tmp/level0-bpe-adamw-seed1337.log
```

## Plot the five-seed baseline

```bash
PAIR_ROOT="$(cat /tmp/nanogpt-level0-current-pair)"

env \
  NANOGPT_LEVEL0_RESULTS_ROOT="$PAIR_ROOT/baseline/results" \
  jupyter lab level_0_baseline/notebooks/02_multiseed.ipynb
```

The notebook plots mean and standard-deviation bands for validation loss, perplexity, secondary accuracy, generalization gap, and WeightWatcher alpha.

## Bounded infrastructure test

This checks code paths only and is not a scientific result:

```bash
env \
  NANOGPT_LEVEL0_MAX_STEPS=2 \
  NANOGPT_LEVEL0_BATCH_SIZE=2 \
  NANOGPT_LEVEL0_GRAD_ACCUM_STEPS=1 \
  NANOGPT_LEVEL0_EVAL_INTERVAL=1 \
  NANOGPT_LEVEL0_WW_INTERVAL=1 \
  NANOGPT_LEVEL0_OVERWRITE=1 \
  bash level_0_baseline/scripts/run_one.sh adamw 1337
```
