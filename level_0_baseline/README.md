# Isolated Level 0 nanoGPT Baseline

This subtree provides a clean, self-contained AdamW/Muon baseline without importing the repository's WW-PGD experiment framework. It is intended to establish a realistic FineWeb-Edu language-model trajectory and WeightWatcher alpha baseline before optimizer interventions are added.

## Scientific protocol

- pinned FineWeb-Edu `sample-10BT` stream
- GPT-2 BPE tokenizer (`50,257` tokens), not raw bytes
- fixed 10M-token training, 1M-token validation, and 1M-token test splits, with document-disjoint split boundaries
- 4 transformer blocks, 4 heads, width 128, context 256
- tied token embedding/output head
- AdamW peak learning rate `6e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`
- matrix-only weight decay `0.1`; no decay on one-dimensional parameters
- micro-batch 4 with eight-step gradient accumulation (effective batch 32 sequences)
- 20-step linear warmup (1% of the run) followed by cosine decay to `6e-5`
- gradient clipping at 1.0 and flat learning rate across layers
- fixed train/validation probes with RNG streams independent of training
- test evaluation only at the final and validation-selected checkpoints
- WeightWatcher analysis every 250 steps on transformer block matrices only; embeddings and the large tied output matrix are excluded

The default 2,000-step run processes 16,384,000 training tokens. The primary metrics are held-out cross-entropy loss and perplexity; exact BPE-token top-1 accuracy is secondary.

## Install with Conda

```bash
conda activate ww_prod310
cd ~/Desktop/work/nanoGPT/nanogpt-experiments/level_0_baseline
python -m pip install -e '.[data,analysis,test]'
python -m pip check
```

## Paths

The corrected BPE experiment uses a new root so it cannot silently reuse the obsolete byte-token data:

```bash
export NANOGPT_LEVEL0_ROOT=/tmp/nanogpt-level0-bpe
export NANOGPT_LEVEL0_DATA_ROOT=$NANOGPT_LEVEL0_ROOT/data
export NANOGPT_LEVEL0_RESULTS_ROOT=$NANOGPT_LEVEL0_ROOT/results
export NANOGPT_LEVEL0_CACHE_ROOT=$NANOGPT_LEVEL0_ROOT/cache
export HF_HOME=$NANOGPT_LEVEL0_CACHE_ROOT/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
```

## Prepare the pinned FineWeb-Edu corpus

```bash
level0-prepare-data \
  --dataset fineweb-edu \
  --verbose \
  --log-interval-seconds 5 \
  2>&1 | tee /tmp/level0-bpe-prepare.log
```

The command writes `train.bin`, `val.bin`, `test.bin`, and `meta.json` under the data root. The binary files contain `uint16` GPT-2 BPE token IDs.

## Run one AdamW seed

```bash
./scripts/run_one.sh adamw 1337 \
  2>&1 | tee /tmp/level0-bpe-adamw-seed1337.log
```

A successful run ends with:

```text
[level0-train] complete ...
```

and writes `run_complete.json`, `metrics.csv`, `checkpoint_best.pt`, `checkpoint_final.pt`, `selected_checkpoint_metrics.json`, and periodic `weightwatcher_step_*.csv` files.

Use a fresh results root for a new scientific run. To deliberately replace an existing run, set:

```bash
export NANOGPT_LEVEL0_OVERWRITE=1
```

## Run multiple AdamW seeds

```bash
NANOGPT_LEVEL0_SEEDS=1337,2027,4099 \
NANOGPT_LEVEL0_OPTIMIZERS=adamw \
  ./scripts/run_multiseed.sh
```

## Plot a single run

```bash
export NANOGPT_LEVEL0_NOTEBOOK_OPTIMIZER=adamw
export NANOGPT_LEVEL0_NOTEBOOK_SEED=1337
jupyter lab notebooks/01_single_seed.ipynb
```

The notebook plots train/validation loss, perplexity, secondary BPE-token accuracy, learning rate, generalization gap, and WeightWatcher alpha by transformer matrix type.

## Bounded infrastructure test

This checks code paths only and is not a scientific result:

```bash
NANOGPT_LEVEL0_MAX_STEPS=2 \
NANOGPT_LEVEL0_BATCH_SIZE=2 \
NANOGPT_LEVEL0_GRAD_ACCUM_STEPS=1 \
NANOGPT_LEVEL0_EVAL_INTERVAL=1 \
NANOGPT_LEVEL0_WW_INTERVAL=1 \
  ./scripts/run_one.sh adamw 1337
```
