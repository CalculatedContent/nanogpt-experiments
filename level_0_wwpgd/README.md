# Isolated Level 0 nanoGPT + WWPGD

This sibling experiment deliberately reuses the exact FineWeb-Edu GPT-2-BPE
model, AdamW optimizer, minibatch stream, learning-rate schedule, evaluation
probes, checkpoint policy, and WeightWatcher cadence from `level_0_baseline`.
The only scientific intervention is a fresh stock WWPGD event projection
immediately after each scheduled AdamW update.

## Frozen paired protocol

- same 10M/1M/1M document-disjoint FineWeb-Edu token splits;
- same 4-layer, 4-head, width-128, context-256 model;
- same initialization and training/evaluation RNG streams for matching seeds;
- same AdamW, effective batch 32, 2,000 steps, 20-step warmup and cosine decay;
- same final and validation-selected test policy;
- same transformer-only WeightWatcher measurements every 250 steps;
- WWPGD uses `event_projection`, target alpha 2.0, and interval 1;
- embeddings, LayerNorms, position embeddings, and the tied LM head are never projected;
- WWPGD candidates are generated on CPU for MPS safety and copied back only
  to the 24 transformer block matrices.

At the default interval, 2,000 optimizer steps imply 2,000 fresh WWPGD calls
and 48,000 matrix projection records. This is intentionally the direct
optimization-intervention protocol, not cached-endpoint relaxation.

## Install with Conda

```bash
conda activate ww_prod310
cd ~/Desktop/work/nanoGPT/nanogpt-experiments
python -m pip install -e .
python -m pip install -e 'level_0_baseline[data,analysis,test]'
python -m pip install -e 'level_0_wwpgd[data,analysis,test]'
```

## Reuse the exact baseline data

```bash
export NANOGPT_LEVEL0_ROOT=/tmp/nanogpt-level0-bpe
export NANOGPT_LEVEL0_DATA_ROOT=$NANOGPT_LEVEL0_ROOT/data

export NANOGPT_LEVEL0_WWPGD_ROOT=/tmp/nanogpt-level0-wwpgd
export NANOGPT_LEVEL0_WWPGD_DATA_ROOT=$NANOGPT_LEVEL0_DATA_ROOT
export NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT=$NANOGPT_LEVEL0_WWPGD_ROOT/results
```

If the BPE data has not been prepared, run:

```bash
./level_0_wwpgd/scripts/prepare_data.sh
```

## Run one seed

```bash
cd level_0_wwpgd
./scripts/run_one.sh adamw 1337 \
  2>&1 | tee /tmp/level0-wwpgd-seed1337.log
```

## Run five seeds

```bash
export NANOGPT_LEVEL0_WWPGD_SEEDS=1337,2027,4099,7919,104729
./scripts/run_multiseed.sh \
  2>&1 | tee /tmp/level0-wwpgd-5seeds.log
```

Each completed run writes `metrics.csv`, `run_complete.json`, final and
selected checkpoints, periodic WeightWatcher files, and
`wwpgd_projection.csv` with one row per projected transformer matrix.

## Notebooks

```bash
export NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT=/tmp/nanogpt-level0-wwpgd/results
jupyter lab notebooks/01_single_seed.ipynb
jupyter lab notebooks/02_multiseed.ipynb
jupyter lab notebooks/03_compare_baseline_wwpgd.ipynb
```
