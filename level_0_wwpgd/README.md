# Isolated Level 0 nanoGPT + WWPGD

This sibling experiment deliberately reuses the exact FineWeb-Edu GPT-2-BPE model, AdamW optimizer, minibatch stream, learning-rate schedule, evaluation probes, checkpoint policy, and WeightWatcher cadence from `level_0_baseline`. The only scientific intervention is a fresh stock WWPGD event projection immediately after each successful AdamW update.

## Frozen paired protocol

- same 10M/1M/1M document-disjoint FineWeb-Edu token splits;
- same 4-layer, 4-head, width-128, context-256 model;
- same initialization and training/evaluation RNG streams for matching seeds;
- same AdamW, effective batch 32, 2,000 steps, 20-step warmup and cosine decay;
- same final and validation-selected test policy;
- same transformer-only WeightWatcher measurements every 250 steps;
- WWPGD uses `event_projection`, target alpha 2.0, and interval 1;
- embeddings, LayerNorms, position embeddings, and the tied LM head are never projected;
- WWPGD candidates are generated on CPU for MPS safety and copied back only to the 24 transformer block matrices.

At the default interval, 2,000 optimizer steps imply 2,000 fresh WWPGD calls and 48,000 matrix projection records per seed. This is the direct optimization-intervention protocol, not cached-endpoint relaxation.

## Shell-safety contract

Run repository scripts with `bash`; do not source them. Each script refuses to run when sourced, and strict shell options stay inside the child script rather than changing the interactive shell.

```bash
bash scripts/prepare_data.sh
bash scripts/run_one.sh adamw 1337
bash scripts/run_multiseed.sh
```

The scripts are also tracked as executable, but the explicit `bash` form remains portable when a checkout or archive loses executable permission.

## Install with Conda

```bash
conda activate ww_prod310
cd ~/Desktop/work/nanoGPT/nanogpt-experiments
python -m pip install -e .
python -m pip install -e 'level_0_baseline[data,analysis,test]'
python -m pip install -e 'level_0_wwpgd[data,analysis,test]'
python -m pip check
```

## Run the complete paired five-seed experiment

The safe paired runner creates a timestamped `/tmp` result root, verifies that the baseline and WWPGD configurations match outside the intervention, prepares or reuses the exact shared data identity, and runs each phase without modifying the parent shell.

Run the five-seed AdamW baseline first:

```bash
cd ~/Desktop/work/nanoGPT/nanogpt-experiments
bash scripts/run_isolated_level0_pair.sh baseline
```

Then, in the same or a later terminal, run the matching WWPGD seeds using the recorded pair root:

```bash
PAIR_ROOT="$(cat /tmp/nanogpt-level0-current-pair)"
bash scripts/run_isolated_level0_pair.sh wwpgd "$PAIR_ROOT"
```

To run both phases sequentially in one child process:

```bash
bash scripts/run_isolated_level0_pair.sh all
```

To inspect completion without launching training:

```bash
bash scripts/run_isolated_level0_pair.sh verify
```

The canonical seeds are:

```text
1337, 2027, 4099, 7919, 104729
```

The runner skips a seed only when its `run_complete.json` explicitly says it completed. It refuses to overwrite a partial directory.

## Direct WWPGD commands

Reuse the baseline data directly:

```bash
env \
  NANOGPT_LEVEL0_WWPGD_DATA_ROOT=/tmp/nanogpt-level0-bpe/data \
  NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT=/tmp/nanogpt-level0-wwpgd/results \
  NANOGPT_LEVEL0_WWPGD_DEVICE=mps \
  bash level_0_wwpgd/scripts/run_one.sh adamw 1337 \
  2>&1 | tee /tmp/level0-wwpgd-seed1337.log
```

For five direct seeds:

```bash
env \
  NANOGPT_LEVEL0_WWPGD_DATA_ROOT=/tmp/nanogpt-level0-bpe/data \
  NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT=/tmp/nanogpt-level0-wwpgd/results \
  NANOGPT_LEVEL0_WWPGD_DEVICE=mps \
  NANOGPT_LEVEL0_WWPGD_SEEDS=1337,2027,4099,7919,104729 \
  bash level_0_wwpgd/scripts/run_multiseed.sh \
  2>&1 | tee /tmp/level0-wwpgd-5seeds.log
```

Each completed run writes `metrics.csv`, `run_complete.json`, final and selected checkpoints, periodic WeightWatcher files, and `wwpgd_projection.csv` with one row per projected transformer matrix.

## Compare baseline and WWPGD

```bash
PAIR_ROOT="$(cat /tmp/nanogpt-level0-current-pair)"

env \
  NANOGPT_LEVEL0_BASELINE_RESULTS_ROOT="$PAIR_ROOT/baseline/results" \
  NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT="$PAIR_ROOT/wwpgd/results" \
  jupyter lab level_0_wwpgd/notebooks/03_compare_baseline_wwpgd.ipynb
```

The paired notebook plots mean and standard-deviation bands and computes seed-matched test-loss differences.
