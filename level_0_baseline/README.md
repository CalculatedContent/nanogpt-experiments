# Level 0 Baseline

A deliberately self-contained nanoGPT baseline. It does not import the repository's existing experiment framework or WW-PGD code.

## Scope

- one transformer block, one ordinary Q/K/V attention head, width 64, context 256
- byte-level next-token language modeling on a fixed FineWeb-Edu subset
- AdamW or Muon with a global warmup/cosine schedule
- Muon applies only to hidden 2-D matrices; AdamW handles embeddings, tied LM head, LayerNorm parameters, and other non-matrix parameters
- deterministic seeds for initialization and sampled training windows
- immutable train, validation, and test splits
- CSV logging of loss, next-token accuracy, perplexity, validation/test generalization gaps, gradient norm, weight norm, tokens, and elapsed time
- optional checkpoint-time WeightWatcher layer analysis
- single-seed and multi-seed notebooks; multi-seed plots use mean ± one standard deviation shaded bands

## Install

```bash
cd level_0_baseline
python -m venv .venv
source .venv/bin/activate
pip install -e '.[data,analysis,test]'
```

## Paths

Defaults are under `/tmp/nanogpt-level0`. Override them without editing code:

```bash
export NANOGPT_LEVEL0_DATA_ROOT=/tmp/my-level0/data
export NANOGPT_LEVEL0_RESULTS_ROOT=/tmp/my-level0/results
export NANOGPT_LEVEL0_CACHE_ROOT=/tmp/my-level0/cache
```

## Prepare the real corpus

This prepares fixed 50 MB training, 2 MB validation, and 2 MB test byte-token splits from streamed FineWeb-Edu:

```bash
level0-prepare-data --dataset fineweb-edu
```

## Run one seed

```bash
./scripts/run_one.sh adamw 1337
./scripts/run_one.sh muon 1337
```

## Run multiple seeds

```bash
NANOGPT_LEVEL0_SEEDS=1337,2027,4099 ./scripts/run_multiseed.sh
```

The notebooks read `NANOGPT_LEVEL0_RESULTS_ROOT`. Select the single-seed run with `NANOGPT_LEVEL0_NOTEBOOK_OPTIMIZER` and `NANOGPT_LEVEL0_NOTEBOOK_SEED`.

For a bounded infrastructure smoke test, override the run length and batch size:

```bash
NANOGPT_LEVEL0_MAX_STEPS=2 NANOGPT_LEVEL0_BATCH_SIZE=2 NANOGPT_LEVEL0_EVAL_INTERVAL=1 ./scripts/run_one.sh adamw 1337
```

Next-token error is `1 - next-token accuracy`; the notebooks derive and plot it explicitly.
