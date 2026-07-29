# Isolated Level 0 nanoGPT Baseline

This subtree is an independent, auditable nanoGPT baseline. It does not import the repository's WW-PGD experiment framework or apply WW-PGD. Its purpose is to establish a credible AdamW language-model baseline and measure WeightWatcher layer spectra before adding an optimizer extension.

## Corrected baseline

The original isolated baseline was an 82K-parameter, one-layer byte model. This version uses a materially more realistic MacBook-scale configuration:

- pinned FineWeb-Edu `sample-10BT` source;
- GPT-2 BPE tokenization (`tiktoken`, vocabulary 50,257);
- 16M training, 1M validation, and 1M test tokens stored as `uint16`;
- 4 transformer blocks, 4 attention heads, width 128, context 256;
- 7,253,248 parameters with tied token embedding/output weights;
- AdamW with decoupled weight decay, gradient clipping, 100-step warmup, and cosine decay;
- microbatch 4 with 8 accumulation steps: 8,192 tokens per optimizer update and 16.384M tokens over 2,000 updates;
- fixed, independent train/validation/test probes that do not advance the training RNG;
- test evaluation only at the final and validation-selected checkpoints;
- non-randomized WeightWatcher analysis of the 24 transformer block matrices only. The large embedding/output matrix is excluded from periodic spectral analysis.

The default layer learning-rate multiplier is flat (`layer_lr_decay: 1.0`). Layerwise decay is available as an explicit ablation, not silently enabled in the baseline.

## Install in the existing Conda environment

```bash
conda activate ww_prod310
cd ~/Desktop/work/nanoGPT/nanogpt-experiments/level_0_baseline
python -m pip install -e '.[data,analysis,test]'
```

## Paths

The corrected format uses a new root so the earlier raw-byte files cannot be mistaken for GPT-2-tokenized data:

```bash
export NANOGPT_LEVEL0_ROOT=/tmp/nanogpt-level0-gpt2
export NANOGPT_LEVEL0_DATA_ROOT=$NANOGPT_LEVEL0_ROOT/data
export NANOGPT_LEVEL0_RESULTS_ROOT=$NANOGPT_LEVEL0_ROOT/results
```

## Prepare the pinned FineWeb-Edu corpus

```bash
level0-prepare-data \
  --config configs/level0.yaml \
  --dataset fineweb-edu \
  --verbose \
  --log-interval-seconds 10
```

The preparer reports documents, GPT-2 tokens, elapsed time, throughput, ETA, and time since the stream last produced tokens. It validates compatible existing data and reuses it; pass `--force` to rebuild it.

A successful preparation produces:

```text
/tmp/nanogpt-level0-gpt2/data/train.bin
/tmp/nanogpt-level0-gpt2/data/val.bin
/tmp/nanogpt-level0-gpt2/data/test.bin
/tmp/nanogpt-level0-gpt2/data/meta.json
```

## Run one AdamW seed

```bash
./scripts/run_one.sh adamw 1337 \
  2>&1 | tee /tmp/level0-gpt2-adamw-seed1337.log
```

The command refuses to overwrite an existing nonempty run. To intentionally replace it:

```bash
NANOGPT_LEVEL0_OVERWRITE=1 ./scripts/run_one.sh adamw 1337
```

The run writes periodic metrics and checkpoints, validation-selected and final test metrics, WeightWatcher CSV files, and `run_complete.json` under:

```text
/tmp/nanogpt-level0-gpt2/results/adamw_seed_1337
```

## Run several AdamW seeds

```bash
NANOGPT_LEVEL0_SEEDS=1337,2027,4099 \
NANOGPT_LEVEL0_OPTIMIZERS=adamw \
  ./scripts/run_multiseed.sh
```

Muon remains available for a later baseline comparison:

```bash
NANOGPT_LEVEL0_OPTIMIZERS=adamw,muon ./scripts/run_multiseed.sh
```

## Analyze one seed

```bash
export NANOGPT_LEVEL0_RESULTS_ROOT=/tmp/nanogpt-level0-gpt2/results
export NANOGPT_LEVEL0_NOTEBOOK_OPTIMIZER=adamw
export NANOGPT_LEVEL0_NOTEBOOK_SEED=1337
jupyter lab notebooks/01_single_seed.ipynb
```

The notebook plots train/validation loss, perplexity, exact next-GPT-2-token accuracy, generalization gap, learning rate, final versus validation-selected test metrics, and WeightWatcher alpha by transformer matrix.

## Bounded smoke run

The smoke run checks the software path only; it is not a scientific result:

```bash
NANOGPT_LEVEL0_MAX_STEPS=2 \
NANOGPT_LEVEL0_BATCH_SIZE=1 \
NANOGPT_LEVEL0_GRAD_ACCUM_STEPS=1 \
NANOGPT_LEVEL0_EVAL_INTERVAL=1 \
NANOGPT_LEVEL0_WEIGHTWATCHER=0 \
NANOGPT_LEVEL0_OVERWRITE=1 \
  ./scripts/run_one.sh adamw 1337
```

## Primary outcomes

Use held-out cross-entropy and perplexity as the principal language-model outcomes. Exact token accuracy is reported as a secondary diagnostic and should not be compared numerically with the old 256-byte-vocabulary accuracy.
