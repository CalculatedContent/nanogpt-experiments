# Realistic isolated Level 0 nanoGPT baseline

This subtree is intentionally independent of the repository's WW-PGD experiment framework. It provides a clean AdamW baseline on natural-language data while retaining deterministic WeightWatcher alpha measurements.

## What Level 0 means here

The default MacBook preset is designed for an Apple M2 Pro with 16 GB unified memory:

- GPT-2 BPE tokenization (`tiktoken:gpt2`), padded model vocabulary 50,304
- four transformer blocks, four attention heads, width 256
- context length 256 BPE tokens
- approximately 16.1 million trainable parameters
- AdamW with standard matrix/no-matrix weight-decay groups
- 200-step linear warmup followed by cosine decay
- peak learning rate `6e-4`, minimum learning rate `6e-5`
- weight decay `0.1`, betas `(0.9, 0.95)`, gradient clipping `1.0`
- batch size 8 with four gradient-accumulation steps
- 5,000 optimizer steps, or 40.96 million processed training tokens
- fixed train and validation probes with independent RNG streams
- test evaluation only at the final and validation-selected checkpoints
- deterministic non-randomized WeightWatcher alpha measurements every 500 steps

This replaces the obsolete one-block, width-64, raw-byte experiment. Old `/tmp/nanogpt-level0/data` byte files are rejected by the new trainer.

## Conda installation

```bash
conda activate ww_prod310
cd ~/Desktop/work/nanoGPT/nanogpt-experiments/level_0_baseline
python -m pip install -e '.[data,analysis,test]'
python -m pip check
```

## Paths

All large artifacts remain under `/tmp` by default:

```bash
export NANOGPT_LEVEL0_ROOT=/tmp/nanogpt-level0-bpe
export NANOGPT_LEVEL0_DATA_ROOT=$NANOGPT_LEVEL0_ROOT/data
export NANOGPT_LEVEL0_RESULTS_ROOT=$NANOGPT_LEVEL0_ROOT/results
export NANOGPT_LEVEL0_CACHE_ROOT=$NANOGPT_LEVEL0_ROOT/cache
```

## Prepare the pinned FineWeb-Edu corpus

The default preparation writes 20 million training tokens and one million tokens each for validation and test. Splits are fixed, atomic, and document-disjoint at boundaries.

```bash
./scripts/prepare_data.sh 2>&1 | tee /tmp/level0-bpe-prepare.log
```

Equivalent direct command:

```bash
level0-prepare-data \
  --dataset fineweb-edu \
  --output-dir /tmp/nanogpt-level0-bpe/data \
  --train-tokens 20000000 \
  --val-tokens 1000000 \
  --test-tokens 1000000 \
  --tokenizer gpt2 \
  --model-vocab-size 50304 \
  --verbose \
  --log-interval-seconds 10
```

The progress heartbeat reports dataset resolution, current split, documents, tokens, throughput, ETA, and time since the last new tokens arrived.

## Run AdamW seed 1337

```bash
./scripts/run_one.sh adamw 1337 mps \
  2>&1 | tee /tmp/level0-bpe-adamw-seed1337.log
```

The script resumes automatically when `checkpoint_latest.pt` exists. To deliberately discard a prior run:

```bash
NANOGPT_LEVEL0_OVERWRITE=1 ./scripts/run_one.sh adamw 1337 mps
```

For a bounded CPU smoke test:

```bash
NANOGPT_LEVEL0_MAX_STEPS=2 \
NANOGPT_LEVEL0_BATCH_SIZE=2 \
NANOGPT_LEVEL0_GRAD_ACCUM_STEPS=1 \
NANOGPT_LEVEL0_EVAL_INTERVAL=1 \
NANOGPT_LEVEL0_DISABLE_WEIGHTWATCHER=1 \
NANOGPT_LEVEL0_OVERWRITE=1 \
./scripts/run_one.sh adamw 1337 cpu
```

## Output contract

Each run writes:

- `manifest.json`: exact model, optimizer, data identity, fixed-probe hashes, and protocol
- `metrics.csv`: train/validation loss, perplexity, bits per token, accuracy, gap, LR, gradient norm, weight norm, and throughput
- `checkpoint_latest.pt`: resumable training state
- `checkpoint_best.pt`: validation-selected model
- `checkpoint_final.pt`: final model
- `final_metrics.json`: final-checkpoint test metrics
- `selected_checkpoint_metrics.json`: validation-selected checkpoint test metrics
- `weightwatcher_step_*.csv`: per-matrix alpha, D, xmin, and tail metadata when available
- `run_complete.json`: transactional completion marker
- `train.log`: persistent progress log

Test data is not evaluated during training. It is touched only after optimization completes, once for the final checkpoint and once for the validation-selected checkpoint.

## Plot one run

```bash
export NANOGPT_LEVEL0_RESULTS_ROOT=/tmp/nanogpt-level0-bpe/results
export NANOGPT_LEVEL0_NOTEBOOK_OPTIMIZER=adamw
export NANOGPT_LEVEL0_NOTEBOOK_SEED=1337
jupyter lab notebooks/01_single_seed.ipynb
```

The notebook plots loss, perplexity, exact next-BPE-token accuracy, bits per token, optimization diagnostics, and WeightWatcher alpha trajectories. It also saves PNG files under the run's `plots/` directory.

## Multiple seeds

```bash
NANOGPT_LEVEL0_SEEDS=1337,2027,4099 \
NANOGPT_LEVEL0_OPTIMIZERS=adamw \
NANOGPT_LEVEL0_DEVICE=mps \
./scripts/run_multiseed.sh
```

Then run `notebooks/02_multiseed.ipynb` for mean and standard-deviation bands.
