#!/usr/bin/env bash
set -euo pipefail

ROOT="${NANOGPT_LEVEL0_ROOT:-/tmp/nanogpt-level0-bpe}"
DATA_ROOT="${NANOGPT_LEVEL0_DATA_ROOT:-$ROOT/data}"
CACHE_ROOT="${NANOGPT_LEVEL0_CACHE_ROOT:-$ROOT/cache}"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$DATA_ROOT" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

level0-prepare-data \
  --dataset fineweb-edu \
  --output-dir "$DATA_ROOT" \
  --train-tokens "${NANOGPT_LEVEL0_TRAIN_TOKENS:-20000000}" \
  --val-tokens "${NANOGPT_LEVEL0_VAL_TOKENS:-1000000}" \
  --test-tokens "${NANOGPT_LEVEL0_TEST_TOKENS:-1000000}" \
  --tokenizer gpt2 \
  --model-vocab-size 50304 \
  --verbose \
  --log-interval-seconds "${NANOGPT_LEVEL0_DATA_LOG_INTERVAL:-10}"
