#!/usr/bin/env bash
set -euo pipefail

ROOT="${NANOGPT_LEVEL0_ROOT:-/tmp/nanogpt-level0-bpe}"
OPTIMIZER="${1:-adamw}"
SEED="${2:-1337}"
DEVICE="${3:-auto}"
DATA_ROOT="${NANOGPT_LEVEL0_DATA_ROOT:-$ROOT/data}"
RESULTS_ROOT="${NANOGPT_LEVEL0_RESULTS_ROOT:-$ROOT/results}"
CONFIG="${NANOGPT_LEVEL0_CONFIG:-configs/level0.yaml}"

ARGS=(
  --config "$CONFIG"
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --device "$DEVICE"
  --data-root "$DATA_ROOT"
  --results-root "$RESULTS_ROOT"
)

RUN_DIR="$RESULTS_ROOT/${OPTIMIZER}_seed_${SEED}"
if [[ "${NANOGPT_LEVEL0_OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
elif [[ -f "$RUN_DIR/checkpoint_latest.pt" || -f "$RUN_DIR/run_complete.json" ]]; then
  ARGS+=(--resume)
fi
if [[ "${NANOGPT_LEVEL0_DISABLE_WEIGHTWATCHER:-0}" == "1" ]]; then
  ARGS+=(--no-weightwatcher)
fi

python -m level0_baseline.train "${ARGS[@]}"
