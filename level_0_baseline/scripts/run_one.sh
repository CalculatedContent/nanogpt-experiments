#!/usr/bin/env bash
set -euo pipefail

ROOT="${NANOGPT_LEVEL0_ROOT:-/tmp/nanogpt-level0-bpe}"
OPTIMIZER="${1:-adamw}"
SEED="${2:-1337}"
DEVICE="${NANOGPT_LEVEL0_DEVICE:-auto}"
OVERWRITE="${NANOGPT_LEVEL0_OVERWRITE:-0}"

ARGS=(
  --config configs/level0.yaml
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --device "$DEVICE"
  --data-root "${NANOGPT_LEVEL0_DATA_ROOT:-$ROOT/data}"
  --results-root "${NANOGPT_LEVEL0_RESULTS_ROOT:-$ROOT/results}"
)
if [ "$OVERWRITE" = "1" ]; then
  ARGS+=(--overwrite)
fi

python -m level0_baseline.train "${ARGS[@]}"
