#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    "error: execute this script with 'bash ${BASH_SOURCE[0]}'; do not source it" \
    >&2
  return 2
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

ROOT="${NANOGPT_LEVEL0_ROOT:-/tmp/nanogpt-level0-bpe}"
OPTIMIZER="${1:-adamw}"
SEED="${2:-1337}"
DEVICE="${NANOGPT_LEVEL0_DEVICE:-auto}"
OVERWRITE="${NANOGPT_LEVEL0_OVERWRITE:-0}"

export PYTHONPATH="$EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --config "$EXPERIMENT_ROOT/configs/level0.yaml"
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --device "$DEVICE"
  --data-root "${NANOGPT_LEVEL0_DATA_ROOT:-$ROOT/data}"
  --results-root "${NANOGPT_LEVEL0_RESULTS_ROOT:-$ROOT/results}"
)
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python -m level0_baseline.train "${ARGS[@]}"
