#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ROOT="${NANOGPT_LEVEL0_ROOT:-/tmp/nanogpt-level0-gpt2}"
OPTIMIZER="${1:-adamw}"
SEED="${2:-1337}"
DEVICE="${NANOGPT_LEVEL0_DEVICE:-auto}"
DATA_ROOT="${NANOGPT_LEVEL0_DATA_ROOT:-$ROOT/data}"
RESULTS_ROOT="${NANOGPT_LEVEL0_RESULTS_ROOT:-$ROOT/results}"

ARGS=(
  --config configs/level0.yaml
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --device "$DEVICE"
  --data-root "$DATA_ROOT"
  --results-root "$RESULTS_ROOT"
)
if [[ "${NANOGPT_LEVEL0_OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

export PYTHONUNBUFFERED=1
python -u -m level0_baseline.train "${ARGS[@]}"
