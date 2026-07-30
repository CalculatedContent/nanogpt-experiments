#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "error: execute with bash; do not source" >&2
  return 2
fi
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$EXPERIMENT_ROOT/.." && pwd)"
ROOT="${NANOGPT_LEVEL1_ROOT:-/tmp/nanogpt-level1}"
OPTIMIZER="${1:-adamw}"
SEED="${2:-1337}"
DEVICE="${NANOGPT_LEVEL1_DEVICE:-mps}"
OVERWRITE="${NANOGPT_LEVEL1_OVERWRITE:-0}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/level_0_baseline/src${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  --config "$EXPERIMENT_ROOT/configs/level1.yaml"
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --device "$DEVICE"
  --data-root "${NANOGPT_LEVEL1_DATA_ROOT:-/tmp/nanogpt-level0-bpe/data}"
  --results-root "${NANOGPT_LEVEL1_RESULTS_ROOT:-$ROOT/results}"
)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)
python -m level0_baseline.train "${ARGS[@]}"
