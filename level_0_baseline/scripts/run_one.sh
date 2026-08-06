#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "error: run this script with bash; do not source it" >&2
  return 2
fi
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT="${NANOGPT_LEVEL0_ROOT:-/tmp/nanogpt-level0-baselines}"
OPTIMIZER="${1:-adamw}"
SEED="${2:-1337}"
DEVICE="${NANOGPT_LEVEL0_DEVICE:-auto}"
DEFAULT_PYTHON="$EXPERIMENT_ROOT/.venv-level0/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="python3"
fi
PYTHON_BIN="${NANOGPT_LEVEL0_PYTHON:-$DEFAULT_PYTHON}"

export PYTHONPATH="$EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ARGS=(
  --config "$EXPERIMENT_ROOT/configs/level0.yaml"
  --data-root "${NANOGPT_LEVEL0_DATA_ROOT:-$ROOT/data}"
  --results-root "${NANOGPT_LEVEL0_RESULTS_ROOT:-$ROOT/results}"
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --device "$DEVICE"
)

RUN_DIR="${NANOGPT_LEVEL0_RESULTS_ROOT:-$ROOT/results}/$OPTIMIZER/seed_$SEED"
if [[ "${NANOGPT_LEVEL0_OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
elif [[ -d "$RUN_DIR" ]]; then
  ARGS+=(--resume)
fi

"$PYTHON_BIN" -m level0_baseline.train "${ARGS[@]}"
