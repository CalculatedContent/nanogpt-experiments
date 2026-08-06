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
SEEDS="${NANOGPT_LEVEL0_SEEDS:-1337,2027,4099}"
DEFAULT_PYTHON="$EXPERIMENT_ROOT/.venv-level0/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="python3"
fi
PYTHON_BIN="${NANOGPT_LEVEL0_PYTHON:-$DEFAULT_PYTHON}"
RESULTS_ROOT="${NANOGPT_LEVEL0_RESULTS_ROOT:-$ROOT/results}"
STORE_ROOT="${NANOGPT_LEVEL0_BASELINE_STORE:-$ROOT/baseline_reference}"

export PYTHONPATH="$EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$PYTHON_BIN" -m level0_baseline.runner \
  --config "$EXPERIMENT_ROOT/configs/level0.yaml" \
  --data-root "${NANOGPT_LEVEL0_DATA_ROOT:-$ROOT/data}" \
  --results-root "$RESULTS_ROOT" \
  --optimizers "$OPTIMIZER" \
  --seeds "$SEEDS" \
  --device "${NANOGPT_LEVEL0_DEVICE:-auto}"

"$PYTHON_BIN" -m level0_baseline.baseline_store \
  --results-root "$RESULTS_ROOT" \
  --store-root "$STORE_ROOT" \
  --optimizers "$OPTIMIZER" \
  --seeds "$SEEDS"
