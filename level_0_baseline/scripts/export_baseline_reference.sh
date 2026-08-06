#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "error: run this script with bash; do not source it" >&2
  return 2
fi
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT="${NANOGPT_LEVEL0_ROOT:-/tmp/nanogpt-level0-baselines}"
DEFAULT_PYTHON="$EXPERIMENT_ROOT/.venv-level0/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="python3"
fi
PYTHON_BIN="${NANOGPT_LEVEL0_PYTHON:-$DEFAULT_PYTHON}"

export PYTHONPATH="$EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m level0_baseline.baseline_store \
  --results-root "${NANOGPT_LEVEL0_RESULTS_ROOT:-$ROOT/results}" \
  --store-root "${NANOGPT_LEVEL0_BASELINE_STORE:-$ROOT/baseline_reference}" \
  --optimizers "${NANOGPT_LEVEL0_OPTIMIZERS:-sgd_momentum,adamw,muon}" \
  --seeds "${NANOGPT_LEVEL0_SEEDS:-1337,2027,4099}"
