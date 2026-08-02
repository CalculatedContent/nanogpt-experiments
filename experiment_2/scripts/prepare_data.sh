#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "error: execute with bash; do not source" >&2
  return 2
fi
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${NANOGPT_EXPERIMENT2_DATA_ROOT:-/tmp/nanogpt-level0-bpe/data}"

export PYTHONPATH="$REPO_ROOT/level_0_baseline/src${PYTHONPATH:+:$PYTHONPATH}"
python -m level0_baseline.data \
  --dataset fineweb-edu \
  --output-dir "$DATA_ROOT" \
  --verbose \
  --log-interval-seconds "${NANOGPT_EXPERIMENT2_DATA_LOG_INTERVAL:-5}"
