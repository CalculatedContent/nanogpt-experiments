#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
BASELINE_ROOT="${NANOGPT_LEVEL0_ROOT:-/tmp/nanogpt-level0-bpe}"
DATA_ROOT="${NANOGPT_LEVEL0_WWPGD_DATA_ROOT:-${NANOGPT_LEVEL0_DATA_ROOT:-$BASELINE_ROOT/data}}"
export NANOGPT_LEVEL0_WWPGD_DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="$REPO_ROOT/level_0_wwpgd/src${PYTHONPATH:+:$PYTHONPATH}"

python -m level0_wwpgd.data \
  --dataset fineweb-edu \
  --output-dir "$DATA_ROOT" \
  --verbose \
  --log-interval-seconds "${NANOGPT_LEVEL0_WWPGD_DATA_LOG_INTERVAL:-5}"
