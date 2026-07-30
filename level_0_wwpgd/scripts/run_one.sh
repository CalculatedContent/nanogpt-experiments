#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"

ROOT="${NANOGPT_LEVEL0_WWPGD_ROOT:-/tmp/nanogpt-level0-wwpgd}"
BASELINE_ROOT="${NANOGPT_LEVEL0_ROOT:-/tmp/nanogpt-level0-bpe}"
OPTIMIZER="${1:-adamw}"
SEED="${2:-1337}"
DEVICE="${NANOGPT_LEVEL0_WWPGD_DEVICE:-mps}"
OVERWRITE="${NANOGPT_LEVEL0_WWPGD_OVERWRITE:-0}"
DATA_ROOT="${NANOGPT_LEVEL0_WWPGD_DATA_ROOT:-${NANOGPT_LEVEL0_DATA_ROOT:-$BASELINE_ROOT/data}}"
RESULTS_ROOT="${NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT:-$ROOT/results}"

if [ "$OPTIMIZER" != "adamw" ]; then
  echo "isolated WWPGD experiment requires optimizer=adamw" >&2
  exit 2
fi
if [ ! -f "$DATA_ROOT/meta.json" ]; then
  echo "missing prepared BPE data: $DATA_ROOT/meta.json" >&2
  echo "prepare it from level_0_baseline or run ./scripts/prepare_data.sh" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/level_0_baseline/src:$EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --config "$EXPERIMENT_ROOT/configs/level0.yaml"
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --device "$DEVICE"
  --data-root "$DATA_ROOT"
  --results-root "$RESULTS_ROOT"
)
if [ "$OVERWRITE" = "1" ]; then
  ARGS+=(--overwrite)
fi

python -m level0_wwpgd.train "${ARGS[@]}"
