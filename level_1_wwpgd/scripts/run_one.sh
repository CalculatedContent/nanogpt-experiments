#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "error: execute with bash; do not source" >&2
  return 2
fi
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$EXPERIMENT_ROOT/.." && pwd)"
ROOT="${NANOGPT_LEVEL1_WWPGD_ROOT:-/tmp/nanogpt-level1-wwpgd}"
OPTIMIZER="${1:-adamw}"
SEED="${2:-1337}"
DEVICE="${NANOGPT_LEVEL1_WWPGD_DEVICE:-mps}"
OVERWRITE="${NANOGPT_LEVEL1_WWPGD_OVERWRITE:-0}"
DATA_ROOT="${NANOGPT_LEVEL1_WWPGD_DATA_ROOT:-${NANOGPT_LEVEL1_DATA_ROOT:-/tmp/nanogpt-level0-bpe/data}}"
RESULTS_ROOT="${NANOGPT_LEVEL1_WWPGD_RESULTS_ROOT:-$ROOT/results}"
[[ "$OPTIMIZER" == "adamw" ]] || { printf '%s\n' "Level 1 WWPGD requires AdamW" >&2; exit 2; }
[[ -f "$DATA_ROOT/meta.json" ]] || { printf '%s\n' "missing prepared BPE data: $DATA_ROOT/meta.json" >&2; exit 1; }
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/level_0_baseline/src:$REPO_ROOT/level_0_wwpgd/src${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  --config "$EXPERIMENT_ROOT/configs/level1.yaml"
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --device "$DEVICE"
  --data-root "$DATA_ROOT"
  --results-root "$RESULTS_ROOT"
)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)
python -m level0_wwpgd.train "${ARGS[@]}"
