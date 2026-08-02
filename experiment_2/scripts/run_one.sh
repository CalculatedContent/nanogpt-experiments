#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "error: execute with bash; do not source" >&2
  return 2
fi
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$EXPERIMENT_ROOT/.." && pwd)"

SCALE="${1:-}"
ARM="${2:-}"
SEED="${3:-1337}"
PAIR_ROOT="${4:-}"
DEVICE="${NANOGPT_EXPERIMENT2_DEVICE:-mps}"
DATA_ROOT="${NANOGPT_EXPERIMENT2_DATA_ROOT:-/tmp/nanogpt-level0-bpe/data}"

case "$SCALE" in level0|level1) ;; *) echo "usage: bash run_one.sh {level0|level1} {adamw|adaptive_wwpgd} SEED PAIR_ROOT" >&2; exit 2;; esac
case "$ARM" in adamw|adaptive_wwpgd) ;; *) echo "usage: bash run_one.sh {level0|level1} {adamw|adaptive_wwpgd} SEED PAIR_ROOT" >&2; exit 2;; esac
[[ -n "$PAIR_ROOT" ]] || { echo "PAIR_ROOT is required" >&2; exit 2; }
[[ -f "$DATA_ROOT/meta.json" ]] || { echo "missing prepared BPE data: $DATA_ROOT/meta.json" >&2; exit 1; }

CONFIG="$EXPERIMENT_ROOT/configs/${SCALE}.yaml"
if [[ "$ARM" == "adamw" ]]; then
  RESULTS_ROOT="$PAIR_ROOT/baseline/results"
  BASELINE_ARGS=()
else
  RESULTS_ROOT="$PAIR_ROOT/adaptive/results"
  BASELINE_RUN="$PAIR_ROOT/baseline/results/adamw_seed_${SEED}"
  [[ -f "$BASELINE_RUN/run_complete.json" ]] || { echo "matching baseline must complete first: $BASELINE_RUN" >&2; exit 1; }
  BASELINE_ARGS=(--baseline-run "$BASELINE_RUN")
fi
mkdir -p "$RESULTS_ROOT"

export PYTHONPATH="$REPO_ROOT/src:$EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m experiment_2.train \
  --config "$CONFIG" \
  --arm "$ARM" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --data-root "$DATA_ROOT" \
  --results-root "$RESULTS_ROOT" \
  "${BASELINE_ARGS[@]}"
