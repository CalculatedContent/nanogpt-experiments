#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
  echo "usage: $0 DATA_ROOT RESULTS_ROOT [TOKEN_MULTIPLIER=20] [DEVICE=auto] [SEEDS=1337,2027,4099]" >&2
  exit 2
fi

DATA_ROOT="$1"
RESULTS_ROOT="$2"
TOKEN_MULTIPLIER="${3:-20}"
DEVICE="${4:-auto}"
SEEDS="${5:-1337,2027,4099}"
ANALYSIS_PLAN="configs/analysis_plan_exploratory.yaml"

mkdir -p "$RESULTS_ROOT"

for LEVEL in 0 1 2; do
  CONFIG="configs/level${LEVEL}_adaptive_alpha.yaml"
  echo "[level0-2] preparing Level ${LEVEL}" >&2
  wwgpt prepare-data \
    --level "$LEVEL" \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --token-multiplier "$TOKEN_MULTIPLIER"

  echo "[level0-2] validating resolved Level ${LEVEL} execution" >&2
  wwgpt run-multiseed \
    --level "$LEVEL" \
    --config "$CONFIG" \
    --analysis-plan "$ANALYSIS_PLAN" \
    --data-root "$DATA_ROOT" \
    --results-root "$RESULTS_ROOT" \
    --token-multiplier "$TOKEN_MULTIPLIER" \
    --seeds "$SEEDS" \
    --optimizer adamw \
    --extensions none,wwpgd \
    --device "$DEVICE" \
    --dry-run > "$RESULTS_ROOT/level${LEVEL}_resolved_execution.txt"

  echo "[level0-2] running Level ${LEVEL}" >&2
  wwgpt run-multiseed \
    --level "$LEVEL" \
    --config "$CONFIG" \
    --analysis-plan "$ANALYSIS_PLAN" \
    --data-root "$DATA_ROOT" \
    --results-root "$RESULTS_ROOT" \
    --token-multiplier "$TOKEN_MULTIPLIER" \
    --seeds "$SEEDS" \
    --optimizer adamw \
    --extensions none,wwpgd \
    --device "$DEVICE"
done

python -m wwgpt.cross_level_analysis \
  --results-root "$RESULTS_ROOT" \
  --output-dir "$RESULTS_ROOT/cross_level_analysis" \
  --figures-dir "$RESULTS_ROOT/cross_level_analysis/figures"

echo "[level0-2] complete: $RESULTS_ROOT/cross_level_analysis" >&2
