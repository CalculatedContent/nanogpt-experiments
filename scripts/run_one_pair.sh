#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
  echo "usage: $0 LEVEL DATA_ROOT RESULTS_ROOT TOKEN_MULTIPLIER [DEVICE=auto] [SEEDS=1337]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LEVEL="$1"
DATA_ROOT="$2"
RESULTS_ROOT="$3"
TOKEN_MULTIPLIER="$4"
DEVICE="${5:-auto}"
SEEDS="${6:-1337}"
CONFIG="configs/level${LEVEL}_adaptive_alpha.yaml"
ANALYSIS_PLAN="${WWGPT_ANALYSIS_PLAN:-configs/analysis_plan_exploratory.yaml}"
RESUME="${WWGPT_RESUME:-0}"

case "$LEVEL" in
  0|1|2) ;;
  *)
    echo "LEVEL must be 0, 1, or 2" >&2
    exit 2
    ;;
esac

if [ ! -f "$CONFIG" ]; then
  echo "missing level configuration: $CONFIG" >&2
  exit 2
fi

mkdir -p "$DATA_ROOT" "$RESULTS_ROOT"

echo "[one-pair] prepare/reuse data level=$LEVEL root=$DATA_ROOT" >&2
wwgpt prepare-data \
  --level "$LEVEL" \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --token-multiplier "$TOKEN_MULTIPLIER"

RUN_ARGS=(
  --level "$LEVEL"
  --config "$CONFIG"
  --analysis-plan "$ANALYSIS_PLAN"
  --data-root "$DATA_ROOT"
  --results-root "$RESULTS_ROOT"
  --token-multiplier "$TOKEN_MULTIPLIER"
  --seeds "$SEEDS"
  --optimizer adamw
  --extensions none,wwpgd
  --device "$DEVICE"
)
if [ "$RESUME" = "1" ]; then
  RUN_ARGS+=(--resume)
fi

echo "[one-pair] run AdamW and AdamW+WWPGD level=$LEVEL seeds=$SEEDS" >&2
wwgpt run-multiseed "${RUN_ARGS[@]}"

wwgpt check-health --experiment-root "$RESULTS_ROOT"
wwgpt analyze-results "$RESULTS_ROOT" --analysis-plan "$ANALYSIS_PLAN"
wwgpt audit-experiment --experiment-root "$RESULTS_ROOT"
wwgpt generate-reproducibility-report --experiment-root "$RESULTS_ROOT"

echo "[one-pair] complete level=$LEVEL results=$RESULTS_ROOT" >&2
