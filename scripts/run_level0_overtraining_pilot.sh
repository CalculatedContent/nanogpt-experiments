#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 DATA_ROOT RESULTS_ROOT [DEVICE=mps] [SEEDS=1337,2027,4099]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="$1"
RESULTS_ROOT="$2"
DEVICE="${3:-mps}"
SEEDS="${4:-1337,2027,4099}"
TOKEN_MULTIPLIER=20
BASE_CONFIG="configs/level0_adaptive_alpha.yaml"
OVERTRAINING_CONFIG="configs/level0_overtraining_pilot.yaml"
ANALYSIS_PLAN="${WWGPT_ANALYSIS_PLAN:-configs/analysis_plan_exploratory.yaml}"
LEVEL_ROOT="$RESULTS_ROOT/experiments/level_00/multiplier_${TOKEN_MULTIPLIER}"

mkdir -p "$DATA_ROOT" "$RESULTS_ROOT"

# The corpus identity remains the nominal multiplier-20 Level 0 corpus.
# Overtraining revisits this fixed corpus through random-window sampling.
wwgpt prepare-data \
  --level 0 \
  --config "$BASE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --token-multiplier "$TOKEN_MULTIPLIER"

wwgpt run-multiseed \
  --level 0 \
  --config "$OVERTRAINING_CONFIG" \
  --analysis-plan "$ANALYSIS_PLAN" \
  --data-root "$DATA_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --token-multiplier "$TOKEN_MULTIPLIER" \
  --seeds "$SEEDS" \
  --optimizer adamw \
  --extensions none,wwpgd \
  --device "$DEVICE"

wwgpt check-health --experiment-root "$LEVEL_ROOT"
wwgpt analyze-results "$LEVEL_ROOT" --analysis-plan "$ANALYSIS_PLAN"
wwgpt audit-experiment --experiment-root "$LEVEL_ROOT"
wwgpt generate-reproducibility-report \
  --experiment-root "$LEVEL_ROOT" \
  --analysis-plan "$ANALYSIS_PLAN" \
  --strict

echo "[level0-overtraining] complete results=$LEVEL_ROOT" >&2
