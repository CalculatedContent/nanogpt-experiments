#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 DATA_ROOT RESULTS_ROOT [TOKEN_MULTIPLIER=20] [DEVICE=auto]" >&2
  exit 2
fi

DATA_ROOT="$1"
RESULTS_ROOT="$2"
TOKEN_MULTIPLIER="${3:-20}"
DEVICE="${4:-auto}"
LEVELS="${WWGPT_LEVELS:-0,1,2}"
SEEDS="${WWGPT_SEEDS:-1337,2027,4099}"
OPTIMIZERS="${WWGPT_OPTIMIZERS:-adamw}"
WWPGD_MODES="${WWGPT_WWPGD_MODES:-adaptive}"
UNIFORM_HARDNESS="${WWGPT_UNIFORM_HARDNESS:-0.25}"
LAYER_LR="${WWGPT_LAYER_LR:-flat}"
LR_SCHEDULE="${WWGPT_LR_SCHEDULE:-warmup_cosine}"
LR_SCALE_RULE="${WWGPT_LR_SCALE_RULE:-fixed}"
LR_REFERENCE_TOKENS="${WWGPT_LR_REFERENCE_TOKENS:-4096}"
CANDIDATE_DEVICE="${WWGPT_CANDIDATE_DEVICE:-auto}"
RESUME="${WWGPT_RESUME:-1}"
RUN_NOTEBOOKS="${WWGPT_RUN_NOTEBOOKS:-1}"
ANALYSIS_PLAN="${WWGPT_ANALYSIS_PLAN:-configs/analysis_plan_exploratory.yaml}"

mkdir -p "$RESULTS_ROOT"
wwgpt local-readiness \
  --device "$DEVICE" \
  --levels "$LEVELS" \
  --optimizers "$OPTIMIZERS" \
  --output "$RESULTS_ROOT/local-readiness"

IFS=',' read -r -a LEVEL_ARRAY <<< "$LEVELS"
IFS=',' read -r -a OPTIMIZER_ARRAY <<< "$OPTIMIZERS"
IFS=',' read -r -a MODE_ARRAY <<< "$WWPGD_MODES"

for MODE in "${MODE_ARRAY[@]}"; do
  if [ "$MODE" = "adaptive" ]; then
    MODE_ARGS=(--wwpgd-adaptive-mode alpha_distance)
  elif [ "$MODE" = "uniform" ]; then
    MODE_ARGS=(--wwpgd-adaptive-mode uniform --wwpgd-alpha-max-hardness "$UNIFORM_HARDNESS")
  else
    echo "unknown WWPGD mode: $MODE (expected adaptive or uniform)" >&2
    exit 2
  fi
  for OPTIMIZER in "${OPTIMIZER_ARRAY[@]}"; do
    VARIANT_ROOT="$RESULTS_ROOT/${MODE}/${OPTIMIZER}/${LAYER_LR}/${LR_SCALE_RULE}"
    mkdir -p "$VARIANT_ROOT"
    for LEVEL in "${LEVEL_ARRAY[@]}"; do
      CONFIG="configs/level${LEVEL}_adaptive_alpha.yaml"
      echo "[local-mac] prepare level=$LEVEL mode=$MODE optimizer=$OPTIMIZER" >&2
      wwgpt prepare-data \
        --level "$LEVEL" --config "$CONFIG" --data-root "$DATA_ROOT" \
        --token-multiplier "$TOKEN_MULTIPLIER"
      COMMON=(
        --level "$LEVEL" --config "$CONFIG" --analysis-plan "$ANALYSIS_PLAN"
        --data-root "$DATA_ROOT" --results-root "$VARIANT_ROOT"
        --token-multiplier "$TOKEN_MULTIPLIER" --seeds "$SEEDS"
        --optimizer "$OPTIMIZER" --extensions none,wwpgd --device "$DEVICE"
        --layer-lr "$LAYER_LR" --lr-schedule "$LR_SCHEDULE"
        --lr-scale-rule "$LR_SCALE_RULE"
        --lr-reference-tokens-per-step "$LR_REFERENCE_TOKENS"
        --wwpgd-candidate-device "$CANDIDATE_DEVICE"
        "${MODE_ARGS[@]}"
      )
      wwgpt run-multiseed "${COMMON[@]}" --dry-run \
        > "$VARIANT_ROOT/level${LEVEL}_resolved_execution.txt"
      if [ "$RESUME" = "1" ]; then
        COMMON+=(--resume)
      fi
      wwgpt run-multiseed "${COMMON[@]}"
    done
    wwgpt analyze-results "$VARIANT_ROOT" --analysis-plan "$ANALYSIS_PLAN"
    python -m wwgpt.cross_level_analysis \
      --results-root "$VARIANT_ROOT" \
      --output-dir "$VARIANT_ROOT/cross_level_analysis" \
      --figures-dir "$VARIANT_ROOT/cross_level_analysis/figures"
    if [ "$RUN_NOTEBOOKS" = "1" ]; then
      export WWGPT_RESULTS_ROOT="$VARIANT_ROOT"
      export WWGPT_NOTEBOOK_OUTPUT_DIR="$VARIANT_ROOT/notebook-analysis"
      export WWGPT_ANALYSIS_PLAN="$ANALYSIS_PLAN"
      export WWGPT_BASE_OPTIMIZER="$OPTIMIZER"
      export WWGPT_LEVEL=""
      export WWGPT_TOKEN_MULTIPLIER="$TOKEN_MULTIPLIER"
      ./scripts/run_analysis_notebooks.sh
    fi
  done
done

echo "[local-mac] complete: $RESULTS_ROOT" >&2
