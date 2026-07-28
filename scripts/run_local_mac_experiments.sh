#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 DATA_ROOT RESULTS_ROOT [TOKEN_MULTIPLIER=20] [DEVICE=auto]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="$1"
RESULTS_ROOT="$2"
TOKEN_MULTIPLIER="${3:-20}"
DEVICE="${4:-auto}"
LEVELS="${WWGPT_LEVELS:-0,1,2}"
SEEDS="${WWGPT_SEEDS:-1337}"
OPTIMIZERS="${WWGPT_OPTIMIZERS:-adamw}"
WWPGD_MODES="${WWGPT_WWPGD_MODES:-adaptive}"
UNIFORM_HARDNESS="${WWGPT_UNIFORM_HARDNESS:-0.25}"
TARGET_ALPHA="${WWGPT_TARGET_ALPHA:-2.0}"
BLEND_ETA="${WWGPT_BLEND_ETA:-0.5}"
CAYLEY_ETA="${WWGPT_CAYLEY_ETA:-0.25}"
MAX_PER_STEP_GAIN="${WWGPT_MAX_PER_STEP_GAIN:-0.02}"
MAX_ENDPOINT_FRACTION="${WWGPT_MAX_ENDPOINT_FRACTION:-0.40}"
CUMULATIVE_DOSE_CAP="${WWGPT_CUMULATIVE_DOSE_CAP:-0.025}"
PER_STEP_RELATIVE_CAP="${WWGPT_PER_STEP_RELATIVE_CAP:-0.001}"
LAYER_LR="${WWGPT_LAYER_LR:-flat}"
LR_SCHEDULE="${WWGPT_LR_SCHEDULE:-warmup_cosine}"
LR_SCALE_RULE="${WWGPT_LR_SCALE_RULE:-fixed}"
LR_REFERENCE_TOKENS="${WWGPT_LR_REFERENCE_TOKENS:-4096}"
CANDIDATE_DEVICE="${WWGPT_CANDIDATE_DEVICE:-auto}"
RESUME="${WWGPT_RESUME:-1}"
RUN_NOTEBOOKS="${WWGPT_RUN_NOTEBOOKS:-0}"
ANALYSIS_PLAN="${WWGPT_ANALYSIS_PLAN:-configs/analysis_plan_exploratory.yaml}"

CACHE_ROOT="${WWGPT_CACHE_ROOT:-$(dirname "$DATA_ROOT")/cache}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$RESULTS_ROOT" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
wwgpt local-readiness \
  --device "$DEVICE" --levels "$LEVELS" --optimizers "$OPTIMIZERS" \
  --output "$RESULTS_ROOT/local-readiness"

IFS=',' read -r -a LEVEL_ARRAY <<< "$LEVELS"
IFS=',' read -r -a OPTIMIZER_ARRAY <<< "$OPTIMIZERS"
IFS=',' read -r -a MODE_ARRAY <<< "$WWPGD_MODES"

# Prepare/reuse each immutable data identity exactly once.
for LEVEL in "${LEVEL_ARRAY[@]}"; do
  CONFIG="configs/level${LEVEL}_adaptive_alpha.yaml"
  echo "[local-mac] ensure data level=$LEVEL" >&2
  wwgpt prepare-data \
    --level "$LEVEL" --config "$CONFIG" --data-root "$DATA_ROOT" \
    --token-multiplier "$TOKEN_MULTIPLIER"
done

for MODE in "${MODE_ARRAY[@]}"; do
  if [ "$MODE" = "adaptive" ]; then
    MODE_ARGS=(--wwpgd-adaptive-mode alpha_distance)
  elif [ "$MODE" = "uniform" ]; then
    MODE_ARGS=(
      --wwpgd-adaptive-mode uniform
      --wwpgd-alpha-max-hardness "$UNIFORM_HARDNESS"
    )
  else
    echo "unknown WWPGD mode: $MODE (expected adaptive or uniform)" >&2
    exit 2
  fi
  for OPTIMIZER in "${OPTIMIZER_ARRAY[@]}"; do
    VARIANT_ROOT="$RESULTS_ROOT/${MODE}/${OPTIMIZER}/${LAYER_LR}/${LR_SCALE_RULE}"
    mkdir -p "$VARIANT_ROOT"
    for LEVEL in "${LEVEL_ARRAY[@]}"; do
      CONFIG="configs/level${LEVEL}_adaptive_alpha.yaml"
      COMMON=(
        --level "$LEVEL" --config "$CONFIG" --analysis-plan "$ANALYSIS_PLAN"
        --data-root "$DATA_ROOT" --results-root "$VARIANT_ROOT"
        --token-multiplier "$TOKEN_MULTIPLIER" --seeds "$SEEDS"
        --optimizer "$OPTIMIZER" --extensions none,wwpgd --device "$DEVICE"
        --layer-lr "$LAYER_LR" --lr-schedule "$LR_SCHEDULE"
        --lr-scale-rule "$LR_SCALE_RULE"
        --lr-reference-tokens-per-step "$LR_REFERENCE_TOKENS"
        --target-alpha "$TARGET_ALPHA"
        --wwpgd-blend-eta "$BLEND_ETA"
        --wwpgd-cayley-eta "$CAYLEY_ETA"
        --wwpgd-max-per-step-gain "$MAX_PER_STEP_GAIN"
        --wwpgd-max-endpoint-fraction-per-refresh "$MAX_ENDPOINT_FRACTION"
        --wwpgd-max-cumulative-relative-change-per-refresh "$CUMULATIVE_DOSE_CAP"
        --wwpgd-per-step-max-relative-change "$PER_STEP_RELATIVE_CAP"
        --wwpgd-candidate-device "$CANDIDATE_DEVICE"
        "${MODE_ARGS[@]}"
      )
      wwgpt run-multiseed "${COMMON[@]}" --dry-run \
        > "$VARIANT_ROOT/level${LEVEL}_resolved_execution.txt"
      if [ "$RESUME" = "1" ]; then
        COMMON+=(--resume)
      fi
      wwgpt run-multiseed "${COMMON[@]}"
      LEVEL_LAYOUT="$VARIANT_ROOT/experiments/level_$(printf '%02d' "$LEVEL")/multiplier_$TOKEN_MULTIPLIER"
      wwgpt check-health --experiment-root "$LEVEL_LAYOUT"
      wwgpt analyze-results "$LEVEL_LAYOUT" --analysis-plan "$ANALYSIS_PLAN"
      wwgpt audit-experiment --experiment-root "$LEVEL_LAYOUT"
      wwgpt generate-reproducibility-report \
        --experiment-root "$LEVEL_LAYOUT" \
        --analysis-plan "$ANALYSIS_PLAN" \
        --strict
    done
    # Validate the combined Level 0--2 root as one experiment as well. This
    # catches cross-level discovery, deduplication, and post-processing defects
    # that cannot be exposed by validating each level in isolation.
    wwgpt check-health --experiment-root "$VARIANT_ROOT"
    wwgpt analyze-results "$VARIANT_ROOT" --analysis-plan "$ANALYSIS_PLAN"
    wwgpt audit-experiment --experiment-root "$VARIANT_ROOT"
    wwgpt generate-reproducibility-report \
      --experiment-root "$VARIANT_ROOT" \
      --analysis-plan "$ANALYSIS_PLAN" \
      --strict
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
      export WWGPT_NOTEBOOK_STRICT=1
      ./scripts/run_analysis_notebooks.sh
    fi
  done
done

echo "[local-mac] complete: $RESULTS_ROOT" >&2
