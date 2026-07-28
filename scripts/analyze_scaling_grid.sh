#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 RESULTS_ROOT [ANALYSIS_PLAN]" >&2
  exit 2
fi

RESULTS_ROOT="$1"
ANALYSIS_PLAN="${2:-configs/analysis_plan_exploratory.yaml}"
FOUND=0

for LAYOUT in "$RESULTS_ROOT"/experiments/level_*/multiplier_*; do
  [ -d "$LAYOUT" ] || continue
  if compgen -G "$LAYOUT/pair_*" > /dev/null || compgen -G "$LAYOUT/trial_*" > /dev/null; then
    FOUND=1
    wwgpt check-health --experiment-root "$LAYOUT"
    wwgpt analyze-results "$LAYOUT" --analysis-plan "$ANALYSIS_PLAN"
    wwgpt audit-experiment --experiment-root "$LAYOUT"
    wwgpt generate-reproducibility-report \
      --experiment-root "$LAYOUT" \
      --analysis-plan "$ANALYSIS_PLAN" \
      --strict
  fi
done

if [ "$FOUND" -ne 1 ]; then
  echo "no level/multiplier experiment layouts found under $RESULTS_ROOT" >&2
  exit 1
fi

python -m wwgpt.cross_level_analysis \
  --results-root "$RESULTS_ROOT" \
  --output-dir "$RESULTS_ROOT/cross_level_analysis" \
  --figures-dir "$RESULTS_ROOT/cross_level_analysis/figures"
