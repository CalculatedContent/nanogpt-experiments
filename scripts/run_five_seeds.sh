#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then echo "usage: $0 LEVEL DATA_ROOT RESULTS_ROOT TOKEN_MULTIPLIER [CONFIG] [ANALYSIS_PLAN]" >&2; exit 2; fi
LEVEL="$1"; DATA_ROOT="$2"; RESULTS_ROOT="$3"; TOKEN_MULTIPLIER="$4"
CONFIG="${5:-configs/default.yaml}"; PLAN="${6:-configs/analysis_plan_exploratory.yaml}"
wwgpt run-canonical-trials --config "$CONFIG" --analysis-plan "$PLAN" --level "$LEVEL" --data-root "$DATA_ROOT" --results-root "$RESULTS_ROOT" --token-multiplier "$TOKEN_MULTIPLIER"
