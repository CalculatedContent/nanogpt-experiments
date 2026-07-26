#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then echo "usage: $0 RESULTS_ROOT [ANALYSIS_PLAN]" >&2; exit 2; fi
wwgpt analyze-results "$1" --analysis-plan "${2:-configs/analysis_plan_exploratory.yaml}"
