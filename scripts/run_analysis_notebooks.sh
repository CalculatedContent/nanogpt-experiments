#!/usr/bin/env bash
set -euo pipefail
: "${WWGPT_RESULTS_ROOT:?WWGPT_RESULTS_ROOT is required}"
: "${WWGPT_NOTEBOOK_OUTPUT_DIR:=notebook-output}"
mkdir -p "$WWGPT_NOTEBOOK_OUTPUT_DIR"/{executed,tables,figures,logs}
args=(--results-root "$WWGPT_RESULTS_ROOT" --output-root "$WWGPT_NOTEBOOK_OUTPUT_DIR"
      --base-optimizer "${WWGPT_BASE_OPTIMIZER:-adamw}" --notebooks "${WWGPT_NOTEBOOKS:-all}")
[[ -n "${WWGPT_ANALYSIS_PLAN:-}" ]] && args+=(--analysis-plan "$WWGPT_ANALYSIS_PLAN")
[[ -n "${WWGPT_PROFILE:-}" ]] && args+=(--profile "$WWGPT_PROFILE")
[[ -n "${WWGPT_LEVEL:-}" ]] && args+=(--level "$WWGPT_LEVEL")
[[ -n "${WWGPT_TOKEN_MULTIPLIER:-}" ]] && args+=(--token-multiplier "$WWGPT_TOKEN_MULTIPLIER")
[[ "${WWGPT_NOTEBOOK_STRICT:-0}" == 1 ]] && args+=(--strict)
[[ "${WWGPT_RUN_ANALYSIS:-0}" == 1 ]] && args+=(--run-analysis)
if [[ "${WWGPT_REUSE_ANALYSIS:-1}" == 0 ]]; then args+=(--no-reuse-existing-analysis); else args+=(--reuse-existing-analysis); fi
exec wwgpt run-notebooks "${args[@]}"
