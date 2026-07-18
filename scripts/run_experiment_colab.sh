#!/usr/bin/env bash
set -euo pipefail
LEVEL=${LEVEL:-0}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to prepared data root}
RESULTS_ROOT=${RESULTS_ROOT:-/content/wwgpt_results}
TOKEN_MULTIPLIER=${TOKEN_MULTIPLIER:-20}
SEEDS=${SEEDS:-1337}
CONFIG_ARGS=()
if [[ -n "${CONFIG:-}" ]]; then CONFIG_ARGS+=(--config "$CONFIG"); fi
python -m pip install -e .
wwgpt device-preflight --device auto
wwgpt run-multiseed --level "$LEVEL" --data-root "$DATA_ROOT" --results-root "$RESULTS_ROOT" --token-multiplier "$TOKEN_MULTIPLIER" --seeds "$SEEDS" --device auto "${CONFIG_ARGS[@]}"
