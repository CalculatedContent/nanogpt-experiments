#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then echo "usage: $0 LEVEL DATA_ROOT RESULTS_ROOT TOKEN_MULTIPLIER [CONFIG]" >&2; exit 2; fi
LEVEL="$1"; DATA_ROOT="$2"; RESULTS_ROOT="$3"; TOKEN_MULTIPLIER="$4"; CONFIG="${5:-configs/default.yaml}"
echo "[run_five_seeds] starting level=${LEVEL} data_root=${DATA_ROOT} results_root=${RESULTS_ROOT} token_multiplier=${TOKEN_MULTIPLIER}" >&2
wwgpt run-canonical-trials --config "$CONFIG" --level "$LEVEL" --data-root "$DATA_ROOT" --results-root "$RESULTS_ROOT" --token-multiplier "$TOKEN_MULTIPLIER"
echo "[run_five_seeds] completed level=${LEVEL} results_root=${RESULTS_ROOT}" >&2
