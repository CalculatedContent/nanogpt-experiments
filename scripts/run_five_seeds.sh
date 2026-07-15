#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then echo "usage: $0 LEVEL DATA_ROOT RESULTS_ROOT TOKEN_MULTIPLIER" >&2; exit 2; fi
LEVEL="$1"; DATA_ROOT="$2"; RESULTS_ROOT="$3"; TOKEN_MULTIPLIER="$4"
wwgpt run-multiseed --level "$LEVEL" --data-root "$DATA_ROOT" --results-root "$RESULTS_ROOT" --token-multiplier "$TOKEN_MULTIPLIER"
