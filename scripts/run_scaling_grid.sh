#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then echo "usage: $0 MAX_LEVEL DATA_ROOT RESULTS_ROOT [TOKEN_MULTIPLIER]" >&2; exit 2; fi
MAX_LEVEL="$1"
DATA_ROOT="$2"
RESULTS_ROOT="$3"
TOKEN_MULTIPLIER="${4:-20}"
for ((LEVEL=0; LEVEL<=MAX_LEVEL; LEVEL++)); do
  wwgpt prepare-data --profile scaling --level "$LEVEL" --data-root "$DATA_ROOT" --token-multiplier "$TOKEN_MULTIPLIER"
  wwgpt run-canonical-trials --profile scaling --level "$LEVEL" --data-root "$DATA_ROOT" --results-root "$RESULTS_ROOT" --token-multiplier "$TOKEN_MULTIPLIER"
done
