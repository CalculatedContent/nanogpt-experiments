#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then echo "usage: $0 LEVEL DATA_ROOT TOKEN_MULTIPLIER [CONFIG]" >&2; exit 2; fi
LEVEL="$1"; DATA_ROOT="$2"; TOKEN_MULTIPLIER="$3"; CONFIG="${4:-configs/default.yaml}"
echo "[download_data] starting prepare-data level=${LEVEL} data_root=${DATA_ROOT} token_multiplier=${TOKEN_MULTIPLIER}" >&2
wwgpt prepare-data --config "$CONFIG" --level "$LEVEL" --data-root "$DATA_ROOT" --token-multiplier "$TOKEN_MULTIPLIER"
echo "[download_data] prepare-data finished" >&2
