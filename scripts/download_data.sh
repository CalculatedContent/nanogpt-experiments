#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "usage: $0 LEVEL DATA_ROOT TOKEN_MULTIPLIER" >&2; exit 2; fi
LEVEL="$1"; DATA_ROOT="$2"; TOKEN_MULTIPLIER="$3"
echo "[download_data] starting prepare-data level=${LEVEL} data_root=${DATA_ROOT} token_multiplier=${TOKEN_MULTIPLIER}" >&2
wwgpt prepare-data --level "$LEVEL" --data-root "$DATA_ROOT" --token-multiplier "$TOKEN_MULTIPLIER"
echo "[download_data] prepare-data finished" >&2
