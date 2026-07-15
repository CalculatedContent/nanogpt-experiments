#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "usage: $0 LEVEL DATA_ROOT TOKEN_MULTIPLIER"; exit 2; fi
echo "Preparing append-only data under $2"
wwgpt prepare-data --max-level "$1" --data-root "$2" --dataset fineweb_edu --token-multiplier "$3"
