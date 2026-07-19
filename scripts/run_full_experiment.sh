#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then echo "usage: $0 LEVEL DATA_ROOT RESULTS_ROOT TOKEN_MULTIPLIER [CONFIG]" >&2; exit 2; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ARGS=()
if [ "$#" -eq 5 ]; then CONFIG_ARGS+=("$5"); fi
"$SCRIPT_DIR/download_data.sh" "$1" "$2" "$4" "${CONFIG_ARGS[@]}"
"$SCRIPT_DIR/run_five_seeds.sh" "$1" "$2" "$3" "$4" "${CONFIG_ARGS[@]}"
"$SCRIPT_DIR/analyze_five_seeds.sh" "$3"
