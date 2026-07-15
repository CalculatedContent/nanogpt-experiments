#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then echo "usage: $0 LEVEL DATA_ROOT RESULTS_ROOT TOKEN_MULTIPLIER"; exit 2; fi
./scripts/download_data.sh "$1" "$2" "$4"
./scripts/run_five_seeds.sh "$1" "$2" "$3" "$4"
./scripts/analyze_five_seeds.sh "$3"
