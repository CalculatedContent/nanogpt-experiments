#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then echo "usage: $0 LEVEL DATA_ROOT RESULTS_ROOT TOKEN_MULTIPLIER"; exit 2; fi
echo "Creating five-seed-compatible sequential paired runs under $3"
wwgpt run-multiseed "$3"
