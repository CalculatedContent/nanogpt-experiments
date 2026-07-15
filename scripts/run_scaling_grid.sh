#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "usage: $0 MAX_LEVEL DATA_ROOT RESULTS_ROOT"; exit 2; fi
echo "Creating scaling-grid smoke-style demonstration under $3"
wwgpt run-multiseed "$3"
