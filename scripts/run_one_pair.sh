#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then echo "usage: $0 LEVEL DATA_ROOT RESULTS_ROOT TOKEN_MULTIPLIER"; exit 2; fi
echo "Creating one paired smoke-style run under $3"
wwgpt smoke-test "$3"
