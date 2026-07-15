#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then echo "usage: $0 RESULTS_ROOT"; exit 2; fi
wwgpt analyze-results "$1"
