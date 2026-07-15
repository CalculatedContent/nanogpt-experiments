#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then echo "usage: $0 RESULTS_ROOT"; exit 2; fi
echo "Creating new analysis directory under $1/analysis"
wwgpt analyze-results "$1"
