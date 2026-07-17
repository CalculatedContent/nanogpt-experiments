#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "Execute this script; do not source it." >&2; return 1 2>/dev/null || exit 1; fi
set -euo pipefail
DATA_ROOT=${1:?data root required}; RESULTS_ROOT=${2:?results root required}
mkdir -p "$RESULTS_ROOT"
LOG="$RESULTS_ROOT/strength_scan_$(date +%Y%m%d-%H%M%S).log"
CMD=(wwgpt run-strength-scan --level 0 --data-root "$DATA_ROOT" --results-root "$RESULTS_ROOT" --token-multiplier 20 --seeds 1337 --strengths 0.02,0.1,0.25,0.5,1.0 --device mps --eval-interval 25 --spectral-interval 100 --checkpoint-interval 500 --immediate-projection-spectral --resume)
if command -v caffeinate >/dev/null 2>&1; then caffeinate -dimsu "${CMD[@]}" 2>&1 | tee "$LOG"; else "${CMD[@]}" 2>&1 | tee "$LOG"; fi
cat "$RESULTS_ROOT/latest_scan_root.txt"
