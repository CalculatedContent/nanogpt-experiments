#!/usr/bin/env bash
set -euo pipefail
ROOT="${NANOGPT_LEVEL0_ROOT:-/tmp/nanogpt-level0}"
OPTIMIZER="${1:-adamw}"
SEED="${2:-1337}"
python -m level0_baseline.train --config configs/level0.yaml --optimizer "$OPTIMIZER" --seed "$SEED" --data-root "${NANOGPT_LEVEL0_DATA_ROOT:-$ROOT/data}" --results-root "${NANOGPT_LEVEL0_RESULTS_ROOT:-$ROOT/results}"
