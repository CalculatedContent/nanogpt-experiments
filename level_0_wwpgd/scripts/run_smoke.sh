#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    "error: execute this script with 'bash ${BASH_SOURCE[0]}'; do not source it" \
    >&2
  return 2
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NANOGPT_LEVEL0_WWPGD_MAX_STEPS=2
export NANOGPT_LEVEL0_WWPGD_BATCH_SIZE=2
export NANOGPT_LEVEL0_WWPGD_GRAD_ACCUM_STEPS=1
export NANOGPT_LEVEL0_WWPGD_EVAL_INTERVAL=1
export NANOGPT_LEVEL0_WWPGD_WW_INTERVAL=1
export NANOGPT_LEVEL0_WWPGD_PROJECTION_INTERVAL=1
export NANOGPT_LEVEL0_WWPGD_OVERWRITE=1
export NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT="${NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT:-/tmp/nanogpt-level0-wwpgd/smoke-results}"

bash "$SCRIPT_DIR/run_one.sh" adamw 1337
