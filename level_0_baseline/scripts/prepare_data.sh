#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "error: run this script with bash; do not source it" >&2
  return 2
fi
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT="${NANOGPT_LEVEL0_ROOT:-/tmp/nanogpt-level0-baselines}"
DATA_ROOT="${NANOGPT_LEVEL0_DATA_ROOT:-$ROOT/data}"
PYTHON_BIN="${NANOGPT_LEVEL0_PYTHON:-python}"

if [[ -f "$DATA_ROOT/meta.json" && -f "$DATA_ROOT/train.bin" && \
      -f "$DATA_ROOT/val.bin" && -f "$DATA_ROOT/test.bin" ]]; then
  printf 'Level Zero data already exist at %s\n' "$DATA_ROOT"
  exit 0
fi

export PYTHONPATH="$EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m level0_baseline.data \
  --dataset fineweb-edu \
  --output-dir "$DATA_ROOT" \
  --verbose \
  --log-interval-seconds 5
