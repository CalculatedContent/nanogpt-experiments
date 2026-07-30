#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    "error: execute this script with 'bash ${BASH_SOURCE[0]}'; do not source it" \
    >&2
  return 2
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${NANOGPT_LEVEL0_SEEDS:-1337,2027,4099}"
OPTIMIZERS="${NANOGPT_LEVEL0_OPTIMIZERS:-adamw}"

IFS=',' read -r -a OPTIMIZER_ARRAY <<< "$OPTIMIZERS"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"

for optimizer in "${OPTIMIZER_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    bash "$SCRIPT_DIR/run_one.sh" "$optimizer" "$seed"
  done
done
