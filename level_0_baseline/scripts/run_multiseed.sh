#!/usr/bin/env bash
set -euo pipefail

SEEDS="${NANOGPT_LEVEL0_SEEDS:-1337,2027,4099}"
OPTIMIZERS="${NANOGPT_LEVEL0_OPTIMIZERS:-adamw}"
IFS=',' read -ra OPTIMIZER_ARRAY <<< "$OPTIMIZERS"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
for optimizer in "${OPTIMIZER_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    ./scripts/run_one.sh "$optimizer" "$seed"
  done
done
