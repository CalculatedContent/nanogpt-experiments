#!/usr/bin/env bash
set -euo pipefail

SEEDS="${NANOGPT_LEVEL0_SEEDS:-1337,2027,4099}"
OPTIMIZERS="${NANOGPT_LEVEL0_OPTIMIZERS:-adamw}"
DEVICE="${NANOGPT_LEVEL0_DEVICE:-auto}"

IFS=',' read -r -a optimizer_array <<< "$OPTIMIZERS"
IFS=',' read -r -a seed_array <<< "$SEEDS"
for optimizer in "${optimizer_array[@]}"; do
  for seed in "${seed_array[@]}"; do
    ./scripts/run_one.sh "$optimizer" "$seed" "$DEVICE"
  done
done
