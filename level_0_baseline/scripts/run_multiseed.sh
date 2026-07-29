#!/usr/bin/env bash
set -euo pipefail
SEEDS="${NANOGPT_LEVEL0_SEEDS:-1337,2027,4099}"
for optimizer in adamw muon; do
  IFS=',' read -ra xs <<< "$SEEDS"
  for seed in "${xs[@]}"; do
    ./scripts/run_one.sh "$optimizer" "$seed"
  done
done
