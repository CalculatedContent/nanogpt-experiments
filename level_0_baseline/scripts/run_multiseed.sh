#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SEEDS="${NANOGPT_LEVEL0_SEEDS:-1337,2027,4099}"
OPTIMIZERS="${NANOGPT_LEVEL0_OPTIMIZERS:-adamw}"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=',' read -r -a OPTIMIZER_ARRAY <<< "$OPTIMIZERS"

for optimizer in "${OPTIMIZER_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    ./scripts/run_one.sh "$optimizer" "$seed"
  done
done
