#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${NANOGPT_LEVEL0_WWPGD_SEEDS:-1337,2027,4099}"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
for seed in "${SEED_ARRAY[@]}"; do
  "$SCRIPT_DIR/run_one.sh" adamw "$seed"
done
