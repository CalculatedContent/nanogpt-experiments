#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    "error: execute this script with 'bash ${BASH_SOURCE[0]}'; do not source it" \
    >&2
  return 2
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${NANOGPT_LEVEL0_WWPGD_SEEDS:-1337,2027,4099}"

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
for seed in "${SEED_ARRAY[@]}"; do
  bash "$SCRIPT_DIR/run_one.sh" adamw "$seed"
done
