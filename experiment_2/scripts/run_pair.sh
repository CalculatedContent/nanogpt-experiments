#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "error: execute with bash; do not source" >&2
  return 2
fi
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

SCALE="${1:-}"
PHASE="${2:-}"
PAIR_ROOT="${3:-}"
SEEDS="${NANOGPT_EXPERIMENT2_SEEDS:-1337,2027,4099,7919,104729}"
STATE_FILE="${NANOGPT_EXPERIMENT2_STATE_FILE:-/tmp/nanogpt-experiment2-${SCALE}-current-pair}"

case "$SCALE" in level0|level1) ;; *) echo "usage: bash run_pair.sh {level0|level1} {baseline|adaptive|all|verify} [PAIR_ROOT]" >&2; exit 2;; esac
case "$PHASE" in baseline|adaptive|all|verify) ;; *) echo "usage: bash run_pair.sh {level0|level1} {baseline|adaptive|all|verify} [PAIR_ROOT]" >&2; exit 2;; esac

if [[ -z "$PAIR_ROOT" ]]; then
  if [[ "$PHASE" == baseline || "$PHASE" == all ]]; then
    PAIR_ROOT="/tmp/nanogpt-experiment2-${SCALE}-paired-$(date +%Y%m%d-%H%M%S)"
  elif [[ -f "$STATE_FILE" ]]; then
    PAIR_ROOT="$(<"$STATE_FILE")"
  else
    echo "no recorded Experiment 2 pair; provide PAIR_ROOT" >&2
    exit 2
  fi
fi

mkdir -p "$PAIR_ROOT/logs" "$PAIR_ROOT/baseline/results" "$PAIR_ROOT/adaptive/results"
printf '%s\n' "$PAIR_ROOT" > "$STATE_FILE"

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"

run_arm() {
  local arm="$1" seed run_name run_dir log
  for seed in "${SEED_ARRAY[@]}"; do
    if [[ "$arm" == adamw ]]; then
      run_name="adamw_seed_${seed}"
      run_dir="$PAIR_ROOT/baseline/results/$run_name"
      log="$PAIR_ROOT/logs/baseline-seed${seed}.log"
    else
      run_name="adamw_adaptive_wwpgd_seed_${seed}"
      run_dir="$PAIR_ROOT/adaptive/results/$run_name"
      log="$PAIR_ROOT/logs/adaptive-seed${seed}.log"
    fi
    if [[ -f "$run_dir/run_complete.json" ]]; then
      echo "[experiment2] $SCALE $arm seed=$seed complete; skipping"
      continue
    fi
    [[ ! -e "$run_dir" ]] || { echo "partial run exists: $run_dir" >&2; exit 1; }
    echo "[experiment2] starting $SCALE $arm seed=$seed"
    bash "$SCRIPT_DIR/run_one.sh" "$SCALE" "$arm" "$seed" "$PAIR_ROOT" 2>&1 | tee "$log"
  done
}

[[ "$PHASE" == baseline || "$PHASE" == all ]] && run_arm adamw
[[ "$PHASE" == adaptive || "$PHASE" == all ]] && run_arm adaptive_wwpgd

PYTHONPATH="$EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m experiment_2.status --pair-root "$PAIR_ROOT" --seeds "$SEEDS"

cat <<EOF
Experiment 2 pair recorded in:
  $STATE_FILE

Notebooks:
  jupyter lab "$EXPERIMENT_ROOT/notebooks/01_protocol_audit.ipynb"
  jupyter lab "$EXPERIMENT_ROOT/notebooks/02_compare_paired.ipynb"
  jupyter lab "$EXPERIMENT_ROOT/notebooks/03_layer_controller_diagnostics.ipynb"
EOF
