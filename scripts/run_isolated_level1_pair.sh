#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "error: execute with bash; do not source" >&2
  return 2
fi
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PHASE="${1:-}"
PAIR_ROOT="${2:-}"
# Three matched seeds are the compute-budgeted Level 1 default. Override with five for confirmation.
SEEDS="${NANOGPT_LEVEL1_PAIRED_SEEDS:-1337,2027,4099}"
DATA_ROOT="${NANOGPT_LEVEL1_DATA_ROOT:-/tmp/nanogpt-level0-bpe/data}"
STATE_FILE="${NANOGPT_LEVEL1_PAIR_STATE_FILE:-/tmp/nanogpt-level1-current-pair}"
DEVICE="${NANOGPT_LEVEL1_DEVICE:-mps}"
case "$PHASE" in baseline|wwpgd|all|verify) ;; *) echo "usage: bash scripts/run_isolated_level1_pair.sh {baseline|wwpgd|all|verify} [PAIR_ROOT]" >&2; exit 2;; esac
if [[ -z "$PAIR_ROOT" ]]; then
  if [[ "$PHASE" == baseline || "$PHASE" == all ]]; then
    PAIR_ROOT="/tmp/nanogpt-level1-paired-$(date +%Y%m%d-%H%M%S)"
  elif [[ -f "$STATE_FILE" ]]; then PAIR_ROOT="$(<"$STATE_FILE")"
  else echo "error: no Level 1 pair recorded; provide PAIR_ROOT" >&2; exit 2; fi
fi
mkdir -p "$PAIR_ROOT/logs" "$PAIR_ROOT/baseline/results" "$PAIR_ROOT/wwpgd/results"
printf '%s\n' "$PAIR_ROOT" > "$STATE_FILE"
[[ -f "$DATA_ROOT/meta.json" ]] || { echo "error: missing shared BPE data at $DATA_ROOT" >&2; exit 1; }
python - "$REPO_ROOT" <<'PY'
import sys, yaml
from pathlib import Path
repo=Path(sys.argv[1])
b=yaml.safe_load((repo/'level_1_baseline/configs/level1.yaml').read_text())
w=yaml.safe_load((repo/'level_1_wwpgd/configs/level1.yaml').read_text())
for section in ('model','training','analysis'):
    if b[section] != w[section]: raise SystemExit(f'config mismatch: {section}')
expected={'n_layer':8,'n_head':8,'n_embd':512,'block_size':512,'vocab_size':50257}
for k,v in expected.items():
    if int(b['model'][k]) != v: raise SystemExit(f'bad Level 1 model.{k}: expected {v}')
if int(b['training']['max_steps']) != 10000: raise SystemExit('Level 1 max_steps must be 10000')
if w['wwpgd']['apply_mode'] != 'event_projection' or int(w['wwpgd']['interval']) != 1 or float(w['wwpgd']['target_alpha']) != 2.0:
    raise SystemExit('bad Level 1 WWPGD protocol')
print('PASS: Level 1 protocol verified (8L/8H/512d, context 512, 10k steps).')
PY
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
run_baseline(){ for seed in "${SEED_ARRAY[@]}"; do run="$PAIR_ROOT/baseline/results/adamw_seed_${seed}"; [[ -f "$run/run_complete.json" ]] && { echo "baseline seed=$seed complete; skipping"; continue; }; [[ ! -e "$run" ]] || { echo "partial run exists: $run" >&2; exit 1; }; NANOGPT_LEVEL1_DATA_ROOT="$DATA_ROOT" NANOGPT_LEVEL1_RESULTS_ROOT="$PAIR_ROOT/baseline/results" NANOGPT_LEVEL1_DEVICE="$DEVICE" bash "$REPO_ROOT/level_1_baseline/scripts/run_one.sh" adamw "$seed" 2>&1 | tee "$PAIR_ROOT/logs/baseline-seed${seed}.log"; done; }
run_wwpgd(){ for seed in "${SEED_ARRAY[@]}"; do run="$PAIR_ROOT/wwpgd/results/adamw_wwpgd_seed_${seed}"; [[ -f "$run/run_complete.json" ]] && { echo "WWPGD seed=$seed complete; skipping"; continue; }; [[ ! -e "$run" ]] || { echo "partial run exists: $run" >&2; exit 1; }; NANOGPT_LEVEL1_DATA_ROOT="$DATA_ROOT" NANOGPT_LEVEL1_WWPGD_DATA_ROOT="$DATA_ROOT" NANOGPT_LEVEL1_WWPGD_RESULTS_ROOT="$PAIR_ROOT/wwpgd/results" NANOGPT_LEVEL1_WWPGD_DEVICE="$DEVICE" bash "$REPO_ROOT/level_1_wwpgd/scripts/run_one.sh" adamw "$seed" 2>&1 | tee "$PAIR_ROOT/logs/wwpgd-seed${seed}.log"; done; }
[[ "$PHASE" == baseline || "$PHASE" == all ]] && run_baseline
[[ "$PHASE" == wwpgd || "$PHASE" == all ]] && run_wwpgd
python - "$PAIR_ROOT" "$SEEDS" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); seeds=[int(x) for x in sys.argv[2].split(',') if x]
def done(path):
    return path.is_file() and json.loads(path.read_text()).get('completed') is True
b=[s for s in seeds if done(root/f'baseline/results/adamw_seed_{s}/run_complete.json')]
w=[s for s in seeds if done(root/f'wwpgd/results/adamw_wwpgd_seed_{s}/run_complete.json')]
print(f'Pair root: {root}')
print(f'Baseline complete ({len(b)}/{len(seeds)}): {b}')
print(f'WWPGD complete ({len(w)}/{len(seeds)}): {w}')
PY
cat <<EOF
Analyze with:
PAIR_ROOT="$(cat "$STATE_FILE")"
env NANOGPT_LEVEL1_BASELINE_RESULTS_ROOT="$PAIR_ROOT/baseline/results" \\
    NANOGPT_LEVEL1_WWPGD_RESULTS_ROOT="$PAIR_ROOT/wwpgd/results" \\
    jupyter lab level_1_wwpgd/notebooks/02_compare_multiseed.ipynb
EOF
