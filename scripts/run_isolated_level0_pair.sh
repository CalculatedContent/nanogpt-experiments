#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    "error: execute this script with 'bash ${BASH_SOURCE[0]}'; do not source it" \
    >&2
  return 2
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PHASE="${1:-}"
PAIR_ROOT="${2:-}"
SEEDS="${NANOGPT_LEVEL0_PAIRED_SEEDS:-1337,2027,4099,7919,104729}"
DATA_ROOT="${NANOGPT_LEVEL0_DATA_ROOT:-/tmp/nanogpt-level0-bpe/data}"
STATE_FILE="${NANOGPT_LEVEL0_PAIR_STATE_FILE:-/tmp/nanogpt-level0-current-pair}"
DEVICE="${NANOGPT_LEVEL0_DEVICE:-mps}"

usage() {
  cat >&2 <<'EOF'
usage: bash scripts/run_isolated_level0_pair.sh PHASE [PAIR_ROOT]

PHASE:
  baseline  run or continue the five-seed AdamW baseline
  wwpgd     run or continue the matching five-seed AdamW+WWPGD arm
  all       run baseline, then WWPGD
  verify    verify completed seeds and print paired paths

PAIR_ROOT defaults to a fresh timestamped /tmp directory for baseline/all.
For wwpgd/verify, an omitted PAIR_ROOT reuses the path recorded in
/tmp/nanogpt-level0-current-pair.
EOF
}

case "$PHASE" in
  baseline|wwpgd|all|verify) ;;
  *) usage; exit 2 ;;
esac

if [[ -z "$PAIR_ROOT" ]]; then
  if [[ "$PHASE" == "baseline" || "$PHASE" == "all" ]]; then
    PAIR_ROOT="/tmp/nanogpt-level0-paired-$(date +%Y%m%d-%H%M%S)"
  elif [[ -f "$STATE_FILE" ]]; then
    PAIR_ROOT="$(<"$STATE_FILE")"
  else
    printf '%s\n' \
      "error: no paired experiment is recorded; provide PAIR_ROOT explicitly" \
      >&2
    exit 2
  fi
fi

mkdir -p \
  "$PAIR_ROOT/logs" \
  "$PAIR_ROOT/baseline/results" \
  "$PAIR_ROOT/wwpgd/results"
printf '%s\n' "$PAIR_ROOT" > "$STATE_FILE"

BASELINE_RESULTS="$PAIR_ROOT/baseline/results"
WWPGD_RESULTS="$PAIR_ROOT/wwpgd/results"

if [[ ! -f "$DATA_ROOT/meta.json" ]]; then
  printf '%s\n' "Prepared BPE data is missing; preparing it under $DATA_ROOT"
  env \
    NANOGPT_LEVEL0_ROOT="$(dirname "$DATA_ROOT")" \
    NANOGPT_LEVEL0_DATA_ROOT="$DATA_ROOT" \
    NANOGPT_LEVEL0_WWPGD_DATA_ROOT="$DATA_ROOT" \
    bash "$REPO_ROOT/level_0_wwpgd/scripts/prepare_data.sh" \
    2>&1 | tee "$PAIR_ROOT/logs/prepare-data.log"
fi

python - "$REPO_ROOT" "$DATA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import yaml

repo = Path(sys.argv[1])
data_root = Path(sys.argv[2])
baseline = yaml.safe_load(
    (repo / "level_0_baseline/configs/level0.yaml").read_text()
)
wwpgd = yaml.safe_load(
    (repo / "level_0_wwpgd/configs/level0.yaml").read_text()
)
for section in ("model", "training", "analysis"):
    if baseline[section] != wwpgd[section]:
        raise SystemExit(f"baseline/WWPGD configuration mismatch in {section}")
if wwpgd["wwpgd"]["apply_mode"] != "event_projection":
    raise SystemExit("WWPGD apply_mode must be event_projection")
if int(wwpgd["wwpgd"]["interval"]) != 1:
    raise SystemExit("WWPGD interval must be one optimizer step")
if float(wwpgd["wwpgd"]["target_alpha"]) != 2.0:
    raise SystemExit("WWPGD target_alpha must be 2.0")
meta = json.loads((data_root / "meta.json").read_text())
expected = {"train": 10_000_000, "val": 1_000_000, "test": 1_000_000}
if meta.get("tokenizer") != "gpt2" or meta.get("splits") != expected:
    raise SystemExit("prepared data does not match the frozen GPT-2-BPE identity")
print("PASS: paired configuration and shared data identity verified.")
PY

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"

run_baseline() {
  local seed run complete log
  for seed in "${SEED_ARRAY[@]}"; do
    run="$BASELINE_RESULTS/adamw_seed_${seed}"
    complete="$run/run_complete.json"
    log="$PAIR_ROOT/logs/baseline-seed${seed}.log"
    if [[ -f "$complete" ]]; then
      printf '%s\n' "[paired-level0] baseline seed=$seed already complete; skipping"
      continue
    fi
    if [[ -e "$run" ]]; then
      printf '%s\n' \
        "error: partial baseline run exists without completion marker: $run" \
        >&2
      exit 1
    fi
    printf '%s\n' "[paired-level0] starting baseline seed=$seed"
    env \
      NANOGPT_LEVEL0_ROOT="$(dirname "$DATA_ROOT")" \
      NANOGPT_LEVEL0_DATA_ROOT="$DATA_ROOT" \
      NANOGPT_LEVEL0_RESULTS_ROOT="$BASELINE_RESULTS" \
      NANOGPT_LEVEL0_DEVICE="$DEVICE" \
      bash "$REPO_ROOT/level_0_baseline/scripts/run_one.sh" adamw "$seed" \
      2>&1 | tee "$log"
  done
}

run_wwpgd() {
  local seed run complete log
  for seed in "${SEED_ARRAY[@]}"; do
    run="$WWPGD_RESULTS/adamw_wwpgd_seed_${seed}"
    complete="$run/run_complete.json"
    log="$PAIR_ROOT/logs/wwpgd-seed${seed}.log"
    if [[ -f "$complete" ]]; then
      printf '%s\n' "[paired-level0] WWPGD seed=$seed already complete; skipping"
      continue
    fi
    if [[ -e "$run" ]]; then
      printf '%s\n' \
        "error: partial WWPGD run exists without completion marker: $run" \
        >&2
      exit 1
    fi
    printf '%s\n' "[paired-level0] starting WWPGD seed=$seed"
    env \
      NANOGPT_LEVEL0_ROOT="$(dirname "$DATA_ROOT")" \
      NANOGPT_LEVEL0_DATA_ROOT="$DATA_ROOT" \
      NANOGPT_LEVEL0_WWPGD_ROOT="$PAIR_ROOT/wwpgd" \
      NANOGPT_LEVEL0_WWPGD_DATA_ROOT="$DATA_ROOT" \
      NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT="$WWPGD_RESULTS" \
      NANOGPT_LEVEL0_WWPGD_DEVICE="$DEVICE" \
      bash "$REPO_ROOT/level_0_wwpgd/scripts/run_one.sh" adamw "$seed" \
      2>&1 | tee "$log"
  done
}

verify_runs() {
  python - "$PAIR_ROOT" "$SEEDS" <<'PY'
import json
import sys
from pathlib import Path

pair_root = Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2].split(",") if value]
baseline_root = pair_root / "baseline/results"
wwpgd_root = pair_root / "wwpgd/results"

baseline_done = []
wwpgd_done = []
for seed in seeds:
    baseline = baseline_root / f"adamw_seed_{seed}" / "run_complete.json"
    wwpgd = wwpgd_root / f"adamw_wwpgd_seed_{seed}" / "run_complete.json"
    if baseline.is_file() and json.loads(baseline.read_text()).get("completed") is True:
        baseline_done.append(seed)
    if wwpgd.is_file() and json.loads(wwpgd.read_text()).get("completed") is True:
        wwpgd_done.append(seed)

print(f"Pair root:        {pair_root}")
print(f"Baseline results: {baseline_root}")
print(f"WWPGD results:    {wwpgd_root}")
print(f"Baseline complete ({len(baseline_done)}/{len(seeds)}): {baseline_done}")
print(f"WWPGD complete    ({len(wwpgd_done)}/{len(seeds)}): {wwpgd_done}")
PY
}

case "$PHASE" in
  baseline) run_baseline ;;
  wwpgd) run_wwpgd ;;
  all) run_baseline; run_wwpgd ;;
  verify) ;;
esac

verify_runs

cat <<EOF

The paired experiment path is recorded in:
  $STATE_FILE

Run the comparison notebook with:
  env NANOGPT_LEVEL0_BASELINE_RESULTS_ROOT="$BASELINE_RESULTS" \\
      NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT="$WWPGD_RESULTS" \\
      jupyter lab "$REPO_ROOT/level_0_wwpgd/notebooks/03_compare_baseline_wwpgd.ipynb"
EOF
