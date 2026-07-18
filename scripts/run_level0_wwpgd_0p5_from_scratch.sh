#!/usr/bin/env bash
set -euo pipefail

# End-to-end level-0 WW-PGD strength-0.5 workflow.
# Override any variable before invocation, for example:
#   DATA_ROOT=/data/wwpgd RESULTS_ROOT=/results/wwpgd DEVICE=cuda ./scripts/run_level0_wwpgd_0p5_from_scratch.sh

LEVEL="${LEVEL:-0}"
TOKEN_MULTIPLIER="${TOKEN_MULTIPLIER:-20}"
SEEDS="${SEEDS:-1337,2027,4099,7919,104729}"
WWPGD_STRENGTH="${WWPGD_STRENGTH:-0.5}"
DEVICE="${DEVICE:-auto}"
DATA_ROOT="${DATA_ROOT:-/tmp/wwpgd_v2/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/tmp/wwpgd_level0_wwpgd_0p5}"
RUN_LOG="${RUN_LOG:-$RESULTS_ROOT/level0_wwpgd_0p5_$(date +%Y%m%d-%H%M%S).log}"
PID_FILE="${PID_FILE:-$RESULTS_ROOT/level0_wwpgd_0p5.pid}"
export DATA_ROOT RESULTS_ROOT RUN_LOG PID_FILE WWGPT_STRENGTH_SCAN_ROOT="$RESULTS_ROOT"

mkdir -p "$DATA_ROOT" "$RESULTS_ROOT" "$RESULTS_ROOT/notebook_runs"
printf '%s\n' "$$" > "$PID_FILE"

echo "[level0] installing package in editable mode" | tee -a "$RUN_LOG"
python -m pip install -e . 2>&1 | tee -a "$RUN_LOG"

echo "[level0] preparing data" | tee -a "$RUN_LOG"
./scripts/download_data.sh "$LEVEL" "$DATA_ROOT" "$TOKEN_MULTIPLIER" 2>&1 | tee -a "$RUN_LOG"

echo "[level0] running five-seed WW-PGD strength scan at strength=${WWPGD_STRENGTH}" | tee -a "$RUN_LOG"
wwgpt run-strength-scan \
  --level "$LEVEL" \
  --data-root "$DATA_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --token-multiplier "$TOKEN_MULTIPLIER" \
  --seeds "$SEEDS" \
  --strengths "$WWPGD_STRENGTH" \
  --device "$DEVICE" \
  --eval-interval "${EVAL_INTERVAL:-25}" \
  --spectral-interval "${SPECTRAL_INTERVAL:-100}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-500}" \
  --immediate-projection-spectral \
  --resume 2>&1 | tee -a "$RUN_LOG"

SCAN_ROOT="$(cat "$RESULTS_ROOT/latest_scan_root.txt")"
export WWGPT_STRENGTH_SCAN_ROOT="$SCAN_ROOT"
echo "[level0] scan root: $SCAN_ROOT" | tee -a "$RUN_LOG"

wwgpt analyze-strength-scan --scan-root "$SCAN_ROOT" 2>&1 | tee -a "$RUN_LOG"

for nb in notebooks/07_strength_scan_overview.ipynb notebooks/08_strength_scan_weightwatcher.ipynb; do
  out="$RESULTS_ROOT/notebook_runs/$(basename "${nb%.ipynb}")_executed.ipynb"
  echo "[level0] executing $nb -> $out" | tee -a "$RUN_LOG"
  jupyter nbconvert --to notebook --execute "$nb" --output "$out" --ExecutePreprocessor.timeout=-1 2>&1 | tee -a "$RUN_LOG"
done

echo "[level0] confirming every logged per-layer WW-PGD projection moved alpha error toward zero" | tee -a "$RUN_LOG"
python - <<'PY' 2>&1 | tee -a "$RUN_LOG"
import os
from pathlib import Path
import pandas as pd
scan = Path(os.environ['WWGPT_STRENGTH_SCAN_ROOT'])
files = list(scan.glob('seeds/*/strengths/*/*/run_*/wwpgd_projection_spectral.csv'))
if not files:
    raise SystemExit(f'No wwpgd_projection_spectral.csv files found under {scan}')
df = pd.concat((pd.read_csv(p).assign(source=str(p)) for p in files), ignore_index=True)
required = {'seed', 'layer_name', 'abs_alpha_error_before', 'abs_alpha_error_after', 'abs_alpha_error_change'}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f'Missing required columns: {sorted(missing)}')
valid = df.dropna(subset=['abs_alpha_error_before', 'abs_alpha_error_after', 'abs_alpha_error_change'])
failed = valid[valid['abs_alpha_error_change'] > 1e-12]
print(f'checked_projection_rows={len(valid)}')
print(f'seeds={sorted(valid.seed.unique().tolist())}')
print(f'layers={sorted(map(str, valid.layer_name.unique().tolist()))}')
print(f'max_abs_alpha_error_change={valid.abs_alpha_error_change.max()}')
print(f'mean_abs_alpha_error_change={valid.abs_alpha_error_change.mean()}')
if not failed.empty:
    print(failed[['seed','layer_name','abs_alpha_error_before','abs_alpha_error_after','abs_alpha_error_change','source']].to_string(index=False))
    raise SystemExit('Some layer alpha errors did not move toward zero')
PY

echo "[level0] complete" | tee -a "$RUN_LOG"
