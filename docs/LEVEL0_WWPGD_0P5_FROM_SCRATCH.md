# Level-0 WW-PGD strength-0.5 workflow from scratch

This workflow runs the level-0 strength-scan path with exactly five seeds and a single WW-PGD mixture/strength value of `0.5`. It uses environment variables for data, results, logs, notebook discovery, and the alpha-error confirmation step.

## Environment variables

```bash
export DATA_ROOT=/tmp/wwpgd_v2/data
export RESULTS_ROOT=/tmp/wwpgd_level0_wwpgd_0p5
export WWGPT_STRENGTH_SCAN_ROOT="$RESULTS_ROOT"
export RUN_LOG="$RESULTS_ROOT/level0_wwpgd_0p5.log"
export PID_FILE="$RESULTS_ROOT/level0_wwpgd_0p5.pid"
export LEVEL=0
export TOKEN_MULTIPLIER=20
export SEEDS=1337,2027,4099,7919,104729
export WWPGD_STRENGTH=0.5
export DEVICE=auto      # use cuda, mps, or cpu if you want to force a backend
```

## One-command run

```bash
./scripts/run_level0_wwpgd_0p5_from_scratch.sh
```

The script installs the package in editable mode, prepares level-0 data, runs `wwgpt run-strength-scan` for the five seeds with `--strengths 0.5`, analyzes the scan, executes the two strength-scan notebooks using `WWGPT_STRENGTH_SCAN_ROOT`, and checks every row in `wwpgd_projection_spectral.csv`.

## Manual commands

```bash
python -m pip install -e .
./scripts/download_data.sh "$LEVEL" "$DATA_ROOT" "$TOKEN_MULTIPLIER"
wwgpt run-strength-scan \
  --level "$LEVEL" \
  --data-root "$DATA_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --token-multiplier "$TOKEN_MULTIPLIER" \
  --seeds "$SEEDS" \
  --strengths "$WWPGD_STRENGTH" \
  --device "$DEVICE" \
  --eval-interval 25 \
  --spectral-interval 100 \
  --checkpoint-interval 500 \
  --immediate-projection-spectral \
  --resume
export WWGPT_STRENGTH_SCAN_ROOT="$(cat "$RESULTS_ROOT/latest_scan_root.txt")"
wwgpt analyze-strength-scan --scan-root "$WWGPT_STRENGTH_SCAN_ROOT"
jupyter nbconvert --to notebook --execute notebooks/07_strength_scan_overview.ipynb --output "$RESULTS_ROOT/notebook_runs/07_strength_scan_overview_executed.ipynb" --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute notebooks/08_strength_scan_weightwatcher.ipynb --output "$RESULTS_ROOT/notebook_runs/08_strength_scan_weightwatcher_executed.ipynb" --ExecutePreprocessor.timeout=-1
```

## Confirming layer alpha movement toward zero error

The WeightWatcher target is `target_alpha = 2.0`; the confirmation checks that each logged model layer has `abs_alpha_error_change <= 0`, meaning its distance to the target moved toward zero after the WW-PGD projection.

```bash
python - <<'PY'
import os
from pathlib import Path
import pandas as pd
scan = Path(os.environ['WWGPT_STRENGTH_SCAN_ROOT'])
files = list(scan.glob('seeds/*/strengths/*/*/run_*/wwpgd_projection_spectral.csv'))
df = pd.concat((pd.read_csv(p).assign(source=str(p)) for p in files), ignore_index=True)
valid = df.dropna(subset=['abs_alpha_error_change'])
failed = valid[valid['abs_alpha_error_change'] > 1e-12]
print(f'checked_projection_rows={len(valid)}')
print(f'max_abs_alpha_error_change={valid.abs_alpha_error_change.max()}')
if not failed.empty:
    print(failed[['seed','layer_name','abs_alpha_error_before','abs_alpha_error_after','abs_alpha_error_change','source']].to_string(index=False))
    raise SystemExit('Some layer alpha errors did not move toward zero')
PY
```
