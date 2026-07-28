#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${WWGPT_VENV:-.venv}"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
./scripts/setup_environment.sh

python - <<'PY'
import importlib.metadata
import pathlib
import weightwatcher
import ww_pgd
print("weightwatcher", importlib.metadata.version("weightwatcher"), pathlib.Path(weightwatcher.__file__).resolve())
try:
    version = importlib.metadata.version("ww-pgd")
except importlib.metadata.PackageNotFoundError:
    version = getattr(ww_pgd, "__version__", "unknown")
print("ww-pgd", version, pathlib.Path(ww_pgd.__file__).resolve())
PY

rm -rf artifacts/macbook-preflight-cpu
wwgpt local-readiness \
  --device cpu \
  --levels 0,1,2 \
  --optimizers adamw,stableadamw,muon \
  --output artifacts/macbook-preflight-cpu

if python - <<'PY'
import torch
raise SystemExit(0 if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 1)
PY
then
  rm -rf artifacts/macbook-preflight-mps
  wwgpt local-readiness \
    --device mps \
    --levels 0,1,2 \
    --optimizers adamw,stableadamw,muon \
    --output artifacts/macbook-preflight-mps
else
  echo "[macbook-preflight] MPS unavailable; CPU validation completed." >&2
fi

for level in 0 1 2; do
  for mode in adaptive fixed; do
    config="configs/level${level}_adaptive_alpha.yaml"
    if [ "$mode" = fixed ]; then
      config="configs/level${level}_fixed_wwpgd.yaml"
    fi
    wwgpt run-multiseed \
      --level "$level" \
      --config "$config" \
      --analysis-plan configs/analysis_plan_exploratory.yaml \
      --data-root /tmp/wwgpt-preflight-data \
      --results-root /tmp/wwgpt-preflight-results \
      --token-multiplier 20 \
      --seeds 1337 \
      --optimizer adamw \
      --extensions none,wwpgd \
      --device cpu \
      --dry-run > "artifacts/macbook-preflight-level${level}-${mode}.txt"
  done
done

echo "[macbook-preflight] PASS" >&2
