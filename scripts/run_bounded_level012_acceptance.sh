#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 RESULTS_ROOT" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_ROOT="$1"
PLAN="$RESULTS_ROOT/release_acceptance_plan.yaml"

python scripts/run_bounded_level012_acceptance.py --results-root "$RESULTS_ROOT"
wwgpt check-health --experiment-root "$RESULTS_ROOT" >"$RESULTS_ROOT/release_health.log"
wwgpt analyze-results "$RESULTS_ROOT" --analysis-plan "$PLAN" >"$RESULTS_ROOT/release_analysis.log"
wwgpt audit-experiment --experiment-root "$RESULTS_ROOT" >"$RESULTS_ROOT/release_audit.log"
FIRST_REPORT="$(wwgpt generate-reproducibility-report --experiment-root "$RESULTS_ROOT" --analysis-plan "$PLAN" --strict)"
SECOND_REPORT="$(wwgpt generate-reproducibility-report --experiment-root "$RESULTS_ROOT" --analysis-plan "$PLAN" --strict)"
if [ "$FIRST_REPORT" != "$SECOND_REPORT" ]; then
  echo "reproducibility report path changed across identical reruns" >&2
  exit 1
fi

python - "$RESULTS_ROOT" <<'PY'
from pathlib import Path
import json
import sys
import pandas as pd

root = Path(sys.argv[1])
analysis = root / "analysis"
complete = sorted(root.rglob("run_complete.json"))
if len(complete) != 6:
    raise SystemExit(f"expected six complete runs, found {len(complete)}")
inventory = pd.read_csv(analysis / "runs_manifest.csv")
if len(inventory) != 6:
    raise SystemExit(f"expected six analyzed arms, found {len(inventory)}")
if set(pd.to_numeric(inventory["level"], errors="raise").astype(int)) != {0, 1, 2}:
    raise SystemExit("analysis did not preserve Levels 0, 1, and 2")
required = [
    analysis / "analysis_eligibility.json",
    analysis / "acceleration_by_seed.csv",
    analysis / "integrity_summary.json",
    analysis / "reproducibility_report.json",
    analysis / "reproducibility_report.pdf",
    analysis / "cross_level_run_inventory.csv",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing release artifacts: {missing}")
eligibility = json.loads((analysis / "analysis_eligibility.json").read_text())
if not eligibility.get("eligible"):
    raise SystemExit(f"analysis ineligible: {eligibility}")
print("LEVEL_0_1_2_RELEASE_ACCEPTANCE_PASS")
PY
