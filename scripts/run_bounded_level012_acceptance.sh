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

# Keep numerical libraries deterministic and prevent excessive thread creation on
# small CI runners and local laptops. Callers may override any value explicitly.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/wwgpt-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

python scripts/run_bounded_level012_acceptance.py --results-root "$RESULTS_ROOT"

# Run all post-processing in one clean Python process. This still exercises the
# installed package from a shell entry point, while avoiding repeated imports of
# Torch, WeightWatcher, pandas, and Matplotlib across five short-lived processes.
python - "$RESULTS_ROOT" "$PLAN" <<'PY'
from pathlib import Path
import json
import sys

import pandas as pd

from wwgpt.analysis import analyze_results
from wwgpt.integrity import audit_experiment
from wwgpt.reproducibility import write_reproducibility_report
from wwgpt.run_health import generate_experiment_health

root = Path(sys.argv[1]).expanduser().resolve()
plan = Path(sys.argv[2]).expanduser().resolve()
analysis = root / "analysis"

health = generate_experiment_health(root)
(root / "release_health.log").write_text(
    json.dumps(health, indent=2, sort_keys=True, default=str) + "\n"
)
if not health.get("ready_for_analysis"):
    raise SystemExit("release health check failed")

analysis_dir = analyze_results(root, plan)
(root / "release_analysis.log").write_text(str(analysis_dir) + "\n")

audit_path = audit_experiment(root)
audit = json.loads(Path(audit_path).read_text())
(root / "release_audit.log").write_text(str(audit_path) + "\n")
if not audit.get("valid_for_publication"):
    raise SystemExit(f"release integrity audit failed: {audit.get('failures', [])}")

report = write_reproducibility_report(root, strict=True, analysis_plan=plan)
repeated_report = write_reproducibility_report(root, strict=True, analysis_plan=plan)
(root / "release_reproducibility.log").write_text(str(report) + "\n")
if report != repeated_report or not report.is_file():
    raise SystemExit("reproducibility report is not stable across identical reruns")

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
