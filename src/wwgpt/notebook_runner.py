"""Single implementation used by the CLI and shell notebook runner."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [f"{i:02d}_{name}.ipynb" for i, name in enumerate([
    "", "validate_repository", "compare_single_level", "weightwatcher_analysis",
    "scaling_laws", "overfitting_and_generalization", "summary_report",
    "wwpgd_diagnostics"] ) if i]


def run_notebooks(results_root: Path, output_root: Path, *, analysis_plan: Path | None = None,
                  profile: str = "", level: int | None = None,
                  token_multiplier: int | None = None, base_optimizer: str = "adamw",
                  notebooks: str = "all", strict: bool = False, run_analysis: bool = False,
                  reuse_existing_analysis: bool = True) -> list[Path]:
    results_root = results_root.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"WWGPT_RESULTS_ROOT does not exist: {results_root}")
    output_root = output_root.resolve()
    for d in ("executed", "tables", "figures", "logs"):
        (output_root / d).mkdir(parents=True, exist_ok=True)
    if run_analysis:
        marker = output_root / "tables" / ".analysis-complete"
        if not (reuse_existing_analysis and marker.exists()):
            cmd = [sys.executable, "-m", "wwgpt.cli", "analyze-results", str(results_root)]
            if analysis_plan: cmd += ["--analysis-plan", str(analysis_plan)]
            subprocess.run(cmd, check=True); marker.touch()
    selected = NOTEBOOKS if notebooks.strip().lower() == "all" else [x.strip() for x in notebooks.split(",") if x.strip()]
    unknown = set(selected) - set(NOTEBOOKS)
    if unknown: raise ValueError(f"unknown notebook(s): {', '.join(sorted(unknown))}")
    repo = Path(__file__).resolve().parents[2]
    outputs = []
    for name in selected:
        source, dest = repo / "notebooks" / name, output_root / "executed" / name
        log = output_root / "logs" / f"{source.stem}.log"
        cmd = [sys.executable, "-m", "papermill", str(source), str(dest), "--log-output",
               "-p", "RESULTS_ROOT", str(results_root), "-p", "OUTPUT_ROOT", str(output_root),
               "-p", "BASE_OPTIMIZER", base_optimizer, "-p", "STRICT", str(strict),
               "-p", "RUN_ANALYSIS", "False", "-p", "REUSE_EXISTING_ANALYSIS", str(reuse_existing_analysis)]
        for key, value in (("ANALYSIS_PLAN", analysis_plan), ("PROFILE", profile),
                           ("LEVEL", level), ("TOKEN_MULTIPLIER", token_multiplier)):
            if value not in (None, ""): cmd += ["-p", key, str(value)]
        with log.open("w") as stream:
            subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
        outputs.append(dest)
    return outputs
