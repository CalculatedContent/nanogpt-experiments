from __future__ import annotations

import json, time
from pathlib import Path
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from wwgpt.analysis import analyze_results, completed_runs
from wwgpt.integrity import audit_experiment
from wwgpt.utils import write_json


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def write_reproducibility_report(experiment_root: Path) -> Path:
    """Write a real PDF and machine-readable reproducibility manifest.

    The report summarizes existing run artifacts only. Missing artifacts are reported as
    missing; no scientific values are inferred or substituted.
    """
    root = Path(experiment_root)
    analysis_dir = analyze_results(root)
    audit_path = audit_experiment(root)
    runs = completed_runs(root, scientific_only=True)
    rows = []
    for run in runs:
        man = _read_json(run / "manifest.json")
        comp = _read_json(run / "run_complete.json")
        rows.append({
            "run_dir": str(run),
            "optimizer": man.get("optimizer"),
            "seed": man.get("seed"),
            "pair_id": man.get("pair_id"),
            "valid_for_science": man.get("valid_for_science"),
            "spectral_estimator": man.get("spectral_estimator"),
            "final_step": comp.get("step"),
            "final_val_loss": comp.get("final_val_loss"),
            "configuration_hash": man.get("configuration_hash"),
            "data_hash": man.get("data_hash") or man.get("corpus_hash"),
            "tokenizer_hash": man.get("tokenizer_hash"),
            "initialization_hash": man.get("initialization_hash"),
            "validation_probe_hash": man.get("validation_probe_hash"),
            "training_probe_hash": man.get("training_probe_hash"),
        })
    df = pd.DataFrame(rows)
    csv_path = analysis_dir / "reproducibility_report_runs.csv"
    df.to_csv(csv_path, index=False)
    manifest = {
        "experiment_root": str(root),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_count": len(rows),
        "analysis_dir": str(analysis_dir),
        "audit_path": str(audit_path),
        "runs_csv": str(csv_path),
        "weightwatcher_only_policy": "scientific spectral rows must use spectral_estimator=weightwatcher; missing measurements remain missing/invalid",
    }
    write_json(analysis_dir / "reproducibility_report.json", manifest)
    pdf_path = analysis_dir / "reproducibility_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.94, "nanoGPT WW-PGD reproducibility report", fontsize=16, weight="bold")
        lines = [
            f"Experiment root: {root}",
            f"Generated (UTC): {manifest['generated_at_utc']}",
            f"Scientific completed runs: {len(rows)}",
            f"Audit artifact: {audit_path}",
            f"Run inventory CSV: {csv_path}",
            "Spectral analysis policy: WeightWatcher only; no surrogate or substitute spectral values.",
        ]
        if not df.empty:
            lines += ["", "Runs:"] + [f"- seed={r.seed} optimizer={r.optimizer} step={r.final_step} valid={r.valid_for_science}" for r in df.itertuples()]
        else:
            lines += ["", "No complete scientific runs discovered; report contains no scientific results."]
        fig.text(0.08, 0.88, "\n".join(lines), fontsize=10, va="top", family="monospace")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    return pdf_path
