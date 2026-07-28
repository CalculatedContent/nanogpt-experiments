from __future__ import annotations

import json, time
from pathlib import Path
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from wwgpt.acceleration_analysis import load_analysis_plan
from wwgpt.analysis import analyze_results, completed_runs, discover_canonical_runs
from wwgpt.integrity import audit_experiment
from wwgpt.run_health import generate_experiment_health


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def write_reproducibility_report(
    experiment_root: Path,
    *,
    strict: bool = False,
    analysis_plan: Path | None = None,
) -> Path:
    """Write a real PDF and machine-readable reproducibility manifest.

    The report summarizes existing run artifacts only. Missing artifacts are reported as
    missing; no scientific values are inferred or substituted.
    """
    root = Path(experiment_root)
    health = generate_experiment_health(root)
    audit_path = audit_experiment(root)
    audit = _read_json(audit_path)
    canonical = discover_canonical_runs(root, include_legacy=True)
    runs = [
        Path(record["run_dir"])
        for record in canonical
        if record.get("run_dir")
        and (Path(record["run_dir"]) / "run_complete.json").is_file()
        and record.get("valid_for_science", True) is True
    ]
    if not runs:
        # Preserve support for standalone legacy fixtures that have no explicit
        # pair/trial layout while preferring canonical append-only selections for
        # current scientific experiments.
        runs = completed_runs(root, scientific_only=True)
    if strict and not health.get("ready_for_analysis", False):
        raise RuntimeError(
            "reproducibility report refused an unhealthy experiment: "
            + json.dumps(
                [
                    report
                    for report in health.get("reports", [])
                    if not report.get("ready_for_analysis", False)
                ],
                sort_keys=True,
                default=str,
            )
        )
    if strict and not audit.get("valid_for_publication", False):
        raise RuntimeError(
            "reproducibility report refused an invalid experiment: "
            + json.dumps(audit.get("failures", []), sort_keys=True, default=str)
        )
    if strict and not runs:
        raise RuntimeError("reproducibility report found no complete scientific runs")
    analysis_dir = root / "analysis"
    analysis_manifest = analysis_dir / "analysis_manifest.json"
    analysis_state = _read_json(analysis_manifest)
    analysis_complete = (
        analysis_manifest.is_file() and analysis_state.get("status") == "complete"
    )
    plan_manifest_path = analysis_dir / "analysis_plan_manifest.json"
    plan_matches = analysis_plan is None
    if analysis_plan is not None and plan_manifest_path.is_file():
        _plan, expected_plan_hash = load_analysis_plan(analysis_plan)
        recorded_plan = _read_json(plan_manifest_path)
        plan_matches = recorded_plan.get("analysis_plan_sha256") == expected_plan_hash
    if not analysis_complete or not plan_matches:
        analysis_dir = analyze_results(root, analysis_plan)
    else:
        analysis_dir.mkdir(parents=True, exist_ok=True)
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
        "health_summary_path": str(analysis_dir / "run_health_summary.json"),
        "analysis_plan": str(analysis_plan) if analysis_plan else None,
        "runs_csv": str(csv_path),
        "weightwatcher_only_policy": "scientific spectral rows must use spectral_estimator=weightwatcher; missing measurements remain missing/invalid",
    }
    manifest_path = analysis_dir / "reproducibility_report.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    temporary_manifest.replace(manifest_path)
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
