from __future__ import annotations
import json, csv, math, hashlib
from pathlib import Path
import pandas as pd



CANONICAL_ARMS = ("adamw", "adamw_wwpgd", "muon", "muon_wwpgd", "stableadamw", "stableadamw_wwpgd")
CANONICAL_PAIRS = {"adamw": "adamw_wwpgd", "muon": "muon_wwpgd", "stableadamw": "stableadamw_wwpgd"}
BASELINE_EXTENSIONS = {"", "none", None}
WWPGD_REQUIRED_EXTENSION_ARTIFACTS = ("wwpgd_projection.csv",)


def _load_json(path: Path) -> tuple[dict, str | None]:
    try:
        return json.loads(path.read_text()), None
    except Exception as e:
        return {}, f"unreadable_json:{path.name}:{type(e).__name__}"


def _read_csv(path: Path) -> tuple[pd.DataFrame, str | None]:
    try:
        df = pd.read_csv(path)
        return df, None if len(df) else f"empty_csv:{path.name}"
    except Exception as e:
        return pd.DataFrame(), f"unreadable_csv:{path.name}:{type(e).__name__}"


def _fingerprint_without_name(fp):
    if not isinstance(fp, dict):
        return fp
    return {k: v for k, v in fp.items() if k not in {"name", "optimizer", "optimizer_name"}}


def _arm_dir(trial: Path, arm: str) -> Path | None:
    root = trial / arm
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    complete = [r for r in runs if (r / "run_complete.json").exists() and (r / "manifest.json").exists()]
    candidates = complete or [r for r in runs if (r / "manifest.json").exists()] or runs
    # Directory names are run ids, not scientific identity; sorting only makes selection deterministic.
    return sorted(candidates)[-1] if candidates else None


def _selected_checkpoint_ok(run: Path, man: dict, metrics: pd.DataFrame, complete: dict) -> tuple[bool, str | None]:
    if metrics.empty:
        return False, "metrics_missing_or_empty"
    artifact = run / "selected_checkpoint_metrics.json"
    if not artifact.exists():
        # Legacy schema: held-out values lived on the last time-series row.
        final = metrics.tail(1).iloc[0]
        step = final.get("selected_checkpoint_step", complete.get("selected_checkpoint_step", complete.get("best_validation_step", complete.get("step"))))
        if pd.isna(step): return False, "selected_checkpoint_step_missing_for_test_metrics"
        selected_path = run / "checkpoints" / f"best_val_step_{int(float(step)):06d}_{int(man.get('seed', 0))}.pt"
        final_path = run / "checkpoints" / f"final_step_{int(float(step)):06d}_{int(man.get('seed', 0))}.pt"
        latest = run / "checkpoints" / "latest.json"
        if not (selected_path.exists() or final_path.exists() or latest.exists()):
            return False, "selected_checkpoint_artifact_missing_for_test_metrics"
        return True, None
    try:
        record = json.loads(artifact.read_text())
        required = {f"{split}_{metric}" for split in ("train", "validation", "test") for metric in ("loss", "perplexity", "accuracy")}
        required |= {"checkpoint_path", "checkpoint_hash", "selected_step", "selection_metric",
                     "train_validation_gap", "train_test_gap", "train_probe_hash",
                     "validation_probe_hash", "test_probe_hash"}
        if not required.issubset(record): return False, "selected_checkpoint_metrics_fields_missing"
        if record["selection_metric"] != "validation_loss": return False, "checkpoint_selection_not_validation_only"
        selected_path = Path(record["checkpoint_path"])
        if not selected_path.is_absolute(): selected_path = run / selected_path
        if not selected_path.exists(): return False, "selected_checkpoint_artifact_missing"
        actual = hashlib.sha256(selected_path.read_bytes()).hexdigest()
        if actual != record["checkpoint_hash"]: return False, "selected_checkpoint_hash_mismatch"
        if f"{int(record['selected_step']):06d}" not in selected_path.name: return False, "selected_checkpoint_step_path_mismatch"
        csv_path = run / "selected_checkpoint_metrics.csv"
        if not csv_path.exists(): return False, "selected_checkpoint_metrics_csv_missing"
        csv_record = pd.read_csv(csv_path).iloc[0].to_dict()
        for key in required:
            if key not in csv_record: return False, "selected_checkpoint_json_csv_schema_mismatch"
    except Exception as exc:
        return False, f"selected_checkpoint_integrity_error:{type(exc).__name__}"
    return True, None


def audit_arm(run: Path, required_arm: str | None = None) -> dict:
    run = Path(run); reasons = []
    mismatch_audits = [json.loads(p.read_text()) for p in run.glob("code_version_mismatch_*.json")]
    if any(not a.get("audit_override", False) for a in mismatch_audits):
        reasons.append("unaudited_code_version_mismatch")
    man, err = _load_json(run / "manifest.json") if (run / "manifest.json").exists() else ({}, "missing_manifest")
    if err: reasons.append(err)
    complete, err = _load_json(run / "run_complete.json") if (run / "run_complete.json").exists() else ({}, "run_incomplete")
    if err: reasons.append(err)
    metrics, err = _read_csv(run / "metrics.csv") if (run / "metrics.csv").exists() else (pd.DataFrame(), "missing_metrics")
    if err: reasons.append(err)
    arm = required_arm or man.get("arm_name") or man.get("optimizer") or run.parent.name
    base = man.get("base_optimizer") or arm.removesuffix("_wwpgd")
    ext = man.get("extension") or ("wwpgd" if str(arm).endswith("_wwpgd") else "none")
    if required_arm and arm != required_arm: reasons.append("arm_name_mismatch")
    if not man.get("valid_for_science", False): reasons.append("fixture_or_invalid_for_science")
    if ext in BASELINE_EXTENSIONS and (run / "wwpgd_projection.csv").exists(): reasons.append("baseline_has_wwpgd_projection_artifact")
    if ext == "wwpgd":
        cached = man.get("adapter_mode") == "cached_endpoint_relaxation_v1"
        if not cached:
            for f in WWPGD_REQUIRED_EXTENSION_ARTIFACTS:
                if not (run / f).exists(): reasons.append(f"missing_required_extension_artifact:{f}")
        if not (man.get("wwpgd_implementation") and man.get("wwpgd_implementation") != "none" and man.get("wwpgd_commit")):
            reasons.append("missing_resolved_wwpgd_metadata")
        if cached:
            metadata = ("endpoint_measurement_source", "endpoint_measurement_interval", "endpoint_apply_interval",
                        "expected_endpoint_measurement_steps", "expected_fast_apply_steps", "expected_measurement_count",
                        "start_step", "max_per_step_gain", "max_relative_frobenius_change_per_step",
                        "endpoint_stop_relative_distance", "max_endpoint_age_steps", "stale_distance_multiplier",
                        "controller_version", "adapter_mode")
            if any(key not in man for key in metadata): reasons.append("missing_cached_controller_metadata")
            measurements, merr = _read_csv(run / "wwpgd_endpoint_measurements.csv") if (run / "wwpgd_endpoint_measurements.csv").exists() else (pd.DataFrame(), "missing_endpoint_measurements")
            if merr or measurements.empty: reasons.append("missing_endpoint_measurement_records")
            elif (not {"skip_reason", "cache_activated"}.issubset(measurements.columns)
                  or ((measurements["skip_reason"].fillna("").astype(str) == "")
                      & ~measurements["cache_activated"].astype(str).str.lower().isin(["true", "1"])).any()):
                reasons.append("measurement_outcomes_missing")
            if int(complete.get("completed_measurement_count", -1)) != int(man.get("expected_measurement_count", -2)): reasons.append("incomplete_endpoint_measurements")
            relaxation, rerr = _read_csv(run / "wwpgd_endpoint_relaxation.csv") if (run / "wwpgd_endpoint_relaxation.csv").exists() else (pd.DataFrame(), "missing_endpoint_relaxation")
            if rerr and not str(rerr).startswith("empty_csv:"): reasons.append("unreadable_endpoint_relaxation")
            if not relaxation.empty and relaxation.get("invalidation_reason", pd.Series(dtype=str)).astype(str).isin(["nonfinite_tensor", "endpoint_corruption"]).any(): reasons.append("endpoint_corruption")
        else:
            if int(complete.get("wwpgd_call_count", man.get("wwpgd_call_count", 0)) or 0) <= 0: reasons.append("missing_wwpgd_call_count")
            if int(complete.get("projected_matrix_count", man.get("projected_matrix_count", 0)) or 0) <= 0: reasons.append("missing_projected_matrix_count")
            proj, perr = _read_csv(run / "wwpgd_projection.csv") if (run / "wwpgd_projection.csv").exists() else (pd.DataFrame(), None)
            if perr or proj.empty: reasons.append("missing_projection_records")
    ok, msg = _selected_checkpoint_ok(run, man, metrics, complete)
    if not ok and msg: reasons.append(msg)
    return {"arm_name": arm, "base_optimizer": base, "extension": ext, "run_dir": str(run), "passed": not reasons, "reasons": reasons, "identity": {k: man.get(k) for k in ("data_hash", "tokenizer_hash", "model_configuration_hash", "model_config_hash", "realized_tokens", "requested_tokens", "target_train_tokens", "initialization_hash", "training_schedule_hash", "resolved_stochastic_seeds", "optimizer_fingerprint", "weight_decay")}}


def audit_trial(trial: Path) -> dict:
    trial = Path(trial); arms = {}; reasons = []
    for arm in CANONICAL_ARMS:
        run = _arm_dir(trial, arm)
        if run is None:
            arms[arm] = {"arm_name": arm, "passed": False, "reasons": ["missing_arm"], "run_dir": None, "identity": {}}
        else:
            arms[arm] = audit_arm(run, arm)
    for arm, rec in arms.items():
        if not rec["passed"]: reasons.append(f"{arm}:" + ",".join(rec["reasons"]))
    for b, w in CANONICAL_PAIRS.items():
        bi, wi = arms[b]["identity"], arms[w]["identity"]
        if _fingerprint_without_name(bi.get("optimizer_fingerprint")) != _fingerprint_without_name(wi.get("optimizer_fingerprint")):
            reasons.append(f"{b}/{w}:base_optimizer_fingerprint_mismatch")
        for k in ("training_schedule_hash", "resolved_stochastic_seeds", "initialization_hash"):
            if bi.get(k) != wi.get(k): reasons.append(f"{b}/{w}:{k}_mismatch")
    for k in ("data_hash", "tokenizer_hash", "model_configuration_hash", "model_config_hash", "realized_tokens", "requested_tokens", "target_train_tokens"):
        vals = {json.dumps(a["identity"].get(k), sort_keys=True, default=str) for a in arms.values()}
        if len(vals) > 1: reasons.append(f"all_arms:{k}_mismatch")
    return {"trial_dir": str(trial), "required_arm_count": len(CANONICAL_ARMS), "passed_arm_count": sum(a["passed"] for a in arms.values()), "publication_eligible": not reasons and all(a["passed"] for a in arms.values()), "reasons": reasons, "arms": arms}


def audit_run(run: Path):
    run=Path(run); reasons=[]
    man={}
    if (run/'manifest.json').exists(): man=json.loads((run/'manifest.json').read_text())
    fixture=not man.get('valid_for_science', False) or man.get('dataset_name')=='local_fixture'
    if fixture: reasons.append('fixture_or_invalid_for_science')
    mismatch_audits=[json.loads(p.read_text()) for p in run.glob('code_version_mismatch_*.json')]
    if any(not a.get('audit_override', False) for a in mismatch_audits): reasons.append('unaudited_code_version_mismatch')
    imm=run/'wwpgd_projection_spectral.csv'
    valid_immediate=False
    if imm.exists():
        df=pd.read_csv(imm)
        required={'immediate_spectral_source','measurement_valid_for_science','alpha_before','alpha_after','weightwatcher_configuration'}
        if not required.issubset(df.columns): reasons.append('legacy_or_missing_measured_provenance_fields')
        elif (df['immediate_spectral_source']=='weightwatcher_measured').all() and df['measurement_valid_for_science'].astype(str).str.lower().isin(['true','1']).any(): valid_immediate=True
        else: reasons.append('no_valid_immediate_weightwatcher_rows')
    complete=(run/'run_complete.json').exists()
    if not complete: reasons.append('run_incomplete')
    valid_loss=complete and not fixture and (run/'metrics.csv').exists()
    out={
        'run_dir':str(run),'valid_for_loss_analysis':valid_loss,'valid_for_accuracy_analysis':valid_loss,
        'valid_for_periodic_weightwatcher_analysis':complete and not fixture and (run/'spectral.csv').exists(),
        'valid_for_immediate_weightwatcher_analysis':valid_immediate,
        'valid_for_projection_analysis':complete and not fixture and (run/'wwpgd_projection.csv').exists(),
        'valid_for_publication':False,'reasons':';'.join(reasons)
    }
    out['valid_for_publication']=all(out[k] for k in out if k.startswith('valid_for_') and k!='valid_for_publication') and not reasons
    return out

def audit_experiment(root: Path):
    root=Path(root); analysis=root/'analysis'; analysis.mkdir(parents=True, exist_ok=True)
    trial_dirs=[p.parent for p in root.rglob('trial_manifest.json')]
    trial_rows=[audit_trial(t) for t in sorted(set(trial_dirs))]
    runs=[p.parent for p in root.rglob('manifest.json') if p.parent.name.startswith('run_')]
    rows=[audit_run(r) for r in runs]
    fields=list(rows[0]) if rows else ['run_dir','valid_for_publication','reasons']
    with (analysis/'integrity_audit.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={'experiment_root':str(root),'run_count':len(rows),'publication_eligible_runs':sum(bool(r.get('valid_for_publication')) for r in rows),'valid_for_publication':bool(rows) and all(r.get('valid_for_publication') for r in rows),'failures':[r for r in rows if not r.get('valid_for_publication')], 'trial_count': len(trial_rows), 'publication_eligible_trials': sum(t['publication_eligible'] for t in trial_rows), 'trials': trial_rows}
    if trial_rows:
        summary['valid_for_publication']=all(t['publication_eligible'] for t in trial_rows)
    (analysis/'integrity_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    lines=['# Integrity audit','']
    if trial_rows:
        for t in trial_rows:
            mark='PASS' if t['publication_eligible'] else 'FAIL'
            lines.append(f"- {mark} {Path(t['trial_dir']).name}: {t['passed_arm_count']}/{t['required_arm_count']} arms passed")
            for r in t['reasons'][:8]: lines.append(f"  - {r}")
    else:
        lines.append(json.dumps(summary, indent=2))
    (analysis/'integrity_report.md').write_text('\n'.join(lines)+'\n')
    return analysis/'integrity_summary.json'

def audit_strength_scan(scan_root: Path):
    return audit_experiment(scan_root)
