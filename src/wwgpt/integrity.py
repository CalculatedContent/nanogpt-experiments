"""Publication-integrity checks based on scientific identity, not path spelling."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from wwgpt.ww import is_projected_layer


CANONICAL_ARMS = ("adamw", "adamw_wwpgd", "muon", "muon_wwpgd", "stableadamw", "stableadamw_wwpgd")
CANONICAL_PAIRS = {"adamw": "adamw_wwpgd", "muon": "muon_wwpgd", "stableadamw": "stableadamw_wwpgd"}
BASELINE_EXTENSIONS = {"", "none", None}


def normalize_arm(value: Any) -> str:
    """Return the canonical scientific arm identity (accept legacy spelling)."""
    return str(value or "").lower().replace("stable_adamw", "stableadamw")


def _eligible_action_name(name: Any) -> bool:
    value = str(name)
    return is_projected_layer(value) or (value.startswith("blocks.") and value.endswith(("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")))


def _load_json(path: Path) -> tuple[dict, str | None]:
    try:
        return json.loads(path.read_text()), None
    except Exception as exc:
        return {}, f"unreadable_json:{path.name}:{type(exc).__name__}"


def _read_csv(path: Path) -> tuple[pd.DataFrame, str | None]:
    try:
        frame = pd.read_csv(path)
        return frame, None if len(frame) else f"empty_csv:{path.name}"
    except Exception as exc:
        return pd.DataFrame(), f"unreadable_csv:{path.name}:{type(exc).__name__}"


def _fingerprint_without_name(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _fingerprint_without_name(v) for k, v in value.items() if k not in {"name", "optimizer", "optimizer_name"}}
    if isinstance(value, list):
        return [_fingerprint_without_name(v) for v in value]
    return value


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


def _run_dirs(container: Path) -> list[Path]:
    direct = [p for p in container.iterdir() if p.is_dir() and p.name.startswith("run_")] if container.is_dir() else []
    candidates = direct or ([container] if (container / "manifest.json").exists() else [])
    return sorted(candidates, key=lambda p: (not (p / "run_complete.json").exists(), p.name))


def _discover_arms(layout: Path) -> dict[str, Path]:
    """Discover arms from manifests, using directory names only as a fallback."""
    found: dict[str, Path] = {}
    for manifest_path in layout.rglob("manifest.json"):
        run = manifest_path.parent
        if not (run.name.startswith("run_") or run == layout):
            continue
        manifest, _ = _load_json(manifest_path)
        arm = normalize_arm(manifest.get("arm_name") or manifest.get("optimizer") or run.parent.name)
        if arm:
            old = found.get(arm)
            if old is None or ((run / "run_complete.json").exists() and not (old / "run_complete.json").exists()):
                found[arm] = run
    return found


def _selected_checkpoint_ok(run: Path, man: dict, metrics: pd.DataFrame, complete: dict) -> tuple[bool, str | None]:
    has_test = not metrics.empty and any(str(c).startswith("test_") and metrics[c].notna().any() for c in metrics.columns)
    artifact = run / "selected_checkpoint_metrics.json"
    if not artifact.exists():
        if not has_test:
            return True, None
        step = complete.get("selected_checkpoint_step", complete.get("best_validation_step"))
        if step is None and "selected_checkpoint_step" in metrics:
            step = metrics["selected_checkpoint_step"].dropna().iloc[-1] if metrics["selected_checkpoint_step"].notna().any() else None
        if step is None:
            return False, "selected_checkpoint_step_missing_for_test_metrics"
        checkpoint_dir = run / "checkpoints"
        if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
            return False, "selected_checkpoint_artifact_missing_for_test_metrics"
        return True, None
    try:
        record = json.loads(artifact.read_text())
        required = {f"{split}_{metric}" for split in ("train", "validation", "test") for metric in ("loss", "perplexity", "accuracy")}
        required |= {"checkpoint_path", "checkpoint_hash", "selected_step", "selection_metric", "train_validation_gap", "train_test_gap", "train_probe_hash", "validation_probe_hash", "test_probe_hash"}
        if not required.issubset(record):
            return False, "selected_checkpoint_metrics_fields_missing"
        if record["selection_metric"] != "validation_loss":
            return False, "checkpoint_selection_not_validation_only"
        selected = Path(record["checkpoint_path"])
        if not selected.is_absolute():
            selected = run / selected
        if not selected.exists():
            return False, "selected_checkpoint_artifact_missing"
        if hashlib.sha256(selected.read_bytes()).hexdigest() != record["checkpoint_hash"]:
            return False, "selected_checkpoint_hash_mismatch"
        if f"{int(record['selected_step']):06d}" not in selected.name:
            return False, "selected_checkpoint_step_path_mismatch"
        csv_path = run / "selected_checkpoint_metrics.csv"
        if not csv_path.exists() or not required.issubset(pd.read_csv(csv_path).columns):
            return False, "selected_checkpoint_json_csv_schema_mismatch"
    except Exception as exc:
        return False, f"selected_checkpoint_integrity_error:{type(exc).__name__}"
    return True, None


def _identity(man: dict) -> dict:
    seeds = man.get("resolved_stochastic_seeds") or {}
    train = man.get("optimizer_hyperparameters") or {}
    return {
        "seed": man.get("seed"), "initialization_hash": man.get("initialization_hash"),
        "model_hash": man.get("model_configuration_hash", man.get("model_config_hash")),
        "dataset": (man.get("dataset_name"), man.get("dataset_config"), man.get("dataset_revision")),
        "corpus_hash": man.get("corpus_hash", man.get("data_hash")), "tokenizer_hash": man.get("tokenizer_hash"),
        "token_budget": (man.get("requested_tokens"), man.get("realized_tokens"), man.get("target_train_tokens"), man.get("resolved_optimizer_steps")),
        "optimizer_fingerprint": _fingerprint_without_name(man.get("optimizer_fingerprint")),
        "lr_schedule": (man.get("lr_schedule"), man.get("scheduler_implementation"), man.get("resolved_warmup_steps"), man.get("resolved_lr_decay_steps"), man.get("min_lr_ratio")),
        "training_reader_seed": seeds.get("train_reader_seed", man.get("training_reader_seed")),
        "evaluation_schedule": (man.get("evaluation_sampling"), man.get("evaluation_schedule_version"), man.get("eval_interval", train.get("eval_interval"))),
        "probe_hashes": (man.get("training_probe_hash"), man.get("validation_probe_hash"), man.get("test_probe_hash")),
        "analysis_plan_hash": man.get("analysis_plan_hash"), "target_alpha": man.get("target_alpha", (man.get("extension_hyperparameters") or {}).get("target_alpha")),
        "measurement_schedule": (train.get("spectral_interval"), (man.get("extension_hyperparameters") or {}).get("measurement", {}).get("alpha_interval", train.get("spectral_interval"))),
    }


def audit_arm(run: Path, required_arm: str | None = None) -> dict:
    run = Path(run); reasons: list[str] = []
    man, err = _load_json(run / "manifest.json") if (run / "manifest.json").exists() else ({}, "missing_manifest")
    if err: reasons.append(err)
    complete, err = _load_json(run / "run_complete.json") if (run / "run_complete.json").exists() else ({}, "run_incomplete")
    if err: reasons.append(err)
    metrics, err = _read_csv(run / "metrics.csv") if (run / "metrics.csv").exists() else (pd.DataFrame(), "missing_metrics")
    if err: reasons.append(err)
    arm = normalize_arm(man.get("arm_name") or man.get("optimizer") or run.parent.name)
    required_arm = normalize_arm(required_arm)
    if required_arm and arm != required_arm: reasons.append("arm_name_mismatch")
    base = normalize_arm(man.get("base_optimizer") or arm.removesuffix("_wwpgd"))
    extension = man.get("extension") or ("wwpgd" if arm.endswith("_wwpgd") else "none")
    if not man.get("valid_for_science", False): reasons.append("fixture_or_invalid_for_science")
    if any(a.get("allowed") or a.get("audit_override") for p in run.glob("code_version_mismatch_*.json") for a in [json.loads(p.read_text())]):
        reasons.append("code_version_mismatch_was_allowed")
    target = man.get("target_alpha", (man.get("extension_hyperparameters") or {}).get("target_alpha"))
    if int(man.get("scientific_schema_version", 0) or 0) >= 3 and target is None: reasons.append("target_alpha_missing")
    if _contains_key(man, "q") or ((run / "config.json").exists() and _contains_key(json.loads((run / "config.json").read_text()), "q")):
        reasons.append("user_configured_q_exposed")

    projection_files = ("wwpgd_projection.csv", "wwpgd_endpoint_measurements.csv", "wwpgd_endpoint_relaxation.csv")
    if extension in BASELINE_EXTENSIONS:
        # Projection artifacts are neither required nor interpreted for controls.
        pass
    elif extension == "wwpgd":
        if not man.get("wwpgd_implementation") or not man.get("wwpgd_commit"): reasons.append("missing_resolved_wwpgd_metadata")
        cached = man.get("adapter_mode") == "cached_endpoint_relaxation_v1" or man.get("projection_schedule_type") == "cached_endpoint_measurement_and_fast_apply"
        required = projection_files[1:] if cached else projection_files[:1]
        frames: dict[str, pd.DataFrame] = {}
        for filename in required:
            frame, ferr = _read_csv(run / filename) if (run / filename).exists() else (pd.DataFrame(), f"missing:{filename}")
            if ferr: reasons.append(f"missing_or_invalid_extension_artifact:{filename}")
            frames[filename] = frame
        actions = pd.concat([f for f in frames.values() if not f.empty], ignore_index=True) if any(not f.empty for f in frames.values()) else pd.DataFrame()
        if "layer_name" not in actions or any(not _eligible_action_name(n) for n in actions.get("layer_name", [])):
            reasons.append("extension_action_names_ineligible_matrix")
        if cached:
            measurement_steps = list(man.get("expected_endpoint_measurement_steps") or [])
            fast_steps = list(man.get("expected_fast_apply_steps") or [])
            measured = frames.get("wwpgd_endpoint_measurements.csv", pd.DataFrame())
            relaxed = frames.get("wwpgd_endpoint_relaxation.csv", pd.DataFrame())
            actual_measurements = sorted(set(measured.get("optimizer_step", pd.Series(dtype=int)).dropna().astype(int)))
            actual_fast = sorted(set(relaxed.get("optimizer_step", pd.Series(dtype=int)).dropna().astype(int)))
            if actual_measurements != sorted(measurement_steps) or int(complete.get("completed_measurement_count", -1)) != len(measurement_steps): reasons.append("slow_measurement_schedule_mismatch")
            if actual_fast != sorted(fast_steps): reasons.append("fast_relaxation_schedule_mismatch")
        else:
            if int(complete.get("wwpgd_call_count", 0) or 0) <= 0: reasons.append("missing_wwpgd_call_count")
            if int(complete.get("projected_matrix_count", 0) or 0) <= 0: reasons.append("missing_projected_matrix_count")
    else:
        reasons.append("unknown_extension")

    # A row explicitly blessed as scientific must actually be a valid measured WW row.
    for path in run.glob("*spectral*.csv"):
        frame, _ = _read_csv(path)
        provenance = {"immediate_spectral_source", "measurement_valid_for_science", "alpha_before", "alpha_after", "weightwatcher_configuration"}
        if path.name == "wwpgd_projection_spectral.csv" and not provenance.issubset(frame.columns):
            reasons.append("legacy_or_missing_measured_provenance_fields")
            continue
        if frame.empty or "measurement_valid_for_science" not in frame: continue
        valid = frame["measurement_valid_for_science"].astype(str).str.lower().isin({"true", "1"})
        source_ok = frame.get("immediate_spectral_source", pd.Series("", index=frame.index)).eq("weightwatcher_measured")
        alpha_ok = pd.to_numeric(frame.get("alpha_after", pd.Series(math.nan, index=frame.index)), errors="coerce").map(math.isfinite)
        if (valid & ~(source_ok & alpha_ok)).any(): reasons.append("invalid_weightwatcher_data_marked_valid")
    ok, selected_reason = _selected_checkpoint_ok(run, man, metrics, complete)
    if not ok: reasons.append(selected_reason)
    confirmatory = bool(man.get("confirmatory") or man.get("analysis_type") == "confirmatory")
    if confirmatory and not man.get("analysis_plan_hash"): reasons.append("confirmatory_analysis_plan_hash_missing")
    return {"arm_name": arm, "base_optimizer": base, "extension": extension, "run_dir": str(run), "passed": not reasons, "reasons": reasons, "identity": _identity(man)}


def audit_trial(layout: Path) -> dict:
    layout = Path(layout); discovered = _discover_arms(layout); arms: dict[str, dict] = {}; reasons: list[str] = []
    pair_layout = (layout / "pair_manifest.json").exists() or layout.name.startswith("pair_")
    if pair_layout:
        pair_manifest, _ = _load_json(layout / "pair_manifest.json") if (layout / "pair_manifest.json").exists() else ({}, None)
        base = normalize_arm(pair_manifest.get("base_optimizer") or next((a.removesuffix("_wwpgd") for a in discovered if a.endswith("_wwpgd")), ""))
        expected = (base, f"{base}_wwpgd") if base else tuple()
    else:
        expected = CANONICAL_ARMS
    for arm in expected:
        run = discovered.get(arm)
        arms[arm] = audit_arm(run, arm) if run else {"arm_name": arm, "passed": False, "reasons": ["missing_arm"], "run_dir": None, "identity": {}}
        if not arms[arm]["passed"]: reasons.append(f"{arm}:" + ",".join(arms[arm]["reasons"]))
    pairs = [(b, w) for b, w in CANONICAL_PAIRS.items() if b in arms] if not pair_layout else ([(expected[0], expected[1])] if len(expected) == 2 else [])
    for baseline, wwpgd in pairs:
        left, right = arms[baseline]["identity"], arms[wwpgd]["identity"]
        for key in left:
            if left.get(key) != right.get(key):
                legacy = "base_optimizer_fingerprint_mismatch" if key == "optimizer_fingerprint" else f"{key}_mismatch"
                reasons.append(f"{baseline}/{wwpgd}:{legacy}")
    # Canonical trials additionally share dataset/model/budget identity across optimizer families.
    if not pair_layout:
        legacy_names = {"corpus_hash": "data_hash", "model_hash": "model_configuration_hash"}
        for key in ("corpus_hash", "tokenizer_hash", "model_hash", "token_budget"):
            values = {json.dumps(a["identity"].get(key), sort_keys=True, default=str) for a in arms.values()}
            if len(values) > 1: reasons.append(f"all_arms:{legacy_names.get(key, key)}_mismatch")
    return {"trial_dir": str(layout), "layout": "pair" if pair_layout else "trial", "required_arm_count": len(expected), "passed_arm_count": sum(a["passed"] for a in arms.values()), "publication_eligible": bool(expected) and not reasons, "reasons": reasons, "arms": arms}


def audit_run(run: Path) -> dict:
    record = audit_arm(run)
    return {"run_dir": str(run), "valid_for_publication": record["passed"], "reasons": ";".join(record["reasons"])}


def audit_experiment(root: Path) -> Path:
    root = Path(root); analysis = root / "analysis"; analysis.mkdir(parents=True, exist_ok=True)
    layouts = sorted({p.parent for name in ("trial_manifest.json", "pair_manifest.json") for p in root.rglob(name)})
    trials = [audit_trial(p) for p in layouts]
    runs = sorted({p.parent for p in root.rglob("manifest.json") if p.parent.name.startswith("run_")})
    rows = [audit_run(r) for r in runs]
    fields = list(rows[0]) if rows else ["run_dir", "valid_for_publication", "reasons"]
    with (analysis / "integrity_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    eligible = all(t["publication_eligible"] for t in trials) if trials else bool(rows) and all(r["valid_for_publication"] for r in rows)
    summary = {"experiment_root": str(root), "run_count": len(rows), "publication_eligible_runs": sum(r["valid_for_publication"] for r in rows), "valid_for_publication": eligible, "failures": [r for r in rows if not r["valid_for_publication"]], "trial_count": len(trials), "publication_eligible_trials": sum(t["publication_eligible"] for t in trials), "trials": trials}
    output = analysis / "integrity_summary.json"; output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = ["# Integrity audit", ""] + [f"- {'PASS' if t['publication_eligible'] else 'FAIL'} {Path(t['trial_dir']).name}: {t['passed_arm_count']}/{t['required_arm_count']} arms passed" for t in trials]
    (analysis / "integrity_report.md").write_text("\n".join(report) + "\n")
    return output
