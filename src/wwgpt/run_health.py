from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

SEVERITIES = ("INFO", "WARNING", "ERROR")


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except Exception:
        return {}


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite_series(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame or frame.empty:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(len(values) and values.map(math.isfinite).all())


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_count(frame: pd.DataFrame, columns: list[str]) -> int:
    keys = [column for column in columns if column in frame]
    return int(frame.duplicated(keys).sum()) if keys and not frame.empty else 0


def _finding(
    findings: list[dict[str, Any]],
    severity: str,
    check: str,
    message: str,
    *,
    value: Any = None,
    threshold: Any = None,
) -> None:
    if severity not in SEVERITIES:
        raise ValueError(f"invalid severity {severity}")
    findings.append(
        {
            "severity": severity,
            "check": check,
            "message": message,
            "value": value,
            "threshold": threshold,
        }
    )


def generate_run_health(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    findings: list[dict[str, Any]] = []
    manifest = _json(run_dir / "manifest.json")
    complete = _json(run_dir / "run_complete.json")
    data = _json(run_dir / "data_manifest.json")
    selected = _json(run_dir / "selected_checkpoint_metrics.json")
    metrics = _csv(run_dir / "metrics.csv")
    alpha = _csv(run_dir / "alpha_measurements.csv")
    traps = _csv(run_dir / "weightwatcher_aggregates.csv")
    measurements = _csv(run_dir / "wwpgd_endpoint_measurements.csv")
    relaxations = _csv(run_dir / "wwpgd_endpoint_relaxation.csv")
    fast_steps = _csv(run_dir / "wwpgd_fast_control_steps.csv")
    diagnostics = _csv(run_dir / "wwpgd_internal_diagnostics.csv")
    projection = _csv(run_dir / "wwpgd_projection.csv")

    for filename in ("manifest.json", "run_complete.json", "metrics.csv", "selected_checkpoint_metrics.json"):
        if not (run_dir / filename).is_file():
            _finding(findings, "ERROR", "required_artifact", f"missing {filename}")

    for column in (
        "train_loss",
        "validation_loss",
        "gradient_norm_before_clip",
        "gradient_norm_after_clip",
        "model_parameter_l2_norm",
        "model_max_abs_parameter",
        "model_nonfinite_parameter_count",
    ):
        if column not in metrics:
            _finding(findings, "ERROR", "metric_schema", f"metrics.csv missing {column}")
        elif not _finite_series(metrics, column):
            _finding(findings, "ERROR", "finite_metrics", f"nonfinite values in {column}")
    if "model_nonfinite_parameter_count" in metrics:
        count = pd.to_numeric(metrics["model_nonfinite_parameter_count"], errors="coerce")
        if (count.fillna(1) != 0).any():
            _finding(findings, "ERROR", "model_finiteness", "model contained nonfinite parameters")
    if "gradient_nonfinite_element_count_before_clip" in metrics:
        count = pd.to_numeric(
            metrics["gradient_nonfinite_element_count_before_clip"], errors="coerce"
        )
        if (count.fillna(1) != 0).any():
            _finding(findings, "ERROR", "gradient_finiteness", "gradient contained nonfinite elements")

    duplicate_specs = {
        "metrics.csv": (metrics, ["evaluation_index"] if "evaluation_index" in metrics else ["step"]),
        "alpha_measurements.csv": (alpha, ["optimizer_step", "layer_name"]),
        "wwpgd_endpoint_measurements.csv": (measurements, ["optimizer_step", "layer_name"]),
        "wwpgd_endpoint_relaxation.csv": (relaxations, ["optimizer_step", "layer_name"]),
        "wwpgd_fast_control_steps.csv": (fast_steps, ["optimizer_step"]),
        "wwpgd_internal_diagnostics.csv": (diagnostics, ["optimizer_step", "measurement_index", "projection_event", "layer_name"]),
        "wwpgd_projection.csv": (projection, ["actual_step", "projection_event", "layer_name"]),
    }
    for filename, (frame, keys) in duplicate_specs.items():
        duplicates = _duplicate_count(frame, keys)
        if duplicates:
            _finding(
                findings, "ERROR", "duplicate_rows", f"{filename} contains duplicate logical rows",
                value=duplicates,
            )

    required_probe = None
    try:
        train_cfg = manifest.get("optimizer_hyperparameters") or {}
        model_cfg = manifest.get("model_config") or {}
        required_probe = (
            int(train_cfg["eval_batches"])
            * int(train_cfg["batch_size"])
            * int(model_cfg["block_size"])
            + 1
        )
    except (KeyError, TypeError, ValueError):
        _finding(findings, "ERROR", "evaluation_capacity", "unable to resolve required probe tokens")
    if required_probe is not None:
        for split, key in (("validation", "validation_tokens"), ("test", "test_tokens")):
            available = data.get(key)
            if available is None:
                _finding(findings, "ERROR", "evaluation_capacity", f"missing {key}")
            elif int(available) < required_probe:
                _finding(
                    findings, "ERROR", "evaluation_capacity",
                    f"insufficient {split} tokens", value=int(available), threshold=required_probe,
                )

    selected_required = (
        "train_loss", "train_perplexity", "train_accuracy",
        "validation_loss", "validation_perplexity", "validation_accuracy",
        "test_loss", "test_perplexity", "test_accuracy",
        "checkpoint_path", "checkpoint_hash", "selection_metric",
    )
    missing_selected = [key for key in selected_required if key not in selected]
    if missing_selected:
        _finding(
            findings, "ERROR", "selected_checkpoint",
            f"selected checkpoint metrics missing fields: {missing_selected}",
        )
    else:
        if selected.get("selection_metric") != "validation_loss":
            _finding(findings, "ERROR", "selected_checkpoint", "test checkpoint was not selected by validation loss")
        for key in (
            "train_loss", "train_perplexity", "train_accuracy",
            "validation_loss", "validation_perplexity", "validation_accuracy",
            "test_loss", "test_perplexity", "test_accuracy",
        ):
            try:
                finite = math.isfinite(float(selected[key]))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                _finding(findings, "ERROR", "selected_checkpoint", f"nonfinite selected metric {key}")
        checkpoint = Path(str(selected.get("checkpoint_path", "")))
        if not checkpoint.is_absolute():
            checkpoint = run_dir / checkpoint
        if not checkpoint.is_file():
            _finding(findings, "ERROR", "selected_checkpoint", "selected checkpoint artifact missing")
        elif _hash(checkpoint) != str(selected.get("checkpoint_hash")):
            _finding(findings, "ERROR", "selected_checkpoint", "selected checkpoint hash mismatch")

    projected_alpha = alpha.copy()
    if not projected_alpha.empty:
        for column in ("valid_for_science", "projected", "included_in_projected_alpha_summary"):
            if column in projected_alpha:
                projected_alpha = projected_alpha[projected_alpha[column].map(_truth)]
        projected_alpha = projected_alpha[
            pd.to_numeric(projected_alpha.get("alpha"), errors="coerce").map(math.isfinite)
        ]
    if projected_alpha.empty:
        _finding(findings, "WARNING", "weightwatcher_alpha", "no valid projected WeightWatcher alpha rows")
    else:
        _finding(
            findings, "INFO", "weightwatcher_alpha",
            "valid projected WeightWatcher alpha rows recorded", value=len(projected_alpha),
        )
    if traps.empty:
        _finding(findings, "WARNING", "weightwatcher_traps", "randomized trap aggregates missing")

    extension = str(manifest.get("extension") or "none")
    if extension == "wwpgd":
        stock_calls = int(
            complete.get("stock_wwpgd_invocation_count", complete.get("wwpgd_call_count", 0)) or 0
        )
        if stock_calls <= 0:
            _finding(findings, "WARNING", "wwpgd_activity", "WWPGD arm made no stock candidate call")
        if stock_calls > 0 and diagnostics.empty:
            _finding(findings, "ERROR", "wwpgd_diagnostics", "stock candidate calls lack diagnostics")
        if not diagnostics.empty:
            modes = sorted(set(diagnostics.get("diagnostics_mode", pd.Series("compatibility", index=diagnostics.index)).fillna("compatibility").astype(str)))
            _finding(findings, "INFO", "wwpgd_diagnostics", f"diagnostic mode(s): {modes}")
            changed = diagnostics.get(
                "candidate_changed", diagnostics.get("changed", pd.Series(False, index=diagnostics.index))
            ).map(_truth)
            movement = pd.to_numeric(
                diagnostics.get(
                    "candidate_relative_frobenius_change",
                    diagnostics.get("original_to_candidate_relative_frobenius_change"),
                ),
                errors="coerce",
            )
            if (changed & (~movement.map(math.isfinite) | ~(movement > 0))).any():
                _finding(findings, "ERROR", "wwpgd_diagnostics", "changed candidate lacks finite positive movement")
            if "compatibility" in modes:
                _finding(
                    findings, "WARNING", "wwpgd_diagnostics",
                    "installed WWPGD exposes compatibility diagnostics only; private midpoint/Cayley/TraceLog fields remain unavailable",
                )
        candidate_count = int(
            diagnostics.get("candidate_changed", pd.Series(dtype=object)).map(_truth).sum()
        ) if not diagnostics.empty else 0
        if candidate_count == 0:
            _finding(findings, "WARNING", "wwpgd_activity", "no stock candidate changed an eligible matrix")

        if measurements.empty and projection.empty:
            _finding(findings, "ERROR", "wwpgd_artifacts", "WWPGD action artifacts missing")
        if not measurements.empty:
            activations = int(
                measurements.get("cache_activated", pd.Series(False, index=measurements.index)).map(_truth).sum()
            )
            if activations == 0:
                _finding(findings, "WARNING", "wwpgd_activity", "no cached endpoint was activated")
            else:
                _finding(findings, "INFO", "wwpgd_activity", "cached endpoints activated", value=activations)
        if not relaxations.empty:
            requested = pd.to_numeric(
                relaxations.get("requested_relative_frobenius_change"), errors="coerce"
            )
            applied = pd.to_numeric(
                relaxations.get("applied_relative_frobenius_change"), errors="coerce"
            )
            per_step_cap = pd.to_numeric(
                relaxations.get("trust_region_limit"), errors="coerce"
            )
            cumulative = pd.to_numeric(
                relaxations.get("cumulative_applied_relative_change_after"), errors="coerce"
            )
            cumulative_cap = pd.to_numeric(
                relaxations.get("max_cumulative_relative_frobenius_change_per_refresh"), errors="coerce"
            )
            tolerance = 1e-9
            if ((applied - per_step_cap) > tolerance).fillna(False).any():
                _finding(findings, "ERROR", "wwpgd_dose", "per-step relative movement exceeded cap")
            if ((cumulative - cumulative_cap) > tolerance).fillna(False).any():
                _finding(findings, "ERROR", "wwpgd_dose", "cumulative refresh movement exceeded cap")
            changed = relaxations.get("changed", pd.Series(False, index=relaxations.index)).map(_truth)
            if not changed.any():
                _finding(findings, "WARNING", "wwpgd_activity", "no fast endpoint relaxation changed a layer")
            saturation = pd.to_numeric(
                relaxations.get("trust_region_scale"), errors="coerce"
            ).lt(1.0 - 1e-12)
            fraction = float(saturation.mean()) if len(saturation) else 0.0
            if fraction > 0.5:
                _finding(
                    findings, "WARNING", "wwpgd_dose",
                    "trust-region clipping occurred on more than half of relaxation rows",
                    value=fraction, threshold=0.5,
                )
            _finding(
                findings, "INFO", "wwpgd_dose",
                "maximum applied relative movement",
                value=float(applied.max()) if applied.notna().any() else None,
            )

    total_elapsed = pd.to_numeric(
        metrics.get("total_elapsed_seconds", metrics.get("elapsed_time")), errors="coerce"
    )
    if len(total_elapsed) and total_elapsed.notna().any():
        elapsed = float(total_elapsed.max())
        overhead_columns = (
            "weightwatcher_measurement_seconds",
            "stock_candidate_generation_seconds",
            "randomized_diagnostic_seconds",
            "fast_relaxation_seconds",
        )
        overhead = 0.0
        for column in overhead_columns:
            if column in metrics:
                values = pd.to_numeric(metrics[column], errors="coerce")
                if values.notna().any():
                    overhead += float(values.max())
        ratio = overhead / max(elapsed, 1e-12)
        if ratio > 0.5:
            _finding(
                findings, "WARNING", "runtime_overhead",
                "WeightWatcher/WWPGD diagnostics consumed more than half of elapsed time",
                value=ratio, threshold=0.5,
            )
        else:
            _finding(findings, "INFO", "runtime_overhead", "diagnostic overhead ratio", value=ratio)

    counts = {severity: sum(row["severity"] == severity for row in findings) for severity in SEVERITIES}
    report = {
        "run_dir": str(run_dir),
        "arm_name": manifest.get("arm_name"),
        "base_optimizer": manifest.get("base_optimizer"),
        "extension": extension,
        "seed": manifest.get("seed"),
        "level": manifest.get("level"),
        "token_multiplier": manifest.get("token_multiplier"),
        "ready_for_analysis": counts["ERROR"] == 0,
        "publication_eligible_from_health": counts["ERROR"] == 0,
        "counts": counts,
        "findings": findings,
    }
    (run_dir / "run_health.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    fields = ["severity", "check", "message", "value", "threshold"]
    with (run_dir / "run_health.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(findings)
    return report


def discover_run_directories(root: Path) -> list[Path]:
    return sorted(
        {
            path.parent
            for path in Path(root).rglob("manifest.json")
            if path.parent.name.startswith("run_")
        }
    )


def generate_experiment_health(root: Path) -> dict[str, Any]:
    root = Path(root)
    reports = [generate_run_health(run) for run in discover_run_directories(root)]
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_dir": report["run_dir"],
            "arm_name": report.get("arm_name"),
            "base_optimizer": report.get("base_optimizer"),
            "extension": report.get("extension"),
            "seed": report.get("seed"),
            "level": report.get("level"),
            "token_multiplier": report.get("token_multiplier"),
            "ready_for_analysis": report["ready_for_analysis"],
            "info_count": report["counts"]["INFO"],
            "warning_count": report["counts"]["WARNING"],
            "error_count": report["counts"]["ERROR"],
        }
        for report in reports
    ]
    pd.DataFrame(rows).to_csv(analysis / "run_health_summary.csv", index=False)
    summary = {
        "experiment_root": str(root),
        "run_count": len(reports),
        "ready_run_count": sum(report["ready_for_analysis"] for report in reports),
        "ready_for_analysis": bool(reports) and all(report["ready_for_analysis"] for report in reports),
        "reports": reports,
    }
    (analysis / "run_health_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    return summary
