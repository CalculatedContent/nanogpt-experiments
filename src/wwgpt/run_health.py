"""Machine-readable operational and scientific health checks for completed runs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _check(rows: list[dict[str, Any]], severity: str, code: str, passed: bool, message: str, **details: Any) -> None:
    rows.append(
        {
            "severity": severity if not passed else "INFO",
            "code": code,
            "passed": bool(passed),
            "message": message,
            "details": json.dumps(details, sort_keys=True, default=str),
        }
    )


def evaluate_run_health(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = Path(run_dir)
    manifest = _load_json(run_dir / "manifest.json")
    complete = _load_json(run_dir / "run_complete.json")
    selected = _load_json(run_dir / "selected_checkpoint_metrics.json")
    data = _load_json(run_dir / "data_manifest.json")
    metrics = _load_csv(run_dir / "metrics.csv")
    alpha = _load_csv(run_dir / "alpha_measurements.csv")
    internal = _load_csv(run_dir / "wwpgd_internal_diagnostics.csv")
    measurement = _load_csv(run_dir / "wwpgd_endpoint_measurements.csv")
    relaxation = _load_csv(run_dir / "wwpgd_endpoint_relaxation.csv")
    rows: list[dict[str, Any]] = []

    _check(rows, "ERROR", "manifest_present", bool(manifest), "manifest.json is readable")
    _check(rows, "ERROR", "run_complete_present", bool(complete), "run_complete.json is readable")
    _check(rows, "ERROR", "metrics_present", not metrics.empty, "metrics.csv contains evaluation records")
    _check(rows, "ERROR", "selected_checkpoint_present", bool(selected), "selected checkpoint metrics are readable")

    for column in ("train_loss", "validation_loss", "gradient_norm_before_clip"):
        present = column in metrics
        finite = bool(present and pd.to_numeric(metrics[column], errors="coerce").map(math.isfinite).all())
        _check(rows, "ERROR", f"finite_{column}", finite, f"{column} is present and finite")

    required_selected = (
        "train_loss",
        "validation_loss",
        "test_loss",
        "train_perplexity",
        "validation_perplexity",
        "test_perplexity",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "checkpoint_hash",
        "selection_metric",
    )
    selected_ok = all(key in selected for key in required_selected)
    _check(rows, "ERROR", "selected_metric_schema", selected_ok, "selected checkpoint contains train/validation/test loss, perplexity, accuracy, and provenance")
    if selected_ok:
        numeric_keys = [key for key in required_selected if key not in {"checkpoint_hash", "selection_metric"}]
        finite_selected = all(math.isfinite(float(selected[key])) for key in numeric_keys)
        _check(rows, "ERROR", "selected_metrics_finite", finite_selected, "selected checkpoint numerical metrics are finite")
        _check(rows, "ERROR", "validation_only_selection", selected.get("selection_metric") == "validation_loss", "checkpoint selection uses validation loss only")

    train_cfg = manifest.get("optimizer_hyperparameters") or {}
    model_cfg = manifest.get("model_config") or {}
    try:
        required_probe_tokens = int(train_cfg["eval_batches"]) * int(train_cfg["batch_size"]) * int(model_cfg["block_size"]) + 1
    except (KeyError, TypeError, ValueError):
        required_probe_tokens = None
    for split, key in (("validation", "validation_tokens"), ("test", "test_tokens")):
        count = data.get(key)
        passed = required_probe_tokens is not None and count is not None and int(count) >= required_probe_tokens
        _check(
            rows,
            "ERROR",
            f"{split}_probe_capacity",
            passed,
            f"{split} split supports the configured fixed evaluation probe",
            required=required_probe_tokens,
            actual=count,
        )

    projected = alpha.get("projected", pd.Series(False, index=alpha.index))
    valid = alpha.get("valid_for_science", pd.Series(False, index=alpha.index))
    included = alpha.get("included_in_projected_alpha_summary", pd.Series(False, index=alpha.index))
    valid_alpha = _truth(projected) & _truth(valid) & _truth(included) if not alpha.empty else pd.Series(dtype=bool)
    _check(rows, "WARNING", "valid_weightwatcher_alpha", bool(valid_alpha.any()), "at least one eligible projected layer has a valid WeightWatcher alpha")
    if len(alpha):
        projected_count = int(_truth(projected).sum())
        valid_count = int(valid_alpha.sum())
        invalid_fraction = 1.0 - valid_count / max(projected_count, 1)
        _check(rows, "WARNING", "alpha_invalid_fraction", invalid_fraction <= 0.75, "no more than 75% of projected alpha rows are invalid or excluded", invalid_fraction=invalid_fraction)

    extension = str(manifest.get("extension") or "none")
    if extension == "wwpgd":
        stock_calls = int(complete.get("stock_wwpgd_invocation_count", complete.get("wwpgd_call_count", 0)) or 0)
        _check(rows, "WARNING", "stock_candidate_invoked", stock_calls > 0, "at least one pip-installed WWPGD candidate generation occurred", stock_calls=stock_calls)
        _check(rows, "ERROR", "wwpgd_diagnostics_present", stock_calls == 0 or not internal.empty, "WWPGD diagnostics exist whenever a stock candidate was invoked")
        if not internal.empty:
            modes = sorted(set(internal.get("diagnostics_mode", pd.Series("compatibility", index=internal.index)).fillna("compatibility").astype(str)))
            compatibility = internal[internal.get("diagnostics_mode", pd.Series("compatibility", index=internal.index)).astype(str).eq("compatibility")]
            private = ["k_pl", "k_detx", "k_star", "selected_tail_size", "trace_log_retraction_residual", "cayley_raw_ratio_min"]
            fabricated = any(field in compatibility and compatibility[field].notna().any() for field in private)
            _check(rows, "ERROR", "compatibility_diagnostics_honest", not fabricated, "compatibility diagnostics do not fabricate private WWPGD internals", modes=modes)
            movement = pd.to_numeric(internal.get("original_to_candidate_relative_frobenius_change"), errors="coerce")
            changed = _truth(internal.get("candidate_changed", pd.Series(False, index=internal.index)))
            movement_ok = bool((~changed | np.isfinite(movement)).all())
            _check(rows, "ERROR", "candidate_movement_observed", movement_ok, "changed candidates have finite observed relative movement")
            _check(rows, "WARNING", "native_diagnostics_available", "native" in modes, "installed WWPGD exposes private internal diagnostics; compatibility mode remains valid when false", modes=modes)
        activated = _truth(measurement.get("cache_activated", pd.Series(False, index=measurement.index))) if not measurement.empty else pd.Series(dtype=bool)
        _check(rows, "WARNING", "endpoint_activation", bool(activated.any()), "at least one cached endpoint was activated")
        changed = _truth(relaxation.get("changed", pd.Series(False, index=relaxation.index))) if not relaxation.empty else pd.Series(dtype=bool)
        _check(rows, "WARNING", "fast_wwpgd_change", bool(changed.any()), "at least one fast endpoint relaxation changed an eligible matrix")
        if not relaxation.empty:
            applied = pd.to_numeric(relaxation.get("applied_relative_frobenius_change"), errors="coerce")
            per_step = pd.to_numeric(relaxation.get("trust_region_limit"), errors="coerce")
            per_step_ok = bool(((applied - per_step) <= 1e-10).fillna(True).all())
            _check(rows, "ERROR", "per_step_dose_cap", per_step_ok, "all fast updates respect the per-step relative Frobenius cap")
            cumulative = pd.to_numeric(relaxation.get("cumulative_applied_relative_change_after"), errors="coerce")
            cumulative_limit = pd.to_numeric(relaxation.get("max_cumulative_relative_frobenius_change_per_refresh"), errors="coerce")
            cumulative_ok = bool(((cumulative - cumulative_limit) <= 1e-10).fillna(True).all())
            _check(rows, "ERROR", "cumulative_dose_cap", cumulative_ok, "all refresh windows respect the cumulative relative Frobenius cap")
            scale = pd.to_numeric(relaxation.get("trust_region_scale"), errors="coerce")
            saturation = float((scale < 1.0 - 1e-12).mean()) if scale.notna().any() else math.nan
            _check(rows, "WARNING", "trust_region_saturation", not math.isfinite(saturation) or saturation <= 0.5, "trust-region clipping is not active on more than half of fast decisions", saturation_fraction=saturation)

    errors = [row for row in rows if row["severity"] == "ERROR" and not row["passed"]]
    warnings = [row for row in rows if row["severity"] == "WARNING" and not row["passed"]]
    summary = {
        "run_dir": str(run_dir),
        "level": manifest.get("level"),
        "seed": manifest.get("seed"),
        "base_optimizer": manifest.get("base_optimizer"),
        "extension": extension,
        "status": "ERROR" if errors else "WARNING" if warnings else "PASS",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "publication_eligible_from_health": not errors,
        "error_codes": [row["code"] for row in errors],
        "warning_codes": [row["code"] for row in warnings],
    }
    return pd.DataFrame(rows), summary


def write_run_health(run_dir: Path) -> dict[str, Any]:
    checks, summary = evaluate_run_health(Path(run_dir))
    checks.to_csv(Path(run_dir) / "run_health.csv", index=False)
    (Path(run_dir) / "run_health.json").write_text(
        json.dumps({"summary": summary, "checks": checks.to_dict("records")}, indent=2, sort_keys=True, default=str) + "\n"
    )
    return summary


def write_experiment_health(results_root: Path, output_dir: Path | None = None) -> pd.DataFrame:
    results_root = Path(results_root)
    output_dir = Path(output_dir or results_root / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for complete in sorted(results_root.rglob("run_complete.json")):
        summary = write_run_health(complete.parent)
        summaries.append(summary)
    frame = pd.DataFrame(summaries)
    frame.to_csv(output_dir / "experiment_run_health.csv", index=False)
    (output_dir / "experiment_run_health.json").write_text(
        json.dumps(frame.to_dict("records"), indent=2, sort_keys=True, default=str) + "\n"
    )
    return frame
