from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

SEVERITIES = ("INFO", "WARNING", "ERROR")


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, None
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return {}, f"expected a JSON object, found {type(value).__name__}"
    return value, None


def _json(path: Path) -> dict[str, Any]:
    return _read_json(path)[0]


def _read_csv(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.is_file():
        return pd.DataFrame(), None
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def _csv(path: Path) -> pd.DataFrame:
    return _read_csv(path)[0]


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite_series(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame or frame.empty:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(len(values) and values.map(math.isfinite).all())


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric Series aligned to ``frame`` without scalar fallbacks."""
    if column not in frame:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    values = frame[column]
    if isinstance(values, pd.DataFrame):
        values = values.bfill(axis=1).iloc[:, 0]
    return pd.to_numeric(values, errors="coerce")


def _truth_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if isinstance(values, pd.DataFrame):
        values = values.bfill(axis=1).iloc[:, 0]
    return values.map(_truth)


def _require_columns(
    findings: list[dict[str, Any]],
    filename: str,
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> bool:
    missing = [column for column in columns if column not in frame]
    if missing:
        _finding(
            findings,
            "ERROR",
            "artifact_schema",
            f"{filename} missing required columns: {missing}",
        )
        return False
    return True


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
    json_artifacts = {
        "manifest.json": _read_json(run_dir / "manifest.json"),
        "run_complete.json": _read_json(run_dir / "run_complete.json"),
        "data_manifest.json": _read_json(run_dir / "data_manifest.json"),
        "selected_checkpoint_metrics.json": _read_json(
            run_dir / "selected_checkpoint_metrics.json"
        ),
    }
    manifest = json_artifacts["manifest.json"][0]
    complete = json_artifacts["run_complete.json"][0]
    data = json_artifacts["data_manifest.json"][0]
    selected = json_artifacts["selected_checkpoint_metrics.json"][0]

    csv_artifacts = {
        "metrics.csv": _read_csv(run_dir / "metrics.csv"),
        "alpha_measurements.csv": _read_csv(run_dir / "alpha_measurements.csv"),
        "weightwatcher_aggregates.csv": _read_csv(
            run_dir / "weightwatcher_aggregates.csv"
        ),
        "wwpgd_endpoint_measurements.csv": _read_csv(
            run_dir / "wwpgd_endpoint_measurements.csv"
        ),
        "wwpgd_endpoint_relaxation.csv": _read_csv(
            run_dir / "wwpgd_endpoint_relaxation.csv"
        ),
        "wwpgd_fast_control_steps.csv": _read_csv(
            run_dir / "wwpgd_fast_control_steps.csv"
        ),
        "wwpgd_internal_diagnostics.csv": _read_csv(
            run_dir / "wwpgd_internal_diagnostics.csv"
        ),
        "wwpgd_projection.csv": _read_csv(run_dir / "wwpgd_projection.csv"),
    }
    metrics = csv_artifacts["metrics.csv"][0]
    alpha = csv_artifacts["alpha_measurements.csv"][0]
    traps = csv_artifacts["weightwatcher_aggregates.csv"][0]
    measurements = csv_artifacts["wwpgd_endpoint_measurements.csv"][0]
    relaxations = csv_artifacts["wwpgd_endpoint_relaxation.csv"][0]
    fast_steps = csv_artifacts["wwpgd_fast_control_steps.csv"][0]
    diagnostics = csv_artifacts["wwpgd_internal_diagnostics.csv"][0]
    projection = csv_artifacts["wwpgd_projection.csv"][0]

    for filename, (_value, error) in {**json_artifacts, **csv_artifacts}.items():
        if error:
            _finding(
                findings,
                "ERROR",
                "artifact_parse",
                f"unable to parse {filename}: {error}",
            )

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

    allow_overtraining = bool(manifest.get("allow_overtraining", False))
    overtraining_active = bool(manifest.get("overtraining_active", False))
    if overtraining_active:
        train_cfg = manifest.get("optimizer_hyperparameters") or {}
        if not allow_overtraining:
            _finding(
                findings,
                "ERROR",
                "overtraining_protocol",
                "overtraining is active without explicit allow_overtraining opt-in",
            )
        if manifest.get("training_protocol") != "fixed_corpus_overtraining":
            _finding(
                findings,
                "ERROR",
                "overtraining_protocol",
                "overtraining run is missing the fixed-corpus protocol label",
            )
        if manifest.get("valid_for_scaling_law_fit") is not False:
            _finding(
                findings,
                "ERROR",
                "overtraining_protocol",
                "overtraining run must be excluded from scaling-law fits",
            )
        if train_cfg.get("training_sampling") != "random_window":
            _finding(
                findings,
                "ERROR",
                "overtraining_protocol",
                "overtraining requires random-window corpus revisitation",
            )
        if train_cfg.get("evaluation_sampling") != "fixed_probe":
            _finding(
                findings,
                "ERROR",
                "overtraining_protocol",
                "overtraining requires a fixed validation probe",
            )
        if data.get("sampling_with_replacement") is not True:
            _finding(
                findings,
                "ERROR",
                "overtraining_protocol",
                "run-local data manifest must record sampling with replacement",
            )
        try:
            actual_tokens = int(manifest["realized_train_tokens"])
            nominal_tokens = int(manifest["nominal_realized_train_tokens"])
        except (KeyError, TypeError, ValueError):
            _finding(
                findings,
                "ERROR",
                "overtraining_protocol",
                "unable to resolve nominal and actual overtraining token counts",
            )
        else:
            if actual_tokens <= nominal_tokens:
                _finding(
                    findings,
                    "ERROR",
                    "overtraining_protocol",
                    "overtraining token count does not exceed the nominal budget",
                    value=actual_tokens,
                    threshold=nominal_tokens,
                )
    elif allow_overtraining:
        _finding(
            findings,
            "ERROR",
            "overtraining_protocol",
            "allow_overtraining was enabled but the resolved horizon was not extended",
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
                continue
            try:
                available_tokens = int(available)
            except (TypeError, ValueError):
                _finding(
                    findings,
                    "ERROR",
                    "evaluation_capacity",
                    f"invalid {key}",
                    value=available,
                )
                continue
            if available_tokens < required_probe:
                _finding(
                    findings, "ERROR", "evaluation_capacity",
                    f"insufficient {split} tokens", value=available_tokens, threshold=required_probe,
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
        alpha_schema_valid = _require_columns(
            findings, "alpha_measurements.csv", projected_alpha, ("alpha",)
        )
        for column in ("valid_for_science", "projected", "included_in_projected_alpha_summary"):
            if column in projected_alpha:
                projected_alpha = projected_alpha[_truth_column(projected_alpha, column)]
        if alpha_schema_valid:
            alpha_values = _numeric_column(projected_alpha, "alpha")
            projected_alpha = projected_alpha[alpha_values.map(math.isfinite)]
        else:
            projected_alpha = pd.DataFrame()
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
        raw_stock_calls = complete.get(
            "stock_wwpgd_invocation_count", complete.get("wwpgd_call_count", 0)
        )
        try:
            stock_calls = int(raw_stock_calls or 0)
        except (TypeError, ValueError):
            stock_calls = 0
            _finding(
                findings,
                "ERROR",
                "completion_schema",
                "run_complete.json contains an invalid WWPGD invocation count",
                value=raw_stock_calls,
            )
        if stock_calls <= 0:
            _finding(findings, "WARNING", "wwpgd_activity", "WWPGD arm made no stock candidate call")
        if stock_calls > 0 and diagnostics.empty:
            _finding(findings, "ERROR", "wwpgd_diagnostics", "stock candidate calls lack diagnostics")
        candidate_count = 0
        if not diagnostics.empty:
            modes = sorted(
                set(
                    diagnostics.get(
                        "diagnostics_mode",
                        pd.Series("compatibility", index=diagnostics.index),
                    )
                    .fillna("compatibility")
                    .astype(str)
                )
            )
            _finding(findings, "INFO", "wwpgd_diagnostics", f"diagnostic mode(s): {modes}")
            changed_column = (
                "candidate_changed"
                if "candidate_changed" in diagnostics
                else "changed" if "changed" in diagnostics else None
            )
            movement_column = (
                "candidate_relative_frobenius_change"
                if "candidate_relative_frobenius_change" in diagnostics
                else "original_to_candidate_relative_frobenius_change"
                if "original_to_candidate_relative_frobenius_change" in diagnostics
                else None
            )
            if changed_column is None or movement_column is None:
                missing = []
                if changed_column is None:
                    missing.append("candidate_changed|changed")
                if movement_column is None:
                    missing.append(
                        "candidate_relative_frobenius_change|"
                        "original_to_candidate_relative_frobenius_change"
                    )
                _finding(
                    findings,
                    "ERROR",
                    "artifact_schema",
                    "wwpgd_internal_diagnostics.csv missing required columns: "
                    f"{missing}",
                )
                changed = pd.Series(False, index=diagnostics.index, dtype=bool)
                movement = pd.Series(float("nan"), index=diagnostics.index, dtype=float)
            else:
                changed = _truth_column(diagnostics, changed_column)
                movement = _numeric_column(diagnostics, movement_column)
            if (changed & (~movement.map(math.isfinite) | ~(movement > 0))).any():
                _finding(findings, "ERROR", "wwpgd_diagnostics", "changed candidate lacks finite positive movement")
            candidate_count = int(changed.sum())
            if "compatibility" in modes:
                _finding(
                    findings, "WARNING", "wwpgd_diagnostics",
                    "installed WWPGD exposes compatibility diagnostics only; private midpoint/Cayley/TraceLog fields remain unavailable",
                )
        if candidate_count == 0:
            _finding(findings, "WARNING", "wwpgd_activity", "no stock candidate changed an eligible matrix")

        if measurements.empty and projection.empty:
            _finding(findings, "ERROR", "wwpgd_artifacts", "WWPGD action artifacts missing")
        if not measurements.empty:
            measurement_schema_valid = _require_columns(
                findings,
                "wwpgd_endpoint_measurements.csv",
                measurements,
                ("cache_activated",),
            )
            activations = (
                int(_truth_column(measurements, "cache_activated").sum())
                if measurement_schema_valid
                else 0
            )
            if activations == 0:
                _finding(findings, "WARNING", "wwpgd_activity", "no cached endpoint was activated")
            else:
                _finding(findings, "INFO", "wwpgd_activity", "cached endpoints activated", value=activations)
        if not relaxations.empty:
            relaxation_columns = (
                "requested_relative_frobenius_change",
                "applied_relative_frobenius_change",
                "trust_region_limit",
                "cumulative_applied_relative_change_after",
                "max_cumulative_relative_frobenius_change_per_refresh",
                "trust_region_scale",
                "changed",
            )
            relaxation_schema_valid = _require_columns(
                findings,
                "wwpgd_endpoint_relaxation.csv",
                relaxations,
                relaxation_columns,
            )
            if relaxation_schema_valid:
                applied = _numeric_column(
                    relaxations, "applied_relative_frobenius_change"
                )
                per_step_cap = _numeric_column(relaxations, "trust_region_limit")
                cumulative = _numeric_column(
                    relaxations, "cumulative_applied_relative_change_after"
                )
                cumulative_cap = _numeric_column(
                    relaxations,
                    "max_cumulative_relative_frobenius_change_per_refresh",
                )
                tolerance = 1e-9
                if ((applied - per_step_cap) > tolerance).fillna(False).any():
                    _finding(findings, "ERROR", "wwpgd_dose", "per-step relative movement exceeded cap")
                if ((cumulative - cumulative_cap) > tolerance).fillna(False).any():
                    _finding(findings, "ERROR", "wwpgd_dose", "cumulative refresh movement exceeded cap")
                changed = _truth_column(relaxations, "changed")
                if not changed.any():
                    _finding(findings, "WARNING", "wwpgd_activity", "no fast endpoint relaxation changed a layer")
                saturation = _numeric_column(
                    relaxations, "trust_region_scale"
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

    if "total_elapsed_seconds" in metrics:
        total_elapsed = pd.to_numeric(
            metrics["total_elapsed_seconds"], errors="coerce"
        )
    elif "elapsed_time" in metrics:
        total_elapsed = pd.to_numeric(metrics["elapsed_time"], errors="coerce")
    else:
        total_elapsed = pd.Series(dtype=float)
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


def _run_recency(run: Path) -> tuple[float, str]:
    """Return a deterministic newest-first key for one run directory."""
    try:
        newest = max(
            [run.stat().st_mtime]
            + [path.stat().st_mtime for path in run.rglob("*") if path.is_file()]
        )
    except OSError:
        newest = 0.0
    return newest, run.name


def select_current_run_directories(root: Path) -> list[Path]:
    """Select one current run for each scientific arm identity.

    Local experiments are deliberately append-only. A failed attempt therefore
    remains beside a later successful retry. Health of the experiment must be
    based on the newest completed run for each level/multiplier/seed/arm, while
    still choosing the newest incomplete attempt when no completion exists so a
    genuinely unfinished experiment fails loudly.
    """
    grouped: dict[tuple[Any, ...], list[Path]] = {}
    for run in discover_run_directories(root):
        manifest = _json(run / "manifest.json")
        arm = manifest.get("arm_name") or manifest.get("optimizer") or run.parent.name
        key = (
            manifest.get("level"),
            manifest.get("token_multiplier"),
            manifest.get("seed"),
            str(manifest.get("base_optimizer") or ""),
            str(manifest.get("extension") or ""),
            str(arm),
        )
        # Very old fixtures may not expose a scientific identity. Keep their
        # containing arm directory separate rather than collapsing unrelated runs.
        if not any(value not in (None, "") for value in key):
            key = (str(run.parent.resolve()),)
        grouped.setdefault(key, []).append(run)

    selected: list[Path] = []
    for candidates in grouped.values():
        completed = [
            run for run in candidates if (run / "run_complete.json").is_file()
        ]
        selected.append(max(completed or candidates, key=_run_recency))
    return sorted(selected)


def generate_experiment_health(root: Path) -> dict[str, Any]:
    root = Path(root)
    all_runs = discover_run_directories(root)
    selected_runs = select_current_run_directories(root)
    all_reports_by_run = {
        run: generate_run_health(run)
        for run in all_runs
    }
    reports = [all_reports_by_run[run] for run in selected_runs]
    excluded_runs = [str(run) for run in all_runs if run not in set(selected_runs)]
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
        "total_attempt_count": len(all_runs),
        "excluded_superseded_attempt_count": len(excluded_runs),
        "excluded_superseded_attempts": excluded_runs,
        "ready_run_count": sum(report["ready_for_analysis"] for report in reports),
        "ready_for_analysis": bool(reports) and all(report["ready_for_analysis"] for report in reports),
        "reports": reports,
    }
    (analysis / "run_health_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    return summary
