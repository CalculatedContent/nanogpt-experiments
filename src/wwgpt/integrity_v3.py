"""Schema-aware scientific integrity checks.

Schema-v3 runs are audited strictly. Older result fixtures retain their
historical compatibility contract so repository regression tests do not weaken
or redefine the current scientific schema.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from wwgpt.ww import is_projected_layer

CANONICAL_ARMS = (
    "adamw",
    "adamw_wwpgd",
    "muon",
    "muon_wwpgd",
    "stableadamw",
    "stableadamw_wwpgd",
)
CANONICAL_PAIRS = {
    "adamw": "adamw_wwpgd",
    "muon": "muon_wwpgd",
    "stableadamw": "stableadamw_wwpgd",
}
BASELINE_EXTENSIONS = {"", "none", None}
PRIVATE_DIAGNOSTIC_FIELDS = {
    "k_pl",
    "k_detx",
    "k_star",
    "selected_lambda_threshold",
    "selected_tail_size",
    "trace_log_before",
    "trace_log_target",
    "trace_log_after_cayley_before_retraction",
    "trace_log_after_retraction",
    "trace_log_retraction_residual",
    "trace_log_retraction_absolute_error",
    "trace_log_retraction_relative_error",
    "trace_log_retraction_pass",
    "cayley_raw_ratio_min",
    "cayley_raw_ratio_max",
    "cayley_applied_ratio_min",
    "cayley_applied_ratio_max",
    "cayley_low_clip_count",
    "cayley_high_clip_count",
}


def normalize_arm(value: Any) -> str:
    return (
        str(value or "")
        .lower()
        .replace("-", "_")
        .replace("stable_adamw", "stableadamw")
    )


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
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


def _truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _eligible_action_name(name: Any) -> bool:
    value = str(name)
    return is_projected_layer(value) or (
        value.startswith("blocks.")
        and value.endswith(("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"))
    )


def _fingerprint_without_name(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _fingerprint_without_name(item)
            for key, item in value.items()
            if key not in {"name", "optimizer", "optimizer_name"}
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprint_without_name(item) for item in value]
    return value


def _identity(manifest: dict[str, Any]) -> dict[str, Any]:
    seeds = manifest.get("resolved_stochastic_seeds") or {}
    train = manifest.get("optimizer_hyperparameters") or {}
    return {
        "seed": manifest.get("seed"),
        "initialization_hash": manifest.get("initialization_hash"),
        "model_hash": manifest.get(
            "model_configuration_hash", manifest.get("model_config_hash")
        ),
        "dataset": (
            manifest.get("dataset_name"),
            manifest.get("dataset_config"),
            manifest.get("dataset_revision"),
        ),
        "corpus_hash": manifest.get("corpus_hash", manifest.get("data_hash")),
        "tokenizer_hash": manifest.get("tokenizer_hash"),
        "token_budget": (
            manifest.get("requested_tokens"),
            manifest.get("realized_tokens"),
            manifest.get("target_train_tokens"),
            manifest.get("resolved_optimizer_steps"),
        ),
        "optimizer_fingerprint": _fingerprint_without_name(
            manifest.get("optimizer_fingerprint")
        ),
        "lr_schedule": (
            manifest.get("lr_schedule"),
            manifest.get("scheduler_implementation"),
            manifest.get("resolved_warmup_steps"),
            manifest.get("resolved_lr_decay_steps"),
            manifest.get("min_lr_ratio"),
        ),
        "training_reader_seed": seeds.get(
            "train_reader_seed", manifest.get("training_reader_seed")
        ),
        "evaluation_schedule": (
            manifest.get("evaluation_sampling"),
            manifest.get("evaluation_schedule_version"),
            manifest.get("eval_interval", train.get("eval_interval")),
        ),
        "probe_hashes": (
            manifest.get("training_probe_hash"),
            manifest.get("validation_probe_hash"),
            manifest.get("test_probe_hash"),
        ),
        "analysis_plan_hash": manifest.get(
            "analysis_plan_sha256", manifest.get("analysis_plan_hash")
        ),
        "target_alpha": manifest.get(
            "target_alpha",
            (manifest.get("extension_hyperparameters") or {}).get("target_alpha"),
        ),
        "measurement_schedule": (
            train.get("spectral_interval"),
            (manifest.get("measurement") or {}).get("alpha_interval"),
        ),
    }


def _selected_checkpoint_ok(
    run: Path,
    schema: int,
    metrics: pd.DataFrame,
    complete: dict[str, Any],
) -> tuple[bool, str | None]:
    artifact = run / "selected_checkpoint_metrics.json"
    if schema < 3 and not artifact.is_file():
        has_test = any(
            str(column).startswith("test_") and metrics[column].notna().any()
            for column in metrics.columns
        )
        if not has_test:
            return True, None
        step = complete.get("selected_checkpoint_step", complete.get("best_validation_step"))
        if step is None and "selected_checkpoint_step" in metrics:
            values = metrics["selected_checkpoint_step"].dropna()
            step = values.iloc[-1] if len(values) else None
        if step is None:
            return False, "selected_checkpoint_step_missing_for_test_metrics"
        checkpoint_dir = run / "checkpoints"
        if not checkpoint_dir.is_dir() or not any(checkpoint_dir.iterdir()):
            return False, "selected_checkpoint_artifact_missing_for_test_metrics"
        return True, None

    if not artifact.is_file():
        return False, "selected_checkpoint_metrics_missing"
    try:
        record = json.loads(artifact.read_text())
        required = {
            f"{split}_{metric}"
            for split in ("train", "validation", "test")
            for metric in ("loss", "perplexity", "accuracy")
        }
        required |= {
            "checkpoint_path",
            "checkpoint_hash",
            "selected_step",
            "selection_metric",
            "train_validation_gap",
            "train_test_gap",
            "train_probe_hash",
            "validation_probe_hash",
            "test_probe_hash",
        }
        if not required.issubset(record):
            return False, "selected_checkpoint_metrics_fields_missing"
        if record["selection_metric"] != "validation_loss":
            return False, "checkpoint_selection_not_validation_only"
        checkpoint = Path(record["checkpoint_path"])
        if not checkpoint.is_absolute():
            checkpoint = run / checkpoint
        if not checkpoint.is_file():
            return False, "selected_checkpoint_artifact_missing"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["checkpoint_hash"]:
            return False, "selected_checkpoint_hash_mismatch"
        csv_path = run / "selected_checkpoint_metrics.csv"
        if not csv_path.is_file() or not required.issubset(pd.read_csv(csv_path).columns):
            return False, "selected_checkpoint_json_csv_schema_mismatch"
    except Exception as exc:
        return False, f"selected_checkpoint_integrity_error:{type(exc).__name__}"
    return True, None


def _required_probe_tokens(manifest: dict[str, Any]) -> int | None:
    train = manifest.get("optimizer_hyperparameters") or {}
    model = manifest.get("model_config") or {}
    try:
        return (
            int(train.get("eval_batches"))
            * int(train.get("batch_size"))
            * int(model.get("block_size"))
            + 1
        )
    except (TypeError, ValueError):
        return None


def _audit_split_capacity(
    run: Path,
    manifest: dict[str, Any],
    schema: int,
    reasons: list[str],
) -> None:
    if schema < 3:
        return
    data_path = run / "data_manifest.json"
    if not data_path.is_file():
        reasons.append("data_manifest_missing")
        return
    data, error = _load_json(data_path)
    if error:
        reasons.append(error)
        return
    required = _required_probe_tokens(manifest)
    if required is None:
        reasons.append("evaluation_probe_capacity_unverifiable")
        return
    for split, key in (("validation", "validation_tokens"), ("test", "test_tokens")):
        value = data.get(key)
        if value is None:
            reasons.append(f"{split}_token_count_missing")
        elif int(value) < required:
            reasons.append(f"insufficient_{split}_tokens_for_evaluation")


def _audit_metrics(metrics: pd.DataFrame, schema: int, reasons: list[str]) -> None:
    if metrics.empty:
        reasons.append("metrics_empty")
        return
    if schema < 3:
        return
    for column in ("train_loss", "validation_loss", "gradient_norm_before_clip"):
        if column not in metrics:
            reasons.append(f"metrics_missing_{column}")
            continue
        values = pd.to_numeric(metrics[column], errors="coerce")
        if not values.map(math.isfinite).all():
            reasons.append(f"nonfinite_{column}")


def _audit_legacy_projection_provenance(run: Path, reasons: list[str]) -> None:
    path = run / "wwpgd_projection_spectral.csv"
    if not path.is_file():
        return
    frame, error = _read_csv(path)
    required = {
        "immediate_spectral_source",
        "measurement_valid_for_science",
        "alpha_before",
        "alpha_after",
        "weightwatcher_configuration",
    }
    if error or not required.issubset(frame.columns):
        reasons.append("legacy_or_missing_measured_provenance_fields")
        return
    valid = _truth(frame["measurement_valid_for_science"])
    source_ok = frame["immediate_spectral_source"].eq("weightwatcher_measured")
    alpha_ok = pd.to_numeric(frame["alpha_after"], errors="coerce").map(math.isfinite)
    if (valid & ~(source_ok & alpha_ok)).any():
        reasons.append("invalid_weightwatcher_data_marked_valid")


def _audit_diagnostics(
    run: Path,
    manifest: dict[str, Any],
    complete: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    diagnostics_expected = bool(
        int(manifest.get("wwpgd_diagnostics_schema_version", 0) or 0)
        or (run / "wwpgd_internal_diagnostics.csv").is_file()
    )
    status = {
        "wwpgd_execution_verified": False,
        "diagnostic_mode": "not_declared",
        "native_private_diagnostics_available": False,
        "observable_candidate_movement_verified": False,
        "package_provenance_verified": False,
    }
    if not diagnostics_expected:
        return status

    stock_calls = int(
        complete.get(
            "stock_wwpgd_invocation_count",
            complete.get("wwpgd_call_count", 0),
        )
        or 0
    )
    path = run / "wwpgd_internal_diagnostics.csv"
    if stock_calls > 0 and not path.is_file():
        reasons.append("missing_internal_diagnostics_after_stock_invocation")
        return status
    if not path.is_file():
        return status

    diagnostics, error = _read_csv(path)
    if error or diagnostics.empty:
        reasons.append("missing_or_invalid_internal_diagnostics")
        return status

    modes = set(
        diagnostics.get("diagnostics_mode", pd.Series("compatibility", index=diagnostics.index))
        .fillna("compatibility")
        .astype(str)
    )
    status["diagnostic_mode"] = ",".join(sorted(modes))
    status["native_private_diagnostics_available"] = "native" in modes

    logical = [
        column
        for column in ("optimizer_step", "measurement_index", "projection_event", "layer_name")
        if column in diagnostics
    ]
    if logical and diagnostics.duplicated(logical).any():
        reasons.append("duplicate_internal_diagnostic_rows")
    if "layer_name" not in diagnostics or any(
        not _eligible_action_name(name) for name in diagnostics.get("layer_name", [])
    ):
        reasons.append("internal_diagnostic_layer_ineligible")

    manifest_commit = str(manifest.get("wwpgd_commit") or "")
    row_commits = set(
        diagnostics.get("wwpgd_commit", pd.Series(dtype=str)).dropna().astype(str)
    )
    if manifest_commit and row_commits and row_commits != {manifest_commit}:
        reasons.append("internal_diagnostics_commit_mismatch")
    status["package_provenance_verified"] = bool(
        manifest.get("wwpgd_installed_version")
        or manifest.get("wwpgd_commit")
        or manifest.get("wwpgd_source_url")
    )
    if not status["package_provenance_verified"]:
        reasons.append("wwpgd_package_provenance_missing")

    changed = _truth(
        diagnostics.get(
            "candidate_changed",
            diagnostics.get("changed", pd.Series(False, index=diagnostics.index)),
        )
    )
    movement = pd.to_numeric(
        diagnostics.get(
            "candidate_relative_frobenius_change",
            diagnostics.get(
                "original_to_candidate_relative_frobenius_change",
                pd.Series(math.nan, index=diagnostics.index),
            ),
        ),
        errors="coerce",
    )
    bad_movement = changed & (~movement.map(math.isfinite) | ~(movement > 0))
    if bad_movement.any():
        reasons.append("changed_matrix_without_finite_candidate_movement")
    status["observable_candidate_movement_verified"] = not bad_movement.any()

    compatibility = diagnostics.get(
        "diagnostics_mode", pd.Series("compatibility", index=diagnostics.index)
    ).astype(str).eq("compatibility")
    if compatibility.any():
        unsupported = diagnostics.get(
            "unsupported_internal_fields", pd.Series("", index=diagnostics.index)
        ).fillna("").astype(str).str.strip()
        if (compatibility & unsupported.eq("")).any():
            reasons.append("compatibility_diagnostics_missing_unsupported_fields")
        for field in PRIVATE_DIAGNOSTIC_FIELDS:
            if field in diagnostics and diagnostics.loc[compatibility, field].notna().any():
                reasons.append("compatibility_diagnostics_fabricated_private_values")
                break
        valid = _truth(
            diagnostics.get(
                "valid_observable_diagnostic",
                pd.Series(False, index=diagnostics.index),
            )
        )
        if (compatibility & ~valid).any():
            reasons.append("invalid_compatibility_diagnostic")

    native = diagnostics.get(
        "diagnostics_mode", pd.Series("", index=diagnostics.index)
    ).astype(str).eq("native")
    if native.any():
        if "trace_log_retraction_pass" in diagnostics:
            passed = _truth(diagnostics["trace_log_retraction_pass"])
            projected = diagnostics.get(
                "status", pd.Series("", index=diagnostics.index)
            ).astype(str).eq("projected")
            if (native & projected & ~passed).any():
                reasons.append("trace_log_retraction_failed")
        if "selected_tail_size" in diagnostics and "configured_min_tail" in diagnostics:
            tail = pd.to_numeric(diagnostics["selected_tail_size"], errors="coerce")
            minimum = pd.to_numeric(diagnostics["configured_min_tail"], errors="coerce")
            if (native & (tail < minimum)).any():
                reasons.append("successful_internal_tail_too_small")

    status["wwpgd_execution_verified"] = stock_calls == 0 or len(diagnostics) > 0
    return status


def _discover_arms(layout: Path) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = {}
    for manifest_path in layout.rglob("manifest.json"):
        run = manifest_path.parent
        if not (run.name.startswith("run_") or run == layout):
            continue
        manifest, _ = _load_json(manifest_path)
        arm = normalize_arm(
            manifest.get("arm_name") or manifest.get("optimizer") or run.parent.name
        )
        if not arm:
            continue
        candidates.setdefault(arm, []).append(run)

    def recency(run: Path) -> tuple[int, float, str]:
        complete = int((run / "run_complete.json").is_file())
        try:
            newest = max(
                [run.stat().st_mtime]
                + [path.stat().st_mtime for path in run.rglob("*") if path.is_file()]
            )
        except OSError:
            newest = 0.0
        return complete, newest, run.name

    return {arm: max(runs, key=recency) for arm, runs in candidates.items()}


def audit_arm(run: Path, required_arm: str | None = None) -> dict[str, Any]:
    run = Path(run)
    reasons: list[str] = []
    manifest, error = (
        _load_json(run / "manifest.json")
        if (run / "manifest.json").is_file()
        else ({}, "missing_manifest")
    )
    if error:
        reasons.append(error)
    complete, error = (
        _load_json(run / "run_complete.json")
        if (run / "run_complete.json").is_file()
        else ({}, "run_incomplete")
    )
    if error:
        reasons.append(error)
    metrics, error = (
        _read_csv(run / "metrics.csv")
        if (run / "metrics.csv").is_file()
        else (pd.DataFrame(), "missing_metrics")
    )
    if error:
        reasons.append(error)

    schema = int(manifest.get("scientific_schema_version", 0) or 0)
    arm = normalize_arm(
        manifest.get("arm_name") or manifest.get("optimizer") or run.parent.name
    )
    required = normalize_arm(required_arm)
    if required and arm != required:
        reasons.append("arm_name_mismatch")
    base = normalize_arm(manifest.get("base_optimizer") or arm.removesuffix("_wwpgd"))
    extension = manifest.get("extension") or (
        "wwpgd" if arm.endswith("_wwpgd") else "none"
    )
    if not manifest.get("valid_for_science", False):
        reasons.append("fixture_or_invalid_for_science")
    if schema >= 3:
        target = manifest.get(
            "target_alpha",
            (manifest.get("extension_hyperparameters") or {}).get("target_alpha"),
        )
        if target is None:
            reasons.append("target_alpha_missing")
        user_config_sections = (
            manifest.get("resolved_config"),
            manifest.get("config"),
            manifest.get("extension_hyperparameters"),
            manifest.get("wwpgd_adaptive_config"),
        )
        if any(_contains_key(section, "q") for section in user_config_sections):
            reasons.append("user_configured_q_exposed")

    _audit_metrics(metrics, schema, reasons)
    _audit_split_capacity(run, manifest, schema, reasons)
    selected_ok, selected_reason = _selected_checkpoint_ok(run, schema, metrics, complete)
    if not selected_ok and selected_reason:
        reasons.append(selected_reason)
    _audit_legacy_projection_provenance(run, reasons)

    diagnostics_status: dict[str, Any] = {}
    if extension == "wwpgd":
        if not manifest.get("wwpgd_implementation") or not manifest.get("wwpgd_commit"):
            reasons.append("missing_resolved_wwpgd_metadata")
        cached = (
            manifest.get("adapter_mode") == "cached_endpoint_relaxation_v1"
            or manifest.get("projection_schedule_type")
            == "cached_endpoint_measurement_and_fast_apply"
        )
        required_files = (
            ("wwpgd_endpoint_measurements.csv", "wwpgd_fast_control_steps.csv")
            if cached
            else ("wwpgd_projection.csv",)
        )
        frames: dict[str, pd.DataFrame] = {}
        for filename in required_files:
            path = run / filename
            frame, file_error = _read_csv(path) if path.is_file() else (
                pd.DataFrame(),
                f"missing:{filename}",
            )
            if file_error:
                reasons.append(f"missing_or_invalid_extension_artifact:{filename}")
            frames[filename] = frame
        action_frames = [
            frame
            for frame in frames.values()
            if not frame.empty and "layer_name" in frame
        ]
        if action_frames:
            actions = pd.concat(action_frames, ignore_index=True)
            if any(not _eligible_action_name(name) for name in actions["layer_name"]):
                reasons.append("extension_action_names_ineligible_matrix")
        elif required_files:
            reasons.append("extension_action_names_ineligible_matrix")

        if cached and schema >= 3:
            measured = frames.get("wwpgd_endpoint_measurements.csv", pd.DataFrame())
            fast = frames.get("wwpgd_fast_control_steps.csv", pd.DataFrame())
            expected_measurements = sorted(
                int(step) for step in manifest.get("expected_endpoint_measurement_steps", [])
            )
            actual_measurements = sorted(
                set(
                    pd.to_numeric(
                        measured.get("optimizer_step", pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    .dropna()
                    .astype(int)
                )
            )
            expected_fast = sorted(
                int(step) for step in manifest.get("expected_fast_apply_steps", [])
            )
            actual_fast = (
                pd.to_numeric(
                    fast.get("optimizer_step", pd.Series(dtype=float)), errors="coerce"
                )
                .dropna()
                .astype(int)
                .tolist()
            )
            if expected_measurements and actual_measurements != expected_measurements:
                reasons.append("slow_measurement_schedule_mismatch")
            if expected_fast and (
                sorted(actual_fast) != expected_fast or len(actual_fast) != len(set(actual_fast))
            ):
                reasons.append("fast_relaxation_schedule_mismatch")
        elif not cached:
            if int(complete.get("wwpgd_call_count", 0) or 0) <= 0:
                reasons.append("missing_wwpgd_call_count")
            if int(complete.get("projected_matrix_count", 0) or 0) <= 0:
                reasons.append("missing_projected_matrix_count")

        diagnostics_status = _audit_diagnostics(run, manifest, complete, reasons)
    elif extension not in BASELINE_EXTENSIONS:
        reasons.append("unknown_extension")

    confirmatory = manifest.get("analysis_mode") == "confirmatory"
    if confirmatory and not manifest.get(
        "analysis_plan_sha256", manifest.get("analysis_plan_hash")
    ):
        reasons.append("confirmatory_analysis_plan_hash_missing")

    return {
        "arm_name": arm,
        "base_optimizer": base,
        "extension": extension,
        "run_dir": str(run),
        "passed": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "identity": _identity(manifest),
        **diagnostics_status,
    }


def audit_trial(layout: Path) -> dict[str, Any]:
    layout = Path(layout)
    discovered = _discover_arms(layout)
    pair_layout = (layout / "pair_manifest.json").is_file() or layout.name.startswith("pair_")
    if pair_layout:
        pair_manifest, _ = (
            _load_json(layout / "pair_manifest.json")
            if (layout / "pair_manifest.json").is_file()
            else ({}, None)
        )
        base = normalize_arm(
            pair_manifest.get("base_optimizer")
            or next(
                (arm.removesuffix("_wwpgd") for arm in discovered if arm.endswith("_wwpgd")),
                "",
            )
        )
        expected = (base, f"{base}_wwpgd") if base else tuple()
    else:
        expected = CANONICAL_ARMS

    arms: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    for arm in expected:
        run = discovered.get(arm)
        record = (
            audit_arm(run, arm)
            if run
            else {
                "arm_name": arm,
                "passed": False,
                "reasons": ["missing_arm"],
                "run_dir": None,
                "identity": {},
            }
        )
        arms[arm] = record
        if not record["passed"]:
            reasons.append(f"{arm}:" + ",".join(record["reasons"]))

    pairs = (
        [(expected[0], expected[1])]
        if pair_layout and len(expected) == 2
        else [(base, wwpgd) for base, wwpgd in CANONICAL_PAIRS.items() if base in arms]
    )
    for baseline, wwpgd in pairs:
        left = arms[baseline]["identity"]
        right = arms[wwpgd]["identity"]
        for key in left:
            if left.get(key) != right.get(key):
                reason = (
                    "base_optimizer_fingerprint_mismatch"
                    if key == "optimizer_fingerprint"
                    else f"{key}_mismatch"
                )
                reasons.append(f"{baseline}/{wwpgd}:{reason}")

    if not pair_layout:
        legacy_names = {
            "corpus_hash": "data_hash",
            "model_hash": "model_configuration_hash",
        }
        for key in ("corpus_hash", "tokenizer_hash", "model_hash", "token_budget"):
            values = {
                json.dumps(record["identity"].get(key), sort_keys=True, default=str)
                for record in arms.values()
                if record.get("identity")
            }
            if len(values) > 1:
                reasons.append(f"all_arms:{legacy_names.get(key, key)}_mismatch")

    return {
        "trial_dir": str(layout),
        "layout": "pair" if pair_layout else "trial",
        "required_arm_count": len(expected),
        "passed_arm_count": sum(record["passed"] for record in arms.values()),
        "publication_eligible": bool(expected) and not reasons,
        "reasons": reasons,
        "arms": arms,
    }


def audit_run(run: Path) -> dict[str, Any]:
    record = audit_arm(run)
    return {
        "run_dir": str(run),
        "valid_for_publication": record["passed"],
        "reasons": ";".join(record["reasons"]),
    }


def _layout_recency(layout: Path) -> tuple[float, str]:
    try:
        newest = max(
            [layout.stat().st_mtime]
            + [path.stat().st_mtime for path in layout.rglob("*") if path.is_file()]
        )
    except OSError:
        newest = 0.0
    return newest, layout.name


def _layout_identity(layout: Path) -> tuple[Any, ...]:
    """Return the scientific identity used to supersede failed layout retries."""
    kind = "pair" if (layout / "pair_manifest.json").is_file() else "trial"
    manifest_path = layout / f"{kind}_manifest.json"
    layout_manifest, _ = _load_json(manifest_path)
    arms = _discover_arms(layout)
    run_manifest: dict[str, Any] = {}
    if arms:
        run_manifest, _ = _load_json(next(iter(arms.values())) / "manifest.json")
    level = layout_manifest.get("level", run_manifest.get("level"))
    multiplier = layout_manifest.get(
        "token_multiplier", run_manifest.get("token_multiplier")
    )
    seed = layout_manifest.get(
        "seed", layout_manifest.get("scientific_seed", run_manifest.get("seed"))
    )
    base = normalize_arm(
        layout_manifest.get("base_optimizer")
        or run_manifest.get("base_optimizer")
        or ""
    )
    if level is None and multiplier is None and seed is None and not base:
        return kind, str(layout.resolve())
    return kind, level, multiplier, seed, base


def _select_current_trial_audits(
    layouts: list[Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for layout in layouts:
        record = audit_trial(layout)
        grouped.setdefault(_layout_identity(layout), []).append(record)

    selected: list[dict[str, Any]] = []
    excluded: list[str] = []
    for candidates in grouped.values():
        eligible = [record for record in candidates if record["publication_eligible"]]
        chosen = max(
            eligible or candidates,
            key=lambda record: _layout_recency(Path(record["trial_dir"])),
        )
        selected.append(chosen)
        excluded.extend(
            record["trial_dir"] for record in candidates if record is not chosen
        )
    return sorted(selected, key=lambda record: record["trial_dir"]), sorted(excluded)


def audit_experiment(root: Path) -> Path:
    root = Path(root)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    layouts = sorted(
        {
            path.parent
            for name in ("trial_manifest.json", "pair_manifest.json")
            for path in root.rglob(name)
        }
    )
    trials, excluded_layouts = _select_current_trial_audits(layouts)
    if trials:
        runs = sorted(
            {
                Path(arm["run_dir"])
                for trial in trials
                for arm in trial["arms"].values()
                if arm.get("run_dir")
            }
        )
    else:
        runs = sorted(
            {
                path.parent
                for path in root.rglob("manifest.json")
                if path.parent.name.startswith("run_")
                or path.parent.name.startswith("run_legacy")
            }
        )
    rows = [audit_run(run) for run in runs]
    fields = list(rows[0]) if rows else [
        "run_dir",
        "valid_for_publication",
        "reasons",
    ]
    with (analysis / "integrity_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    eligible = (
        all(trial["publication_eligible"] for trial in trials)
        if trials
        else bool(rows) and all(row["valid_for_publication"] for row in rows)
    )
    summary = {
        "experiment_root": str(root),
        "run_count": len(rows),
        "publication_eligible_runs": sum(row["valid_for_publication"] for row in rows),
        "valid_for_publication": eligible,
        "failures": [row for row in rows if not row["valid_for_publication"]],
        "trial_count": len(trials),
        "total_layout_attempt_count": len(layouts),
        "excluded_superseded_layout_count": len(excluded_layouts),
        "excluded_superseded_layouts": excluded_layouts,
        "publication_eligible_trials": sum(
            trial["publication_eligible"] for trial in trials
        ),
        "trials": trials,
    }
    output = analysis / "integrity_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = ["# Integrity audit", ""] + [
        f"- {'PASS' if trial['publication_eligible'] else 'FAIL'} "
        f"{Path(trial['trial_dir']).name}: "
        f"{trial['passed_arm_count']}/{trial['required_arm_count']} arms passed"
        for trial in trials
    ]
    (analysis / "integrity_report.md").write_text("\n".join(report) + "\n")
    return output


__all__ = [
    "BASELINE_EXTENSIONS",
    "CANONICAL_ARMS",
    "CANONICAL_PAIRS",
    "audit_arm",
    "audit_experiment",
    "audit_run",
    "audit_trial",
    "normalize_arm",
]
