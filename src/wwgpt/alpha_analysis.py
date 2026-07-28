"""Target-alpha trajectory analysis sourced only from alpha_measurements.csv."""
from __future__ import annotations

import fnmatch
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wwgpt.adaptive_wwpgd import matrix_type
from wwgpt.ww import alpha_measurement_exclusion_reason

KEYS = [
    "pair_id",
    "level",
    "token_multiplier",
    "arm_name",
    "seed",
    "optimizer_step",
    "tokens_seen",
]
METRICS = ["median_alpha", "mean_alpha", "median_absolute_alpha_error",
           "mean_absolute_alpha_error", "maximum_absolute_alpha_error",
           "fraction_inside_configured_target_deadband", "fraction_above_target_band",
           "fraction_below_target_band", "valid_layer_count", "excluded_layer_count"]


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"immutable run manifest has no valid {name}") from None
    if not math.isfinite(result):
        raise ValueError(f"immutable run manifest has no valid {name}")
    return result


def _run_configuration(manifest: dict[str, Any]) -> tuple[float, int, float | None, dict[str, Any]]:
    # Run manifests are the immutable, pre-observation source of truth.  In
    # particular, alpha_measurements.csv is never consulted for the target.
    ext = manifest.get("extension_hyperparameters", {})
    target = manifest.get("target_alpha", ext.get("target_alpha"))
    adaptive = manifest.get("wwpgd_adaptive_config", ext.get("adaptive", {})) or {}
    min_tail = manifest.get("min_tail", ext.get("min_tail"))
    if min_tail is None:
        raise ValueError("immutable run manifest has no min_tail")
    max_d = adaptive.get("max_D")
    return _finite(target, "target_alpha"), int(min_tail), None if max_d is None else _finite(max_d, "max_D"), adaptive


def _layer_config(adaptive: dict[str, Any], layer: str) -> tuple[float, float, float | None]:
    """Resolve only quality/deadband fields with controller override precedence."""
    above = dict(adaptive.get("above_target", {})); below = dict(adaptive.get("below_target", {}))
    max_d = adaptive.get("max_D")
    def apply(override: dict[str, Any]) -> None:
        nonlocal max_d
        if override.get("max_D") is not None: max_d = override["max_D"]
        if override.get("above_target"): above.update(override["above_target"])
        if override.get("below_target"): below.update(override["below_target"])
    mt = matrix_type(layer)
    apply((adaptive.get("matrix_type_overrides") or {}).get(mt, {}))
    overrides = adaptive.get("layer_overrides") or {}
    patterns = [p for p in overrides if any(c in p for c in "*?[") and fnmatch.fnmatchcase(layer, p)]
    for pattern in sorted(patterns, key=lambda p: (sum(c not in "*?[]" for c in p), len(p), p)):
        apply(overrides[pattern])
    apply(overrides.get(layer, {}))
    # Legacy one-sided configurations retain their explicitly configured band.
    above_dead = above.get("deadband", adaptive.get("deadband_above_target", 0.0))
    below_dead = below.get("deadband", 0.0)
    return float(above_dead), float(below_dead), None if max_d is None else float(max_d)


def _summarize(group: pd.DataFrame) -> dict[str, Any]:
    valid = group[group["alpha_valid"]]
    alpha = pd.to_numeric(valid["alpha"], errors="coerce")
    error = (alpha - valid["target_alpha"]).abs()
    return {
        "target_alpha": group["target_alpha"].iloc[0],
        "median_alpha": alpha.median(), "mean_alpha": alpha.mean(),
        "median_absolute_alpha_error": error.median(),
        "mean_absolute_alpha_error": error.mean(),
        "maximum_absolute_alpha_error": error.max(),
        "fraction_inside_configured_target_deadband": valid["inside_band"].mean(),
        "fraction_above_target_band": valid["above_band"].mean(),
        "fraction_below_target_band": valid["below_band"].mean(),
        "valid_layer_count": int(len(valid)), "excluded_layer_count": int(len(group) - len(valid)),
    }


def prepare_alpha_measurements(run_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    target, min_tail, default_max_d, adaptive = _run_configuration(manifest)
    frame = pd.read_csv(run_dir / "alpha_measurements.csv")
    if frame.empty: return frame
    frame = frame.copy()
    frame["pair_id"] = frame.get("pair_id", manifest.get("pair_id", ""))
    frame["level"] = frame.get("level", manifest.get("level"))
    frame["token_multiplier"] = frame.get(
        "token_multiplier", manifest.get("token_multiplier")
    )
    frame["arm_name"] = frame.get("arm_name", manifest.get("arm_name", manifest.get("optimizer_name", "")))
    frame["seed"] = frame.get("seed", manifest.get("seed"))
    frame["target_alpha"] = target
    reasons=[]; inside=[]; above=[]; below=[]
    for _, row in frame.iterrows():
        layer = str(row.get("layer_name", "")); hi, lo, layer_max_d = _layer_config(adaptive, layer)
        reason = alpha_measurement_exclusion_reason(row, max_D=layer_max_d if layer_max_d is not None else default_max_d,
                                                    min_tail=min_tail, require_projected=True)
        value = pd.to_numeric(pd.Series([row.get("alpha")]), errors="coerce").iloc[0]
        reasons.append(reason); inside.append(not reason and target - lo <= value <= target + hi)
        above.append(not reason and value > target + hi); below.append(not reason and value < target - lo)
    frame["alpha_exclusion_reason"] = reasons; frame["alpha_valid"] = frame["alpha_exclusion_reason"].eq("")
    frame["inside_band"] = inside; frame["above_band"] = above; frame["below_band"] = below
    return frame


def _group_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows=[]
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(values, tuple): values=(values,)
        rows.append(dict(zip(keys, values)) | _summarize(group))
    return pd.DataFrame(rows)


def analyze_alpha_trajectories(runs: list[Any], output_dir: Path) -> None:
    frames=[]
    for run in runs:
        path = Path(run.run_dir if hasattr(run, "run_dir") else run["run_dir"])
        manifest = run.manifest if hasattr(run, "manifest") else run["manifest"]
        if (path / "alpha_measurements.csv").exists(): frames.append(prepare_alpha_measurements(path, manifest))
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    if all_rows.empty:
        pd.DataFrame(columns=KEYS + METRICS).to_csv(output_dir / "alpha_summary_by_step.csv", index=False)
        pd.DataFrame(columns=KEYS + ["matrix_type"] + METRICS).to_csv(output_dir / "alpha_summary_by_matrix_type.csv", index=False)
        pd.DataFrame(columns=KEYS + ["block"] + METRICS).to_csv(output_dir / "alpha_summary_by_transformer_block.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "paired_alpha_distance_differences.csv", index=False)
        for name in ("alpha_distance_trajectories.png", "fraction_near_target.png"):
            fig, ax=plt.subplots(); ax.text(.5,.5,"No valid alpha measurements",ha="center"); fig.savefig(output_dir/name); plt.close(fig)
        return
    step = _group_summary(all_rows, KEYS)
    step.to_csv(output_dir / "alpha_summary_by_step.csv", index=False)
    by_type = _group_summary(all_rows, KEYS + ["matrix_type"])
    by_type.to_csv(output_dir / "alpha_summary_by_matrix_type.csv", index=False)
    _group_summary(all_rows, KEYS + ["block"]).to_csv(output_dir / "alpha_summary_by_transformer_block.csv", index=False)
    step["pairing_base_optimizer"] = step.arm_name.astype(str).str.replace("_wwpgd(?:_reference)?$", "", regex=True)
    base = step[~step.arm_name.astype(str).str.contains("wwpgd")]
    treated = step[step.arm_name.astype(str).str.contains("wwpgd")]
    paired = treated.merge(
        base,
        on=[
            "pair_id",
            "level",
            "token_multiplier",
            "pairing_base_optimizer",
            "seed",
            "optimizer_step",
            "tokens_seen",
        ],
        suffixes=("_wwpgd", "_baseline"),
    )
    if len(paired):
        paired["paired_median_alpha_distance_difference"] = paired["median_absolute_alpha_error_wwpgd"] - paired["median_absolute_alpha_error_baseline"]
        paired["paired_mean_alpha_distance_difference"] = paired["mean_absolute_alpha_error_wwpgd"] - paired["mean_absolute_alpha_error_baseline"]
    paired.to_csv(output_dir / "paired_alpha_distance_differences.csv", index=False)
    for metric, filename, ylabel in [("median_absolute_alpha_error", "alpha_distance_trajectories.png", "Median |alpha - target alpha|"), ("fraction_inside_configured_target_deadband", "fraction_near_target.png", "Fraction inside configured target deadband")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (level, multiplier, arm), group in step.groupby(
            ["level", "token_multiplier", "arm_name"], dropna=False
        ):
            curve=group.groupby("tokens_seen")[metric].mean(); ax.plot(curve.index, curve.values, marker="o", label=f"L{level} M{multiplier} {arm}")
        ax.set(xlabel="Tokens seen", ylabel=ylabel); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / filename, dpi=160); plt.close(fig)
