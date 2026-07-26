"""Cross-run WWPGD diagnostics using only recorded package and adapter outputs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wwgpt.notebook_support import (
    discover_completed_runs,
    filter_runs,
    load_wwpgd_artifact,
    load_wwpgd_internal_diagnostics,
    normalize_arm,
)

PRIVATE_FIELDS = {
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
    "cayley_raw_ratio_min",
    "cayley_raw_ratio_max",
    "cayley_applied_ratio_min",
    "cayley_applied_ratio_max",
    "cayley_low_clip_count",
    "cayley_high_clip_count",
}


def _truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _metadata(frame: pd.DataFrame, row: Any) -> pd.DataFrame:
    out = frame.copy()
    out["seed"] = row.seed
    out["level"] = row.level
    out["token_multiplier"] = row.token_multiplier
    out["base_optimizer"] = row.base_optimizer
    out["arm"] = row.arm
    out["run_dir"] = str(row.run_dir)
    return out


def collect_wwpgd_diagnostics(
    results_root: Path,
    *,
    level: int | None = None,
    token_multiplier: int | None = None,
    base_optimizer: str | None = None,
) -> dict[str, pd.DataFrame]:
    runs = filter_runs(
        discover_completed_runs(Path(results_root)),
        level=level,
        token_multiplier=token_multiplier,
    )
    runs = runs[runs.extension.eq("wwpgd")]
    if base_optimizer:
        base = normalize_arm(base_optimizer).removesuffix("_wwpgd")
        runs = runs[runs.base_optimizer.map(normalize_arm).eq(base)]
    buckets: dict[str, list[pd.DataFrame]] = {
        "internal": [],
        "measurements": [],
        "relaxations": [],
        "fast_steps": [],
        "controller": [],
    }
    names = {
        "measurements": "wwpgd_endpoint_measurements.csv",
        "relaxations": "wwpgd_endpoint_relaxation.csv",
        "fast_steps": "wwpgd_fast_control_steps.csv",
        "controller": "wwpgd_controller.csv",
    }
    for row in runs.itertuples(index=False):
        run = Path(row.run_dir)
        internal = load_wwpgd_internal_diagnostics(run)
        if internal is not None and not internal.empty:
            buckets["internal"].append(_metadata(internal, row))
        for key, filename in names.items():
            frame = load_wwpgd_artifact(run, filename)
            if frame is not None and not frame.empty:
                buckets[key].append(_metadata(frame, row))
    return {
        key: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for key, frames in buckets.items()
    }


def _compatibility_fabrication(internal: pd.DataFrame) -> bool:
    if internal.empty or "diagnostics_mode" not in internal:
        return False
    compatibility = internal[internal.diagnostics_mode.astype(str).eq("compatibility")]
    for field in PRIVATE_FIELDS:
        if field in compatibility and compatibility[field].notna().any():
            return True
    return False


def diagnostic_capability(internal: pd.DataFrame) -> pd.DataFrame:
    if internal.empty:
        return pd.DataFrame(
            [
                {
                    "diagnostic_rows": 0,
                    "native_rows": 0,
                    "compatibility_rows": 0,
                    "native_private_diagnostics_available": False,
                    "compatibility_diagnostics_available": False,
                    "fabricated_private_compatibility_values": False,
                    "unsupported_internal_fields": "",
                }
            ]
        )
    mode = internal.get("diagnostics_mode", pd.Series("compatibility", index=internal.index)).astype(str)
    unsupported: set[str] = set()
    for value in internal.get("unsupported_internal_fields", pd.Series(dtype=str)).dropna().astype(str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                unsupported.update(map(str, parsed))
            else:
                unsupported.add(str(parsed))
        except json.JSONDecodeError:
            unsupported.add(value)
    return pd.DataFrame(
        [
            {
                "diagnostic_rows": len(internal),
                "native_rows": int(mode.eq("native").sum()),
                "compatibility_rows": int(mode.eq("compatibility").sum()),
                "native_private_diagnostics_available": bool(mode.eq("native").any()),
                "compatibility_diagnostics_available": bool(mode.eq("compatibility").any()),
                "fabricated_private_compatibility_values": _compatibility_fabrication(internal),
                "unsupported_internal_fields": ";".join(sorted(unsupported)),
            }
        ]
    )


def diagnostic_health(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    internal = data["internal"]
    measurements = _numeric(
        data["measurements"],
        ["initial_endpoint_relative_distance", "alpha_distance", "alpha_hardness"],
    )
    relaxations = _numeric(
        data["relaxations"],
        [
            "requested_relative_frobenius_change",
            "applied_relative_frobenius_change",
            "trust_region_limit",
            "trust_region_scale",
            "endpoint_relative_distance_before",
            "endpoint_relative_distance_after",
            "cumulative_applied_relative_change_after",
            "max_cumulative_relative_frobenius_change_per_refresh",
            "effective_base_gain",
        ],
    )
    rows: list[dict[str, Any]] = []
    identities = set()
    for frame in (internal, measurements, relaxations, data["fast_steps"]):
        if not frame.empty:
            identities.update(
                tuple(value)
                for value in frame[["level", "base_optimizer", "seed", "run_dir"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
    for level, base, seed, run_dir in sorted(identities):
        subset = lambda frame: frame[
            frame.level.eq(level)
            & frame.base_optimizer.eq(base)
            & frame.seed.eq(seed)
            & frame.run_dir.eq(run_dir)
        ] if not frame.empty else frame
        diag = subset(internal)
        measure = subset(measurements)
        dose = subset(relaxations)
        changed = _truth(dose.get("changed", pd.Series(False, index=dose.index))) if not dose.empty else pd.Series(dtype=bool)
        activated = _truth(measure.get("cache_activated", pd.Series(False, index=measure.index))) if not measure.empty else pd.Series(dtype=bool)
        cap_violation = False
        cumulative_violation = False
        distance_increase_fraction = math.nan
        saturation_fraction = math.nan
        if not dose.empty:
            applied = pd.to_numeric(dose.get("applied_relative_frobenius_change"), errors="coerce")
            limit = pd.to_numeric(dose.get("trust_region_limit"), errors="coerce")
            cap_violation = bool(((applied - limit) > 1e-10).fillna(False).any())
            cumulative = pd.to_numeric(dose.get("cumulative_applied_relative_change_after"), errors="coerce")
            cumulative_limit = pd.to_numeric(
                dose.get("max_cumulative_relative_frobenius_change_per_refresh"), errors="coerce"
            )
            cumulative_violation = bool(((cumulative - cumulative_limit) > 1e-10).fillna(False).any())
            before = pd.to_numeric(dose.get("endpoint_relative_distance_before"), errors="coerce")
            after = pd.to_numeric(dose.get("endpoint_relative_distance_after"), errors="coerce")
            valid = np.isfinite(before) & np.isfinite(after) & changed
            distance_increase_fraction = float((after[valid] > before[valid] + 1e-12).mean()) if valid.any() else math.nan
            scale = pd.to_numeric(dose.get("trust_region_scale"), errors="coerce")
            saturation_fraction = float((scale < 1.0 - 1e-12).mean()) if scale.notna().any() else math.nan
        modes = ",".join(sorted(set(diag.get("diagnostics_mode", pd.Series(dtype=str)).dropna().astype(str))))
        errors = []
        warnings = []
        if cap_violation:
            errors.append("per_step_dose_cap_violation")
        if cumulative_violation:
            errors.append("cumulative_refresh_dose_cap_violation")
        if _compatibility_fabrication(diag):
            errors.append("fabricated_private_compatibility_values")
        if measure.empty:
            errors.append("missing_endpoint_measurements")
        if diag.empty and not measure.empty:
            errors.append("missing_package_diagnostics_after_measurement")
        if not activated.any():
            warnings.append("no_endpoint_activated")
        if not changed.any():
            warnings.append("no_fast_layer_change")
        if math.isfinite(distance_increase_fraction) and distance_increase_fraction > 0.25:
            warnings.append("endpoint_distance_often_increases")
        if math.isfinite(saturation_fraction) and saturation_fraction > 0.5:
            warnings.append("trust_region_often_saturated")
        if "compatibility" in modes and "native" not in modes:
            warnings.append("private_wwpgd_internals_not_exposed_by_installed_package")
        rows.append(
            {
                "level": level,
                "base_optimizer": base,
                "seed": seed,
                "run_dir": run_dir,
                "diagnostic_mode": modes or "missing",
                "diagnostic_rows": len(diag),
                "measurement_rows": len(measure),
                "endpoint_activation_count": int(activated.sum()),
                "fast_relaxation_rows": len(dose),
                "changed_fast_layer_count": int(changed.sum()),
                "per_step_dose_cap_violation": cap_violation,
                "cumulative_dose_cap_violation": cumulative_violation,
                "endpoint_distance_increase_fraction": distance_increase_fraction,
                "trust_region_saturation_fraction": saturation_fraction,
                "status": "ERROR" if errors else "WARNING" if warnings else "PASS",
                "errors": ";".join(errors),
                "warnings": ";".join(warnings),
            }
        )
    return pd.DataFrame(rows)


def summarize_dose(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    relaxations = _numeric(
        data["relaxations"],
        [
            "applied_relative_frobenius_change",
            "requested_relative_frobenius_change",
            "controller_gain_applied",
            "controller_gain_requested",
            "alpha_distance",
            "cached_alpha_distance",
            "cached_alpha_hardness",
            "trust_region_scale",
            "endpoint_relative_distance_before",
            "endpoint_relative_distance_after",
        ],
    )
    measurements = data["measurements"]
    controller = data["controller"]
    dose_by_layer = pd.DataFrame()
    if not relaxations.empty and "layer_name" in relaxations:
        dose_by_layer = (
            relaxations.groupby(
                ["level", "base_optimizer", "seed", "layer_name", "matrix_type"],
                dropna=False,
            )
            .agg(
                fast_decisions=("optimizer_step", "count"),
                cumulative_applied_movement=("applied_relative_frobenius_change", "sum"),
                maximum_applied_movement=("applied_relative_frobenius_change", "max"),
                mean_applied_gain=("controller_gain_applied", "mean"),
                mean_trust_region_scale=("trust_region_scale", "mean"),
                minimum_endpoint_distance=("endpoint_relative_distance_after", "min"),
            )
            .reset_index()
        )
    endpoint_summary = pd.DataFrame()
    if not measurements.empty:
        frame = measurements.copy()
        frame["cache_activated_bool"] = _truth(
            frame.get("cache_activated", pd.Series(False, index=frame.index))
        )
        endpoint_summary = (
            frame.groupby(["level", "base_optimizer", "seed"], dropna=False)
            .agg(
                measurement_rows=("optimizer_step", "count"),
                endpoint_activations=("cache_activated_bool", "sum"),
                candidate_changed_count=("stock_candidate_changed", lambda value: int(_truth(value).sum())),
                mean_alpha_distance=("alpha_distance", "mean"),
                mean_initial_endpoint_distance=("initial_endpoint_relative_distance", "mean"),
            )
            .reset_index()
        )
    skip_summary = pd.DataFrame()
    if not controller.empty and "skip_reason" in controller:
        skip_summary = (
            controller.assign(skip_reason=controller.skip_reason.fillna("").astype(str))
            .groupby(["level", "base_optimizer", "seed", "skip_reason"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
    return dose_by_layer, endpoint_summary, skip_summary


def _write_figures(data: dict[str, pd.DataFrame], figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    relax = _numeric(
        data["relaxations"],
        [
            "applied_relative_frobenius_change",
            "controller_gain_applied",
            "cached_alpha_distance",
            "cached_alpha_hardness",
            "endpoint_relative_distance_before",
            "endpoint_relative_distance_after",
            "trust_region_scale",
        ],
    )
    measurements = _numeric(
        data["measurements"],
        ["alpha_distance", "alpha_hardness", "initial_endpoint_relative_distance"],
    )
    internal = _numeric(
        data["internal"],
        ["optimizer_step", "candidate_relative_frobenius_change", "original_to_candidate_relative_frobenius_change", "k_star", "selected_tail_size", "trace_log_retraction_residual"],
    )
    plot_specs = [
        (relax, "optimizer_step", "applied_relative_frobenius_change", "Applied WWPGD movement per fast step"),
        (relax, "cached_alpha_distance", "controller_gain_applied", "Alpha distance versus applied controller gain"),
        (relax, "endpoint_relative_distance_before", "endpoint_relative_distance_after", "Endpoint distance before versus after"),
        (measurements, "alpha_distance", "alpha_hardness", "Alpha distance versus layer hardness"),
        (internal, "optimizer_step", "candidate_relative_frobenius_change", "Stock WWPGD candidate movement"),
    ]
    for index, (frame, x_name, y_name, title) in enumerate(plot_specs):
        if frame.empty or x_name not in frame or y_name not in frame:
            continue
        clean = frame[[x_name, y_name, "level", "seed"]].dropna(subset=[x_name, y_name])
        if clean.empty:
            continue
        figure, axis = plt.subplots(figsize=(7.5, 5.0))
        for (level, seed), group in clean.groupby(["level", "seed"]):
            axis.scatter(group[x_name], group[y_name], s=14, alpha=0.65, label=f"L{level} seed {seed}")
        if "before versus after" in title.lower():
            low = min(float(clean[x_name].min()), float(clean[y_name].min()))
            high = max(float(clean[x_name].max()), float(clean[y_name].max()))
            axis.plot([low, high], [low, high], linestyle="--", linewidth=1, label="no change")
        axis.set_xlabel(x_name.replace("_", " "))
        axis.set_ylabel(y_name.replace("_", " "))
        axis.set_title(title)
        axis.legend(fontsize="x-small")
        figure.tight_layout()
        path = figures_dir / f"wwpgd_diagnostic_{index:02d}_{y_name}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        outputs.append(path)
    if not internal.empty and "diagnostics_mode" in internal and internal.diagnostics_mode.astype(str).eq("native").any():
        native = internal[internal.diagnostics_mode.astype(str).eq("native")]
        for y_name in ("k_star", "selected_tail_size", "trace_log_retraction_residual"):
            if y_name not in native:
                continue
            figure, axis = plt.subplots(figsize=(8.0, 4.8))
            for (seed, layer), group in native.groupby(["seed", "layer_name"], dropna=False):
                axis.plot(group.optimizer_step, group[y_name], marker=".", label=f"{seed}:{layer}")
            axis.set_xlabel("Optimizer step")
            axis.set_ylabel(y_name.replace("_", " "))
            axis.set_title(f"Native WWPGD {y_name}")
            axis.legend(fontsize="xx-small")
            figure.tight_layout()
            path = figures_dir / f"wwpgd_native_{y_name}.png"
            figure.savefig(path, dpi=170)
            plt.close(figure)
            outputs.append(path)
    return outputs


def analyze_wwpgd_diagnostics(
    results_root: Path,
    output_dir: Path,
    *,
    figures_dir: Path | None = None,
    level: int | None = None,
    token_multiplier: int | None = None,
    base_optimizer: str | None = None,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir or output_dir / "figures")
    data = collect_wwpgd_diagnostics(
        Path(results_root),
        level=level,
        token_multiplier=token_multiplier,
        base_optimizer=base_optimizer,
    )
    capability = diagnostic_capability(data["internal"])
    health = diagnostic_health(data)
    dose_by_layer, endpoint_summary, skip_summary = summarize_dose(data)
    outputs = {
        "wwpgd_internal_diagnostics_all.csv": data["internal"],
        "wwpgd_endpoint_measurements_all.csv": data["measurements"],
        "wwpgd_endpoint_relaxation_all.csv": data["relaxations"],
        "wwpgd_fast_control_steps_all.csv": data["fast_steps"],
        "wwpgd_controller_all.csv": data["controller"],
        "wwpgd_diagnostic_capability.csv": capability,
        "wwpgd_diagnostic_health.csv": health,
        "wwpgd_dose_by_layer.csv": dose_by_layer,
        "wwpgd_endpoint_summary.csv": endpoint_summary,
        "wwpgd_skip_reason_summary.csv": skip_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    _write_figures(data, figures_dir)
    return outputs
