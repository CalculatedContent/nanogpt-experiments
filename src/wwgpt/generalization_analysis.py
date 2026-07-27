"""Descriptive paired generalization analysis across seeds."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from wwgpt.notebook_support import (
    discover_completed_runs,
    filter_runs,
    filter_scientific_alpha,
    load_alpha_measurements,
    load_metrics,
    load_selected_checkpoint_metrics,
    load_weightwatcher_aggregates,
    load_wwpgd_artifact,
    normalize_arm,
    pair_arms,
)


def _stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def _finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _auc(frame: pd.DataFrame, metric: str) -> float:
    if frame.empty or metric not in frame:
        return math.nan
    x = pd.to_numeric(frame.get("tokens_seen", frame.get("step")), errors="coerce")
    y = pd.to_numeric(frame[metric], errors="coerce")
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return math.nan
    ordered = pd.DataFrame({"x": x[valid], "y": y[valid]}).sort_values("x").drop_duplicates("x")
    span = float(ordered.x.iloc[-1] - ordered.x.iloc[0])
    return float(np.trapz(ordered.y, ordered.x) / span) if span > 0 else math.nan


def _run_diagnostics(run: Path) -> dict[str, float]:
    alpha = filter_scientific_alpha(load_alpha_measurements(run))
    final_alpha_distance = math.nan
    median_alpha_distance = math.nan
    if not alpha.empty:
        target = pd.to_numeric(alpha.get("target_alpha", 2.0), errors="coerce")
        values = (pd.to_numeric(alpha.alpha, errors="coerce") - target).abs()
        median_alpha_distance = float(values.median()) if values.notna().any() else math.nan
        step = pd.to_numeric(alpha.get("optimizer_step"), errors="coerce")
        if step.notna().any():
            last = alpha.loc[step.eq(step.max())]
            target_last = pd.to_numeric(last.get("target_alpha", 2.0), errors="coerce")
            distance_last = (pd.to_numeric(last.alpha, errors="coerce") - target_last).abs()
            final_alpha_distance = float(distance_last.median()) if distance_last.notna().any() else math.nan
    traps = load_weightwatcher_aggregates(run)
    final_trap_fraction = math.nan
    if traps is not None and not traps.empty and "trap_layer_fraction" in traps:
        order = "tokens_seen" if "tokens_seen" in traps else "step"
        traps = traps.sort_values(order)
        final_trap_fraction = _finite(traps.iloc[-1].get("trap_layer_fraction"))
    relaxation = load_wwpgd_artifact(run, "wwpgd_endpoint_relaxation.csv")
    cumulative_dose = 0.0
    dose_saturation_fraction = math.nan
    if relaxation is not None and not relaxation.empty:
        applied = pd.to_numeric(
            relaxation.get("applied_relative_frobenius_change"), errors="coerce"
        )
        cumulative_dose = float(applied.fillna(0).sum())
        if "dose_cap_saturated" in relaxation:
            dose_saturation_fraction = float(
                relaxation.dose_cap_saturated.astype(str).str.lower().isin({"true", "1"}).mean()
            )
    return {
        "median_alpha_distance": median_alpha_distance,
        "final_alpha_distance": final_alpha_distance,
        "final_trap_layer_fraction": final_trap_fraction,
        "cumulative_wwpgd_dose": cumulative_dose,
        "dose_saturation_fraction": dose_saturation_fraction,
    }


def collect_generalization_rows(
    results_root: Path,
    *,
    level: int | None = None,
    token_multiplier: int | None = None,
    base_optimizer: str | None = None,
) -> pd.DataFrame:
    runs = filter_runs(
        discover_completed_runs(Path(results_root)),
        level=level,
        token_multiplier=token_multiplier,
    )
    if runs.empty or "base_optimizer" not in runs.columns:
        return pd.DataFrame()
    if base_optimizer:
        base = normalize_arm(base_optimizer).removesuffix("_wwpgd")
        runs = runs[runs.base_optimizer.map(normalize_arm).eq(base)]
    rows: list[dict[str, Any]] = []
    for run_row in runs.itertuples(index=False):
        run = Path(run_row.run_dir)
        metrics = load_metrics(run)
        selected = load_selected_checkpoint_metrics(run)
        if metrics.empty or selected is None or selected.empty:
            continue
        val = pd.to_numeric(metrics.get("validation_loss"), errors="coerce")
        train = pd.to_numeric(metrics.get("train_loss"), errors="coerce")
        valid_val = val[np.isfinite(val)]
        valid_train = train[np.isfinite(train)]
        if valid_val.empty or valid_train.empty:
            continue
        selected_row = selected.iloc[0]
        record: dict[str, Any] = {
            "layout": run_row.layout,
            "run_dir": str(run),
            "seed": run_row.seed,
            "level": run_row.level,
            "token_multiplier": run_row.token_multiplier,
            "base_optimizer": run_row.base_optimizer,
            "arm": run_row.arm,
            "extension": run_row.extension,
            "best_validation_loss": float(valid_val.min()),
            "final_validation_loss": float(valid_val.iloc[-1]),
            "late_validation_degradation": float(valid_val.iloc[-1] - valid_val.min()),
            "final_train_loss": float(valid_train.iloc[-1]),
            "final_train_validation_gap": float(valid_val.iloc[-1] - valid_train.iloc[-1]),
            "validation_loss_auc": _auc(metrics, "validation_loss"),
            "generalization_gap_auc": _auc(metrics, "train_validation_gap"),
        }
        for metric in (
            "train_loss",
            "validation_loss",
            "test_loss",
            "train_perplexity",
            "validation_perplexity",
            "test_perplexity",
            "train_next_token_accuracy",
            "validation_next_token_accuracy",
            "test_next_token_accuracy",
            "train_validation_loss_gap",
            "train_test_loss_gap",
            "train_validation_perplexity_gap",
            "train_test_perplexity_gap",
            "selected_checkpoint_step",
        ):
            record[f"selected_{metric}"] = _finite(selected_row.get(metric))
        record.update(_run_diagnostics(run))
        rows.append(record)
    return pd.DataFrame(rows)


def paired_generalization_effects(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    metrics = [
        "best_validation_loss",
        "final_validation_loss",
        "late_validation_degradation",
        "final_train_validation_gap",
        "validation_loss_auc",
        "generalization_gap_auc",
        "selected_test_loss",
        "selected_test_perplexity",
        "selected_test_next_token_accuracy",
        "selected_train_test_loss_gap",
        "selected_train_test_perplexity_gap",
        "final_alpha_distance",
        "final_trap_layer_fraction",
        "cumulative_wwpgd_dose",
    ]
    records: list[dict[str, Any]] = []
    keys = ["level", "token_multiplier", "base_optimizer", "seed"]
    for identity, group in rows.groupby(keys, dropna=False):
        baseline = group[group.extension.eq("none")]
        wwpgd = group[group.extension.eq("wwpgd")]
        if len(baseline) != 1 or len(wwpgd) != 1:
            continue
        baseline_row, wwpgd_row = baseline.iloc[0], wwpgd.iloc[0]
        for metric in metrics:
            b, w = _finite(baseline_row.get(metric)), _finite(wwpgd_row.get(metric))
            if not (math.isfinite(b) and math.isfinite(w)):
                continue
            direction = "higher" if "accuracy" in metric else "descriptive" if metric == "cumulative_wwpgd_dose" else "lower"
            effect = w - b
            records.append(
                {
                    "level": identity[0],
                    "token_multiplier": identity[1],
                    "base_optimizer": identity[2],
                    "seed": identity[3],
                    "metric": metric,
                    "beneficial_direction": direction,
                    "baseline_value": b,
                    "wwpgd_value": w,
                    "paired_effect": effect,
                    "wwpgd_better": effect > 0 if direction == "higher" else effect < 0 if direction == "lower" else None,
                }
            )
    return pd.DataFrame(records)


def summarize_generalization_effects(effects: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if effects.empty:
        return pd.DataFrame()
    keys = ["level", "token_multiplier", "base_optimizer", "metric", "beneficial_direction"]
    for identity, group in effects.groupby(keys, dropna=False):
        values = pd.to_numeric(group.paired_effect, errors="coerce")
        values = values[np.isfinite(values)].to_numpy(float)
        count = len(values)
        if count == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if count > 1 else 0.0
        sem = std / math.sqrt(count) if count > 1 else 0.0
        if count > 1:
            critical = float(stats.t.ppf(0.975, count - 1))
            ci_low, ci_high = mean - critical * sem, mean + critical * sem
            rng = np.random.default_rng(_stable_seed(*identity))
            bootstrap = rng.choice(values, size=(10_000, count), replace=True).mean(axis=1)
            boot_low, boot_high = np.quantile(bootstrap, [0.025, 0.975])
        else:
            ci_low = ci_high = boot_low = boot_high = mean
        records.append(
            {
                "level": identity[0],
                "token_multiplier": identity[1],
                "base_optimizer": identity[2],
                "metric": identity[3],
                "beneficial_direction": identity[4],
                "n_pairs": count,
                "mean_paired_effect": mean,
                "median_paired_effect": float(np.median(values)),
                "sample_std": std,
                "standard_error": sem,
                "t_ci_low": float(ci_low),
                "t_ci_high": float(ci_high),
                "bootstrap_ci_low": float(boot_low),
                "bootstrap_ci_high": float(boot_high),
                "two_sd_lower": mean - 2 * std,
                "two_sd_upper": mean + 2 * std,
                "wwpgd_better_seed_fraction": float(
                    pd.to_numeric(group.wwpgd_better, errors="coerce").mean()
                ) if group.wwpgd_better.notna().any() else math.nan,
            }
        )
    return pd.DataFrame(records)


def diagnostic_correlations(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if rows.empty:
        return pd.DataFrame()
    outcomes = ["selected_test_loss", "selected_test_perplexity", "selected_test_next_token_accuracy", "selected_train_test_loss_gap"]
    diagnostics = ["final_alpha_distance", "final_trap_layer_fraction", "cumulative_wwpgd_dose", "dose_saturation_fraction"]
    wwpgd = rows[rows.extension.eq("wwpgd")]
    for identity, group in wwpgd.groupby(["level", "token_multiplier", "base_optimizer"], dropna=False):
        for diagnostic in diagnostics:
            for outcome in outcomes:
                clean = group[[diagnostic, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(clean) < 3:
                    continue
                rho, p_value = stats.spearmanr(clean[diagnostic], clean[outcome])
                records.append(
                    {
                        "level": identity[0],
                        "token_multiplier": identity[1],
                        "base_optimizer": identity[2],
                        "diagnostic": diagnostic,
                        "outcome": outcome,
                        "n_runs": len(clean),
                        "spearman_rho": float(rho),
                        "p_value_descriptive": float(p_value),
                        "interpretation": "association only; not causal and not confirmatory",
                    }
                )
    return pd.DataFrame(records)


def _write_figures(rows: pd.DataFrame, effects: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for metric in ("selected_test_loss", "selected_test_perplexity", "selected_test_next_token_accuracy", "late_validation_degradation"):
        frame = effects[effects.metric.eq(metric)] if not effects.empty else pd.DataFrame()
        if frame.empty:
            continue
        for identity, group in frame.groupby(["level", "token_multiplier", "base_optimizer"], dropna=False):
            figure, axis = plt.subplots(figsize=(6.5, 5.0))
            for row in group.itertuples(index=False):
                axis.plot([0, 1], [row.baseline_value, row.wwpgd_value], marker="o", alpha=0.8)
            axis.set_xticks([0, 1], ["baseline", "WWPGD"])
            axis.set_ylabel(metric.replace("_", " "))
            axis.set_title(f"Level {identity[0]} {identity[2]}: {metric}")
            figure.tight_layout()
            path = figures_dir / f"level{identity[0]}_{identity[2]}_{metric}.png"
            figure.savefig(path, dpi=170)
            plt.close(figure)
            outputs.append(path)
    wwpgd = rows[rows.extension.eq("wwpgd")] if not rows.empty else pd.DataFrame()
    for x_name, y_name in (("final_alpha_distance", "selected_test_loss"), ("cumulative_wwpgd_dose", "selected_test_loss"), ("final_trap_layer_fraction", "selected_train_test_loss_gap")):
        clean = wwpgd[[x_name, y_name, "level", "seed"]].apply(
            lambda column, x_name=x_name, y_name=y_name: pd.to_numeric(column, errors="coerce") if column.name in {x_name, y_name} else column
        ).dropna(subset=[x_name, y_name]) if not wwpgd.empty else pd.DataFrame()
        if clean.empty:
            continue
        figure, axis = plt.subplots(figsize=(7.0, 5.0))
        for level, group in clean.groupby("level"):
            axis.scatter(group[x_name], group[y_name], label=f"Level {level}")
        axis.set_xlabel(x_name.replace("_", " "))
        axis.set_ylabel(y_name.replace("_", " "))
        axis.set_title("Descriptive association across WWPGD runs")
        axis.legend()
        figure.tight_layout()
        path = figures_dir / f"association_{x_name}_vs_{y_name}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        outputs.append(path)
    return outputs


def analyze_generalization_results(
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
    rows = collect_generalization_rows(
        Path(results_root),
        level=level,
        token_multiplier=token_multiplier,
        base_optimizer=base_optimizer,
    )
    effects = paired_generalization_effects(rows)
    summary = summarize_generalization_effects(effects)
    correlations = diagnostic_correlations(rows)
    outputs = {
        "generalization_run_summary.csv": rows,
        "generalization_paired_effects_by_seed.csv": effects,
        "generalization_paired_effect_summary.csv": summary,
        "generalization_diagnostic_correlations.csv": correlations,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    _write_figures(rows, effects, figures_dir)
    return outputs
