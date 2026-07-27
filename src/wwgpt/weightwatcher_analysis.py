"""WeightWatcher analysis across seeds without recomputing spectra."""
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
    load_weightwatcher_aggregates,
    normalize_arm,
)

ALPHA_MEASURES = (
    "alpha",
    "alpha_distance",
    "D",
    "xmin",
    "detX_num",
    "num_evals",
    "spectral_norm",
    "stable_rank",
)


def _stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _target_and_deadbands(manifest: dict[str, Any]) -> tuple[float, float, float]:
    ext = manifest.get("extension_hyperparameters") or {}
    adaptive = ext.get("adaptive") or manifest.get("wwpgd_adaptive_config") or {}
    target = float(manifest.get("target_alpha", ext.get("target_alpha", 2.0)))
    above = adaptive.get("above_target") or {}
    below = adaptive.get("below_target") or {}
    return target, float(above.get("deadband", 0.4)), float(below.get("deadband", 0.2))


def collect_weightwatcher_rows(
    results_root: Path,
    *,
    level: int | None = None,
    token_multiplier: int | None = None,
    base_optimizer: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = filter_runs(
        discover_completed_runs(Path(results_root)),
        level=level,
        token_multiplier=token_multiplier,
    )
    if runs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if base_optimizer:
        base = normalize_arm(base_optimizer).removesuffix("_wwpgd")
        runs = runs[runs["base_optimizer"].map(normalize_arm).eq(base)]
    valid_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    trap_frames: list[pd.DataFrame] = []
    for row in runs.itertuples(index=False):
        run = Path(row.run_dir)
        raw = load_alpha_measurements(run)
        if raw is not None and not raw.empty:
            audit = raw.copy()
            target, above_deadband, below_deadband = _target_and_deadbands(row.manifest)
            audit["target_alpha"] = target
            audit["above_deadband"] = above_deadband
            audit["below_deadband"] = below_deadband
            audit["seed"] = row.seed
            audit["level"] = row.level
            audit["token_multiplier"] = row.token_multiplier
            audit["base_optimizer"] = row.base_optimizer
            audit["arm"] = row.arm
            audit["extension"] = row.extension
            alpha = pd.to_numeric(audit.get("alpha"), errors="coerce")
            audit["alpha_distance"] = (alpha - target).abs()
            audit["alpha_side"] = np.select(
                [alpha > target, alpha < target],
                ["above_target", "below_target"],
                default="at_target",
            )
            audit["inside_target_deadband"] = np.where(
                alpha >= target,
                (alpha - target) <= above_deadband,
                (target - alpha) <= below_deadband,
            )
            audit_frames.append(audit)
            valid = filter_scientific_alpha(audit)
            if not valid.empty:
                valid_frames.append(valid)
        traps = load_weightwatcher_aggregates(run)
        if traps is not None and not traps.empty:
            traps = traps.copy()
            traps["seed"] = row.seed
            traps["level"] = row.level
            traps["token_multiplier"] = row.token_multiplier
            traps["base_optimizer"] = row.base_optimizer
            traps["arm"] = row.arm
            traps["extension"] = row.extension
            trap_frames.append(traps)
    valid_all = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()
    audit_all = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    trap_all = pd.concat(trap_frames, ignore_index=True) if trap_frames else pd.DataFrame()
    return valid_all, audit_all, trap_all


def _stat_rows(frame: pd.DataFrame, value: str, group_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty or value not in frame:
        return rows
    for identity, group in frame.groupby(group_columns, dropna=False):
        values = pd.to_numeric(group[value], errors="coerce")
        values = values[np.isfinite(values)].to_numpy(float)
        count = len(values)
        if count == 0:
            continue
        mean = float(values.mean())
        median = float(np.median(values))
        std = float(values.std(ddof=1)) if count > 1 else 0.0
        sem = std / math.sqrt(count) if count > 1 else 0.0
        if count > 1:
            critical = float(stats.t.ppf(0.975, count - 1))
            ci_low, ci_high = mean - critical * sem, mean + critical * sem
            rng = np.random.default_rng(_stable_seed(value, *identity))
            bootstrap = rng.choice(values, size=(10_000, count), replace=True).mean(axis=1)
            boot_low, boot_high = np.quantile(bootstrap, [0.025, 0.975])
        else:
            ci_low = ci_high = boot_low = boot_high = mean
        record = dict(zip(group_columns, identity if isinstance(identity, tuple) else (identity,), strict=True))
        record.update(
            {
                "measure": value,
                "n": count,
                "mean": mean,
                "median": median,
                "std": std,
                "sem": sem,
                "t_ci_low": float(ci_low),
                "t_ci_high": float(ci_high),
                "bootstrap_ci_low": float(boot_low),
                "bootstrap_ci_high": float(boot_high),
                "two_sd_lower": mean - 2.0 * std,
                "two_sd_upper": mean + 2.0 * std,
            }
        )
        rows.append(record)
    return rows


def summarize_weightwatcher(
    valid: pd.DataFrame, audit: pd.DataFrame, traps: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    if not valid.empty:
        if "tokens_seen" not in valid and "optimizer_step" in valid:
            valid["tokens_seen"] = valid["optimizer_step"]
        group = [
            "level",
            "token_multiplier",
            "base_optimizer",
            "arm",
            "tokens_seen",
            "matrix_type",
        ]
        for measure in ALPHA_MEASURES:
            summary_rows.extend(_stat_rows(valid, measure, group))
    summary = pd.DataFrame(summary_rows)

    coverage_rows: list[dict[str, Any]] = []
    if not audit.empty:
        group_columns = ["level", "token_multiplier", "base_optimizer", "arm", "optimizer_step"]
        for identity, group in audit.groupby(group_columns, dropna=False):
            valid_flag = group.get("valid_for_science", pd.Series(False, index=group.index)).map(_truth)
            projected_flag = group.get("projected", pd.Series(False, index=group.index)).map(_truth)
            included = group.get(
                "included_in_projected_alpha_summary", pd.Series(False, index=group.index)
            ).map(_truth)
            primary = group[valid_flag & projected_flag & included]
            alpha = pd.to_numeric(primary.get("alpha"), errors="coerce")
            target = pd.to_numeric(primary.get("target_alpha"), errors="coerce")
            inside = primary.get(
                "inside_target_deadband", pd.Series(False, index=primary.index)
            ).map(_truth)
            record = dict(zip(group_columns, identity, strict=True))
            record.update(
                {
                    "all_weighted_rows": len(group),
                    "projected_rows": int(projected_flag.sum()),
                    "valid_projected_rows": len(primary),
                    "invalid_or_excluded_rows": len(group) - len(primary),
                    "valid_projected_fraction": len(primary) / max(int(projected_flag.sum()), 1),
                    "median_alpha": float(alpha.median()) if alpha.notna().any() else math.nan,
                    "median_alpha_distance": float((alpha - target).abs().median()) if alpha.notna().any() else math.nan,
                    "inside_deadband_fraction": float(inside.mean()) if len(inside) else math.nan,
                    "above_target_fraction": float((alpha > target).mean()) if alpha.notna().any() else math.nan,
                    "below_target_fraction": float((alpha < target).mean()) if alpha.notna().any() else math.nan,
                }
            )
            coverage_rows.append(record)
    coverage = pd.DataFrame(coverage_rows)

    exclusion = pd.DataFrame()
    if not audit.empty:
        reason = audit.get(
            "validity_exclusion_reason", pd.Series("", index=audit.index)
        ).fillna("").astype(str)
        excluded = audit.assign(exclusion_reason=reason)
        excluded = excluded[excluded.exclusion_reason.ne("")]
        if not excluded.empty:
            exclusion = (
                excluded.groupby(
                    ["level", "base_optimizer", "arm", "exclusion_reason"], dropna=False
                )
                .size()
                .rename("count")
                .reset_index()
            )

    trap_summary_rows: list[dict[str, Any]] = []
    if not traps.empty:
        if "tokens_seen" not in traps and "step" in traps:
            traps["tokens_seen"] = traps["step"]
        group = ["level", "token_multiplier", "base_optimizer", "arm", "tokens_seen"]
        for measure in (
            "trap_layer_count",
            "trap_layer_fraction",
            "mean_alpha",
            "mean_spectral_norm",
            "mean_stable_rank",
        ):
            trap_summary_rows.extend(_stat_rows(traps, measure, group))
    trap_summary = pd.DataFrame(trap_summary_rows)
    return summary, coverage, exclusion, trap_summary


def _write_figures(
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    trap_summary: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for measure in ("alpha", "alpha_distance", "D", "stable_rank"):
        frame = summary[summary["measure"].eq(measure)] if not summary.empty else pd.DataFrame()
        if frame.empty:
            continue
        for identity, group in frame.groupby(
            ["level", "token_multiplier", "base_optimizer", "matrix_type"], dropna=False
        ):
            figure, axis = plt.subplots(figsize=(8.5, 4.8))
            for arm, part in group.groupby("arm"):
                part = part.sort_values("tokens_seen")
                axis.plot(part.tokens_seen, part["mean"], label=f"{arm} mean")
                axis.fill_between(part.tokens_seen, part.t_ci_low, part.t_ci_high, alpha=0.24)
                axis.plot(part.tokens_seen, part.two_sd_lower, linestyle="--", linewidth=0.7)
                axis.plot(part.tokens_seen, part.two_sd_upper, linestyle="--", linewidth=0.7)
            if measure == "alpha":
                axis.axhline(2.0, linewidth=1, linestyle=":", label="target α=2")
            axis.set_xlabel("Tokens processed")
            axis.set_ylabel(measure.replace("_", " "))
            axis.set_title(f"Level {identity[0]} {identity[2]} {identity[3]} {measure}")
            axis.legend(fontsize="small")
            figure.tight_layout()
            path = figures_dir / f"level{identity[0]}_{identity[2]}_{identity[3]}_{measure}.png"
            figure.savefig(path, dpi=170)
            plt.close(figure)
            outputs.append(path)
    if not coverage.empty:
        for identity, group in coverage.groupby(
            ["level", "token_multiplier", "base_optimizer"], dropna=False
        ):
            figure, axis = plt.subplots(figsize=(8.5, 4.8))
            for arm, part in group.groupby("arm"):
                part = part.sort_values("optimizer_step")
                axis.plot(part.optimizer_step, part.inside_deadband_fraction, marker=".", label=arm)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xlabel("Optimizer step")
            axis.set_ylabel("Fraction of valid projected layers inside target deadband")
            axis.set_title(f"Level {identity[0]} {identity[2]} α-target occupancy")
            axis.legend()
            figure.tight_layout()
            path = figures_dir / f"level{identity[0]}_{identity[2]}_alpha_deadband_fraction.png"
            figure.savefig(path, dpi=170)
            plt.close(figure)
            outputs.append(path)
    if not trap_summary.empty:
        frame = trap_summary[trap_summary.measure.eq("trap_layer_fraction")]
        for identity, group in frame.groupby(
            ["level", "token_multiplier", "base_optimizer"], dropna=False
        ):
            figure, axis = plt.subplots(figsize=(8.5, 4.8))
            for arm, part in group.groupby("arm"):
                part = part.sort_values("tokens_seen")
                axis.plot(part.tokens_seen, part["mean"], label=arm)
                axis.fill_between(part.tokens_seen, part.t_ci_low, part.t_ci_high, alpha=0.24)
            axis.set_xlabel("Tokens processed")
            axis.set_ylabel("Trap-layer fraction")
            axis.set_title(f"Level {identity[0]} {identity[2]} randomized WeightWatcher traps")
            axis.legend()
            figure.tight_layout()
            path = figures_dir / f"level{identity[0]}_{identity[2]}_trap_layer_fraction.png"
            figure.savefig(path, dpi=170)
            plt.close(figure)
            outputs.append(path)
    return outputs


def analyze_weightwatcher_results(
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
    valid, audit, traps = collect_weightwatcher_rows(
        Path(results_root),
        level=level,
        token_multiplier=token_multiplier,
        base_optimizer=base_optimizer,
    )
    summary, coverage, exclusion, trap_summary = summarize_weightwatcher(valid, audit, traps)
    outputs = {
        "weightwatcher_valid_alpha_rows.csv": valid,
        "weightwatcher_alpha_audit_rows.csv": audit,
        "weightwatcher_alpha_seed_summary.csv": summary,
        "weightwatcher_alpha_coverage.csv": coverage,
        "weightwatcher_exclusion_reasons.csv": exclusion,
        "weightwatcher_trap_rows.csv": traps,
        "weightwatcher_trap_seed_summary.csv": trap_summary,
        "scientific_alpha.csv": valid,
        "trap_diagnostics.csv": traps,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    _write_figures(summary, coverage, trap_summary, figures_dir)
    return outputs
