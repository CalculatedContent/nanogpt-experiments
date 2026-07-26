"""Paired multi-seed learning-curve and selected-checkpoint analysis.

Seed is always the experimental unit.  Curves are aligned within a level,
token budget, and base optimizer; layers and evaluation events are never
counted as independent replicates.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from wwgpt.notebook_support import (
    discover_completed_runs,
    filter_runs,
    load_metrics,
    load_selected_checkpoint_metrics,
    normalize_arm,
    pair_arms,
)

CURVE_METRICS: dict[str, str] = {
    "train_loss": "lower",
    "validation_loss": "lower",
    "train_perplexity": "lower",
    "validation_perplexity": "lower",
    "train_next_token_accuracy": "higher",
    "validation_next_token_accuracy": "higher",
    "train_validation_gap": "lower",
    "gradient_norm_before_clip": "descriptive",
    "learning_rate": "descriptive",
    "model_parameter_norm": "descriptive",
}

SELECTED_METRICS: dict[str, str] = {
    "train_loss": "lower",
    "validation_loss": "lower",
    "test_loss": "lower",
    "train_perplexity": "lower",
    "validation_perplexity": "lower",
    "test_perplexity": "lower",
    "train_next_token_accuracy": "higher",
    "validation_next_token_accuracy": "higher",
    "test_next_token_accuracy": "higher",
    "train_validation_loss_gap": "lower",
    "train_test_loss_gap": "lower",
    "train_validation_perplexity_gap": "lower",
    "train_test_perplexity_gap": "lower",
    "selected_checkpoint_step": "descriptive",
}


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _finite_series(values: Iterable[Any]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    return series[np.isfinite(series)].to_numpy(float)


def _exact_sign_flip_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    count = len(values)
    if count == 0:
        return math.nan
    observed = abs(float(values.mean()))
    if count <= 20:
        total = 0
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=count):
            total += 1
            value = abs(float(np.mean(values * np.asarray(signs, dtype=float))))
            extreme += value >= observed - 1e-15
        return extreme / total
    rng = np.random.default_rng(_stable_seed("sign-flip", *values.tolist()))
    signs = rng.choice((-1.0, 1.0), size=(100_000, count))
    means = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(means >= observed) + 1) / (len(means) + 1))


def _summary(values: np.ndarray, *, seed: int) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    count = len(values)
    if count == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "t_ci_low": math.nan,
            "t_ci_high": math.nan,
            "bootstrap_ci_low": math.nan,
            "bootstrap_ci_high": math.nan,
            "bollinger_2sd_low": math.nan,
            "bollinger_2sd_high": math.nan,
        }
    mean = float(values.mean())
    median = float(np.median(values))
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    sem = std / math.sqrt(count) if count > 1 else 0.0
    if count > 1:
        critical = float(stats.t.ppf(0.975, count - 1))
        t_low, t_high = mean - critical * sem, mean + critical * sem
        rng = np.random.default_rng(seed)
        bootstrap = rng.choice(values, size=(10_000, count), replace=True).mean(axis=1)
        bootstrap_low, bootstrap_high = np.quantile(bootstrap, [0.025, 0.975])
    else:
        t_low = t_high = bootstrap_low = bootstrap_high = mean
    return {
        "n": count,
        "mean": mean,
        "median": median,
        "std": std,
        "sem": sem,
        "t_ci_low": float(t_low),
        "t_ci_high": float(t_high),
        "bootstrap_ci_low": float(bootstrap_low),
        "bootstrap_ci_high": float(bootstrap_high),
        "bollinger_2sd_low": mean - 2.0 * std,
        "bollinger_2sd_high": mean + 2.0 * std,
    }


def discover_pairs(
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
    if runs.empty:
        return pd.DataFrame()
    bases = (
        [normalize_arm(base_optimizer).removesuffix("_wwpgd")]
        if base_optimizer
        else sorted(set(runs["base_optimizer"].dropna().map(normalize_arm)))
    )
    frames: list[pd.DataFrame] = []
    for base in bases:
        paired = pair_arms(runs, base)
        if paired.empty:
            continue
        meta = (
            runs[runs["base_optimizer"].map(normalize_arm).eq(base)]
            .groupby(["layout", "seed"], dropna=False)[["level", "token_multiplier"]]
            .first()
            .reset_index()
        )
        paired = paired.merge(meta, on=["layout", "seed"], how="left")
        paired["base_optimizer"] = base
        frames.append(paired)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_learning_curves(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if pairs.empty:
        return pd.DataFrame()
    for pair in pairs.itertuples(index=False):
        for arm, run_value in (("baseline", pair.baseline_run), ("wwpgd", pair.wwpgd_run)):
            metrics = load_metrics(Path(run_value))
            if metrics.empty:
                continue
            x = pd.to_numeric(metrics.get("tokens_seen", metrics.get("step")), errors="coerce")
            steps = pd.to_numeric(metrics.get("optimizer_steps", metrics.get("step")), errors="coerce")
            for metric, direction in CURVE_METRICS.items():
                if metric not in metrics:
                    continue
                values = pd.to_numeric(metrics[metric], errors="coerce")
                for token, step, value in zip(x, steps, values, strict=False):
                    if not (math.isfinite(float(token)) and math.isfinite(float(value))):
                        continue
                    rows.append(
                        {
                            "layout": pair.layout,
                            "seed": pair.seed,
                            "level": pair.level,
                            "token_multiplier": pair.token_multiplier,
                            "base_optimizer": pair.base_optimizer,
                            "arm": arm,
                            "metric": metric,
                            "beneficial_direction": direction,
                            "tokens_seen": float(token),
                            "optimizer_step": float(step),
                            "value": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def _common_grid(curves: list[pd.DataFrame], points: int) -> np.ndarray:
    clean = [
        curve[["tokens_seen", "value"]]
        .dropna()
        .sort_values("tokens_seen")
        .drop_duplicates("tokens_seen")
        for curve in curves
        if len(curve) >= 2
    ]
    if not clean:
        return np.asarray([], dtype=float)
    low = max(float(curve.tokens_seen.min()) for curve in clean)
    high = min(float(curve.tokens_seen.max()) for curve in clean)
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        return np.asarray([], dtype=float)
    return np.linspace(low, high, points)


def summarize_arm_curves(curves: pd.DataFrame, *, points: int = 200) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if curves.empty:
        return pd.DataFrame()
    keys = ["level", "token_multiplier", "base_optimizer", "metric", "arm"]
    for identity, group in curves.groupby(keys, dropna=False):
        seed_curves = [part for _, part in group.groupby("seed", dropna=False)]
        grid = _common_grid(seed_curves, points)
        if len(grid) == 0:
            continue
        matrix = []
        for part in seed_curves:
            part = part.sort_values("tokens_seen").drop_duplicates("tokens_seen")
            matrix.append(np.interp(grid, part.tokens_seen, part.value))
        values = np.vstack(matrix)
        direction = str(group["beneficial_direction"].iloc[0])
        for index, token in enumerate(grid):
            summary = _summary(values[:, index], seed=_stable_seed("curve", *identity, index))
            rows.append(
                {
                    "level": identity[0],
                    "token_multiplier": identity[1],
                    "base_optimizer": identity[2],
                    "metric": identity[3],
                    "arm": identity[4],
                    "beneficial_direction": direction,
                    "tokens_seen": float(token),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def paired_curve_effects(curves: pd.DataFrame, *, points: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed_rows: list[dict[str, Any]] = []
    if curves.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["level", "token_multiplier", "base_optimizer", "metric", "seed"]
    for identity, group in curves.groupby(keys, dropna=False):
        arms = {arm: part for arm, part in group.groupby("arm")}
        if set(arms) != {"baseline", "wwpgd"}:
            continue
        grid = _common_grid([arms["baseline"], arms["wwpgd"]], points)
        if len(grid) == 0:
            continue
        baseline = arms["baseline"].sort_values("tokens_seen").drop_duplicates("tokens_seen")
        wwpgd = arms["wwpgd"].sort_values("tokens_seen").drop_duplicates("tokens_seen")
        effect = np.interp(grid, wwpgd.tokens_seen, wwpgd.value) - np.interp(
            grid, baseline.tokens_seen, baseline.value
        )
        direction = str(group["beneficial_direction"].iloc[0])
        for token, value in zip(grid, effect, strict=True):
            by_seed_rows.append(
                {
                    "level": identity[0],
                    "token_multiplier": identity[1],
                    "base_optimizer": identity[2],
                    "metric": identity[3],
                    "seed": identity[4],
                    "beneficial_direction": direction,
                    "tokens_seen": float(token),
                    "paired_effect": float(value),
                }
            )
    by_seed = pd.DataFrame(by_seed_rows)
    summary_rows: list[dict[str, Any]] = []
    if by_seed.empty:
        return by_seed, pd.DataFrame()
    group_keys = ["level", "token_multiplier", "base_optimizer", "metric", "beneficial_direction"]
    for identity, group in by_seed.groupby(group_keys, dropna=False):
        seed_curves = [part.rename(columns={"paired_effect": "value"}) for _, part in group.groupby("seed")]
        grid = _common_grid(seed_curves, points)
        if len(grid) == 0:
            continue
        matrix = []
        for part in seed_curves:
            part = part.sort_values("tokens_seen").drop_duplicates("tokens_seen")
            matrix.append(np.interp(grid, part.tokens_seen, part.value))
        values = np.vstack(matrix)
        for index, token in enumerate(grid):
            summary_rows.append(
                {
                    "level": identity[0],
                    "token_multiplier": identity[1],
                    "base_optimizer": identity[2],
                    "metric": identity[3],
                    "beneficial_direction": identity[4],
                    "tokens_seen": float(token),
                    **_summary(values[:, index], seed=_stable_seed("paired-curve", *identity, index)),
                }
            )
    return by_seed, pd.DataFrame(summary_rows)


def collect_selected_checkpoint_effects(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if pairs.empty:
        return pd.DataFrame()
    for pair in pairs.itertuples(index=False):
        baseline = load_selected_checkpoint_metrics(Path(pair.baseline_run))
        wwpgd = load_selected_checkpoint_metrics(Path(pair.wwpgd_run))
        if baseline is None or wwpgd is None or baseline.empty or wwpgd.empty:
            continue
        baseline_row = baseline.iloc[0]
        wwpgd_row = wwpgd.iloc[0]
        for metric, direction in SELECTED_METRICS.items():
            if metric not in baseline_row.index or metric not in wwpgd_row.index:
                continue
            values = _finite_series([baseline_row[metric], wwpgd_row[metric]])
            if len(values) != 2:
                continue
            effect = float(wwpgd_row[metric]) - float(baseline_row[metric])
            rows.append(
                {
                    "layout": pair.layout,
                    "seed": pair.seed,
                    "level": pair.level,
                    "token_multiplier": pair.token_multiplier,
                    "base_optimizer": pair.base_optimizer,
                    "metric": metric,
                    "beneficial_direction": direction,
                    "baseline_value": float(baseline_row[metric]),
                    "wwpgd_value": float(wwpgd_row[metric]),
                    "paired_effect": effect,
                    "wwpgd_better": effect < 0 if direction == "lower" else effect > 0 if direction == "higher" else None,
                }
            )
    return pd.DataFrame(rows)


def summarize_selected_effects(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if effects.empty:
        return pd.DataFrame()
    keys = ["level", "token_multiplier", "base_optimizer", "metric", "beneficial_direction"]
    for identity, group in effects.groupby(keys, dropna=False):
        values = _finite_series(group["paired_effect"])
        summary = _summary(values, seed=_stable_seed("selected", *identity))
        rows.append(
            {
                "level": identity[0],
                "token_multiplier": identity[1],
                "base_optimizer": identity[2],
                "metric": identity[3],
                "beneficial_direction": identity[4],
                **summary,
                "exact_sign_flip_p": _exact_sign_flip_p(values),
                "wwpgd_better_seed_fraction": (
                    float(pd.to_numeric(group["wwpgd_better"], errors="coerce").mean())
                    if group["wwpgd_better"].notna().any()
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_curve_figures(
    arm_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    selected: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for identity, group in arm_summary.groupby(
        ["level", "token_multiplier", "base_optimizer", "metric"], dropna=False
    ):
        figure, axis = plt.subplots(figsize=(8.5, 5.0))
        for arm, part in group.groupby("arm"):
            part = part.sort_values("tokens_seen")
            axis.plot(part.tokens_seen, part["mean"], label=f"{arm} mean")
            axis.fill_between(part.tokens_seen, part.t_ci_low, part.t_ci_high, alpha=0.22, label=f"{arm} 95% t CI")
            axis.plot(part.tokens_seen, part.bollinger_2sd_low, linestyle="--", linewidth=0.8, alpha=0.7)
            axis.plot(part.tokens_seen, part.bollinger_2sd_high, linestyle="--", linewidth=0.8, alpha=0.7, label=f"{arm} mean ± 2 SD")
        axis.set_xlabel("Tokens processed")
        axis.set_ylabel(str(identity[3]).replace("_", " "))
        axis.set_title(f"Level {identity[0]} {identity[2]}: {identity[3]} across seeds")
        axis.legend(fontsize="small")
        figure.tight_layout()
        path = figures_dir / f"level{identity[0]}_{identity[2]}_{identity[3]}_seed_bands.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        outputs.append(path)
    for identity, group in paired_summary.groupby(
        ["level", "token_multiplier", "base_optimizer", "metric"], dropna=False
    ):
        group = group.sort_values("tokens_seen")
        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        axis.plot(group.tokens_seen, group["mean"], label="mean paired effect")
        axis.fill_between(group.tokens_seen, group.t_ci_low, group.t_ci_high, alpha=0.25, label="95% t CI")
        axis.plot(group.tokens_seen, group.bollinger_2sd_low, linestyle="--", linewidth=0.8)
        axis.plot(group.tokens_seen, group.bollinger_2sd_high, linestyle="--", linewidth=0.8, label="mean ± 2 SD")
        axis.axhline(0.0, linewidth=1)
        axis.set_xlabel("Tokens processed")
        axis.set_ylabel(f"WWPGD − baseline {identity[3]}")
        axis.set_title(f"Level {identity[0]} {identity[2]} paired effect across seeds")
        axis.legend(fontsize="small")
        figure.tight_layout()
        path = figures_dir / f"level{identity[0]}_{identity[2]}_{identity[3]}_paired_band.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        outputs.append(path)
    for identity, group in selected.groupby(
        ["level", "token_multiplier", "base_optimizer", "metric"], dropna=False
    ):
        figure, axis = plt.subplots(figsize=(6.5, 5.0))
        for row in group.itertuples(index=False):
            axis.plot([0, 1], [row.baseline_value, row.wwpgd_value], marker="o", alpha=0.8)
        axis.set_xticks([0, 1], ["baseline", "WWPGD"])
        axis.set_ylabel(str(identity[3]).replace("_", " "))
        axis.set_title(f"Level {identity[0]} validation-selected {identity[3]}")
        figure.tight_layout()
        path = figures_dir / f"level{identity[0]}_{identity[2]}_selected_{identity[3]}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        outputs.append(path)
    return outputs


def analyze_seed_results(
    results_root: Path,
    output_dir: Path,
    *,
    figures_dir: Path | None = None,
    level: int | None = None,
    token_multiplier: int | None = None,
    base_optimizer: str | None = None,
    points: int = 200,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir or output_dir / "figures")
    pairs = discover_pairs(
        Path(results_root),
        level=level,
        token_multiplier=token_multiplier,
        base_optimizer=base_optimizer,
    )
    curves = collect_learning_curves(pairs)
    arm_summary = summarize_arm_curves(curves, points=points)
    paired_by_seed, paired_summary = paired_curve_effects(curves, points=points)
    selected = collect_selected_checkpoint_effects(pairs)
    selected_summary = summarize_selected_effects(selected)
    outputs = {
        "seed_pair_inventory.csv": pairs,
        "seed_learning_curves_raw.csv": curves,
        "seed_learning_curve_summary.csv": arm_summary,
        "paired_learning_curve_effects_by_seed.csv": paired_by_seed,
        "paired_learning_curve_effect_summary.csv": paired_summary,
        "selected_checkpoint_effects_by_seed.csv": selected,
        "selected_checkpoint_effect_summary.csv": selected_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    _write_curve_figures(arm_summary, paired_summary, selected, figures_dir)
    return outputs
