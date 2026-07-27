from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = {
    "test_loss": "lower",
    "test_perplexity": "lower",
    "test_accuracy": "higher",
    "validation_loss": "lower",
    "validation_perplexity": "lower",
    "validation_accuracy": "higher",
    "train_validation_gap": "lower",
    "train_test_gap": "lower",
}


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _parameter_count(manifest: dict[str, Any]) -> float:
    report = manifest.get("parameter_report") or {}
    for value in (
        manifest.get("selected_parameter_count"),
        manifest.get("parameter_count_used"),
        report.get("transformer_body_parameters"),
        report.get("non_embedding_parameters"),
        report.get("total_parameters"),
    ):
        number = _finite_float(value)
        if math.isfinite(number):
            return number
    return math.nan


def collect_selected_checkpoint_runs(results_root: Path) -> pd.DataFrame:
    """Discover complete schema-v3 selected-checkpoint results at every level."""
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(Path(results_root).rglob("manifest.json")):
        run_dir = manifest_path.parent
        selected_path = run_dir / "selected_checkpoint_metrics.json"
        if not (run_dir / "run_complete.json").is_file() or not selected_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            selected = json.loads(selected_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if int(manifest.get("scientific_schema_version", 0) or 0) < 3:
            continue
        if manifest.get("valid_for_science", True) is not True:
            continue
        base = str(manifest.get("base_optimizer") or "").replace(
            "stable_adamw", "stableadamw"
        )
        extension = str(manifest.get("extension") or "none")
        if not base:
            arm = str(manifest.get("arm_name") or manifest.get("optimizer") or "")
            base = arm.removesuffix("_wwpgd").replace("stable_adamw", "stableadamw")
            extension = "wwpgd" if arm.endswith("_wwpgd") else "none"
        if extension not in {"none", "wwpgd"}:
            continue
        row: dict[str, Any] = {
            "run_dir": str(run_dir),
            "pair_id": manifest.get("pair_id"),
            "seed": manifest.get("seed"),
            "level": manifest.get("level"),
            "token_multiplier": manifest.get("token_multiplier"),
            "base_optimizer": base,
            "extension": extension,
            "parameter_count": _parameter_count(manifest),
            "selected_checkpoint_step": selected.get(
                "selected_checkpoint_step", selected.get("selected_step")
            ),
        }
        for metric in METRICS:
            row[metric] = _finite_float(selected.get(metric))
        rows.append(row)
    return pd.DataFrame(rows)


def pair_selected_checkpoint_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pair_id",
        "seed",
        "level",
        "token_multiplier",
        "base_optimizer",
        "parameter_count",
        "metric",
        "beneficial_direction",
        "baseline_value",
        "wwpgd_value",
        "paired_effect",
        "wwpgd_better",
    ]
    if runs.empty:
        return pd.DataFrame(columns=columns)
    keys = ["level", "token_multiplier", "base_optimizer", "seed"]
    output: list[dict[str, Any]] = []
    for identity, group in runs.groupby(keys, dropna=False):
        by_extension = {
            str(row.extension): row
            for row in group.itertuples(index=False)
            if str(row.extension) in {"none", "wwpgd"}
        }
        if set(by_extension) != {"none", "wwpgd"}:
            continue
        baseline = by_extension["none"]
        wwpgd = by_extension["wwpgd"]
        for metric, direction in METRICS.items():
            baseline_value = _finite_float(getattr(baseline, metric))
            wwpgd_value = _finite_float(getattr(wwpgd, metric))
            if not (math.isfinite(baseline_value) and math.isfinite(wwpgd_value)):
                continue
            effect = wwpgd_value - baseline_value
            output.append(
                {
                    "pair_id": getattr(
                        wwpgd, "pair_id", getattr(baseline, "pair_id", None)
                    ),
                    "seed": identity[3],
                    "level": identity[0],
                    "token_multiplier": identity[1],
                    "base_optimizer": identity[2],
                    "parameter_count": _finite_float(
                        getattr(wwpgd, "parameter_count")
                    ),
                    "metric": metric,
                    "beneficial_direction": direction,
                    "baseline_value": baseline_value,
                    "wwpgd_value": wwpgd_value,
                    "paired_effect": effect,
                    "wwpgd_better": effect < 0 if direction == "lower" else effect > 0,
                }
            )
    return pd.DataFrame(output, columns=columns)


def _bootstrap_interval(
    values: np.ndarray, *, seed: int, draws: int = 10_000
) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_flip_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    count = len(values)
    if count == 0:
        return math.nan
    observed = abs(float(values.mean()))
    if count <= 20:
        means = [
            abs(float(np.mean(values * np.asarray(signs, dtype=float))))
            for signs in itertools.product((-1.0, 1.0), repeat=count)
        ]
        return float(sum(value >= observed - 1e-15 for value in means) / len(means))
    rng = np.random.default_rng(_stable_seed("sign-flip", *values.tolist()))
    signs = rng.choice((-1.0, 1.0), size=(100_000, count))
    means = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(means >= observed) + 1) / (len(means) + 1))


def summarize_paired_effects(paired: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "level",
        "token_multiplier",
        "base_optimizer",
        "parameter_count",
        "metric",
        "beneficial_direction",
        "n_pairs",
        "mean_paired_effect",
        "median_paired_effect",
        "sample_std",
        "standard_error",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "exact_sign_flip_p",
        "wwpgd_better_seed_fraction",
    ]
    if paired.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_keys = [
        "level",
        "token_multiplier",
        "base_optimizer",
        "metric",
        "beneficial_direction",
    ]
    for identity, group in paired.groupby(group_keys, dropna=False):
        values = (
            pd.to_numeric(group["paired_effect"], errors="coerce")
            .dropna()
            .to_numpy(float)
        )
        if len(values) == 0:
            continue
        standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        low, high = _bootstrap_interval(values, seed=_stable_seed(*identity))
        rows.append(
            {
                "level": identity[0],
                "token_multiplier": identity[1],
                "base_optimizer": identity[2],
                "parameter_count": pd.to_numeric(
                    group["parameter_count"], errors="coerce"
                ).mean(),
                "metric": identity[3],
                "beneficial_direction": identity[4],
                "n_pairs": len(values),
                "mean_paired_effect": float(values.mean()),
                "median_paired_effect": float(np.median(values)),
                "sample_std": standard_deviation,
                "standard_error": standard_deviation / math.sqrt(len(values)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "exact_sign_flip_p": _exact_sign_flip_p(values),
                "wwpgd_better_seed_fraction": float(group["wwpgd_better"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def model_size_trends(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "token_multiplier",
        "base_optimizer",
        "metric",
        "beneficial_direction",
        "level_count",
        "min_level",
        "max_level",
        "effect_slope_per_level",
        "trend_direction",
        "trend_is_descriptive_only",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for identity, group in summary.groupby(
        ["token_multiplier", "base_optimizer", "metric", "beneficial_direction"],
        dropna=False,
    ):
        clean = (
            group[["level", "mean_paired_effect"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .groupby("level", as_index=False)["mean_paired_effect"]
            .mean()
            .sort_values("level")
        )
        if len(clean) < 2:
            continue
        slope = float(np.polyfit(clean["level"], clean["mean_paired_effect"], 1)[0])
        direction = identity[3]
        favorable = slope < 0 if direction == "lower" else slope > 0
        rows.append(
            {
                "token_multiplier": identity[0],
                "base_optimizer": identity[1],
                "metric": identity[2],
                "beneficial_direction": direction,
                "level_count": int(clean["level"].nunique()),
                "min_level": int(clean["level"].min()),
                "max_level": int(clean["level"].max()),
                "effect_slope_per_level": slope,
                "trend_direction": (
                    "more_favorable_with_level"
                    if favorable
                    else "less_favorable_with_level"
                ),
                "trend_is_descriptive_only": True,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def scaling_readiness(runs: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    levels = int(runs["level"].nunique()) if not runs.empty else 0
    multipliers = int(runs["token_multiplier"].nunique()) if not runs.empty else 0
    seeds = int(paired["seed"].nunique()) if not paired.empty else 0
    return pd.DataFrame(
        [
            {
                "levels_found": levels,
                "token_multipliers_found": multipliers,
                "complete_paired_seeds_found": seeds,
                "ready_for_descriptive_level_trend": levels >= 3 and seeds >= 1,
                "ready_for_scaling_law_fit": (
                    levels >= 3 and multipliers >= 2 and seeds >= 3
                ),
                "note": (
                    "Cross-level effects are descriptive; a scaling-law fit requires "
                    "at least three levels, two token multipliers, and three complete "
                    "paired seeds."
                ),
            }
        ]
    )


def _write_figures(summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("test_loss", "test_perplexity", "test_accuracy"):
        frame = summary[summary["metric"].eq(metric)].copy()
        if frame.empty:
            continue
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for (base, multiplier), group in frame.groupby(
            ["base_optimizer", "token_multiplier"]
        ):
            group = group.sort_values("level")
            error = np.vstack(
                [
                    group["mean_paired_effect"] - group["bootstrap_ci_low"],
                    group["bootstrap_ci_high"] - group["mean_paired_effect"],
                ]
            )
            axis.errorbar(
                group["level"],
                group["mean_paired_effect"],
                yerr=error,
                marker="o",
                label=f"{base}, D/N={multiplier}",
            )
        axis.axhline(0.0, linewidth=1)
        axis.set_xlabel("Model level")
        axis.set_ylabel(f"WWPGD − baseline {metric}")
        axis.set_title(f"Paired {metric} effect across model levels")
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures_dir / f"cross_level_{metric}.png", dpi=160)
        plt.close(figure)


def analyze_cross_level_effects(
    results_root: Path,
    output_dir: Path,
    *,
    figures_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir) if figures_dir is not None else output_dir / "figures"
    runs = collect_selected_checkpoint_runs(Path(results_root))
    paired = pair_selected_checkpoint_metrics(runs)
    summary = summarize_paired_effects(paired)
    trends = model_size_trends(summary)
    readiness = scaling_readiness(runs, paired)
    outputs = {
        "cross_level_run_inventory.csv": runs,
        "cross_level_paired_effects_by_seed.csv": paired,
        "cross_level_paired_effect_summary.csv": summary,
        "cross_level_model_size_trends.csv": trends,
        "cross_level_scaling_readiness.csv": readiness,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    _write_figures(summary, figures_dir)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze paired WWPGD effects across model levels"
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path)
    args = parser.parse_args()
    outputs = analyze_cross_level_effects(
        args.results_root, args.output_dir, figures_dir=args.figures_dir
    )
    readiness = outputs["cross_level_scaling_readiness.csv"].iloc[0].to_dict()
    print(json.dumps(readiness, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
