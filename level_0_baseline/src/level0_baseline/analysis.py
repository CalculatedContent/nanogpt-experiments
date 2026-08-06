from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SUPPORTED_OPTIMIZERS


OPTIMIZER_LABELS = {
    "sgd_momentum": "SGD + momentum",
    "adamw": "AdamW",
    "muon": "Muon",
}

# Okabe-Ito colorblind-safe palette.
OPTIMIZER_COLORS = {
    "sgd_momentum": "#0072B2",
    "adamw": "#E69F00",
    "muon": "#009E73",
}

_T_975 = {
    1: 12.7062047364,
    2: 4.3026527297,
    3: 3.1824463053,
    4: 2.7764451052,
    5: 2.5705818356,
    6: 2.4469118511,
    7: 2.3646242510,
    8: 2.3060041352,
    9: 2.2621571629,
    10: 2.2281388520,
    11: 2.2009851601,
    12: 2.1788128297,
    13: 2.1603686565,
    14: 2.1447866879,
    15: 2.1314495456,
    16: 2.1199052992,
    17: 2.1098155778,
    18: 2.1009220402,
    19: 2.0930240544,
    20: 2.0859634473,
    21: 2.0796138447,
    22: 2.0738730679,
    23: 2.0686576104,
    24: 2.0638985616,
    25: 2.0595385528,
    26: 2.0555294386,
    27: 2.0518305165,
    28: 2.0484071418,
    29: 2.0452296421,
    30: 2.0422724563,
}


def run_directory(results_root: str | Path, optimizer: str, seed: int) -> Path:
    return Path(results_root) / optimizer / f"seed_{int(seed)}"


def run_is_complete(results_root: str | Path, optimizer: str, seed: int) -> bool:
    path = run_directory(results_root, optimizer, seed) / "run_complete.json"
    if not path.is_file():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8"))["completed"])
    except (KeyError, json.JSONDecodeError, OSError):
        return False


def run_status_table(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337, 2027, 4099),
) -> pd.DataFrame:
    rows = []
    for optimizer in optimizers:
        for seed in seeds:
            run_dir = run_directory(results_root, optimizer, seed)
            completion = run_dir / "run_complete.json"
            payload = (
                json.loads(completion.read_text(encoding="utf-8"))
                if completion.is_file()
                else {}
            )
            rows.append(
                {
                    "optimizer": optimizer,
                    "optimizer_label": OPTIMIZER_LABELS.get(optimizer, optimizer),
                    "seed": int(seed),
                    "complete": bool(payload.get("completed", False)),
                    "steps": payload.get("optimizer_steps", np.nan),
                    "train_epochs": payload.get("train_epochs", np.nan),
                    "final_test_loss": payload.get("final_test_loss", np.nan),
                    "elapsed_seconds": payload.get("elapsed_seconds", np.nan),
                    "run_dir": str(run_dir),
                }
            )
    return pd.DataFrame(rows)


def _require_complete(
    results_root: str | Path, optimizer: str, seed: int, require_complete: bool
) -> None:
    if require_complete and not run_is_complete(results_root, optimizer, seed):
        raise FileNotFoundError(
            f"missing completed run for optimizer={optimizer} seed={seed}: "
            f"{run_directory(results_root, optimizer, seed)}"
        )


def load_metrics(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337, 2027, 4099),
    require_complete: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for optimizer in optimizers:
        for seed in seeds:
            _require_complete(results_root, optimizer, seed, require_complete)
            path = run_directory(results_root, optimizer, seed) / "metrics.csv"
            if not path.is_file():
                if require_complete:
                    raise FileNotFoundError(path)
                continue
            frame = pd.read_csv(path)
            frame["optimizer"] = optimizer
            frame["optimizer_label"] = OPTIMIZER_LABELS.get(optimizer, optimizer)
            frame["seed"] = int(seed)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    result = result.sort_values(["optimizer", "seed", "step"])
    return result.drop_duplicates(["optimizer", "seed", "step"], keep="last")


def load_spectral_metrics(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337, 2027, 4099),
    require_complete: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for optimizer in optimizers:
        for seed in seeds:
            _require_complete(results_root, optimizer, seed, require_complete)
            path = run_directory(results_root, optimizer, seed) / "spectral" / "summary.csv"
            if not path.is_file():
                if require_complete:
                    raise FileNotFoundError(path)
                continue
            frame = pd.read_csv(path)
            frame["optimizer"] = optimizer
            frame["optimizer_label"] = OPTIMIZER_LABELS.get(optimizer, optimizer)
            frame["seed"] = int(seed)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    result = result.sort_values(["optimizer", "seed", "step"])
    return result.drop_duplicates(["optimizer", "seed", "step"], keep="last")


def bollinger_band(
    frame: pd.DataFrame,
    value: str,
    *,
    x: str = "epoch",
    sigma: float = 2.0,
    group: Sequence[str] = ("optimizer",),
) -> pd.DataFrame:
    """Across-seed mean and mean ± sigma·sample-SD envelope.

    This is a Bollinger-style ensemble envelope, not a rolling time-series
    smoother. All seeds are evaluated on the same fixed step grid.
    """
    if frame.empty:
        return pd.DataFrame()
    required = set(group) | {x, value, "seed"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing columns for band calculation: {sorted(missing)}")
    subset = frame[list(group) + [x, value, "seed"]].copy()
    subset[value] = pd.to_numeric(subset[value], errors="coerce")
    subset = subset.dropna(subset=[x, value])
    keys = list(group) + [x]
    aggregate = (
        subset.groupby(keys, as_index=False)[value]
        .agg(mean="mean", sd=lambda values: values.std(ddof=1), n="count")
        .sort_values(keys)
    )
    aggregate["sd"] = aggregate["sd"].fillna(0.0)
    aggregate["lower"] = aggregate["mean"] - float(sigma) * aggregate["sd"]
    aggregate["upper"] = aggregate["mean"] + float(sigma) * aggregate["sd"]
    aggregate["sigma"] = float(sigma)
    return aggregate


def mean_ci95(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "ci95_half_width": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
        }
    mean = float(array.mean())
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "sd": np.nan,
            "sem": np.nan,
            "ci95_half_width": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
        }
    sd = float(array.std(ddof=1))
    sem = sd / math.sqrt(n)
    critical = _T_975.get(n - 1, 1.9599639845)
    half_width = critical * sem
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_half_width": half_width,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def load_test_results(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337, 2027, 4099),
    require_complete: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for optimizer in optimizers:
        for seed in seeds:
            _require_complete(results_root, optimizer, seed, require_complete)
            path = run_directory(results_root, optimizer, seed) / "test_results.json"
            if not path.is_file():
                if require_complete:
                    raise FileNotFoundError(path)
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for checkpoint in ("final", "validation_selected"):
                values = payload[checkpoint]
                rows.append(
                    {
                        "optimizer": optimizer,
                        "optimizer_label": OPTIMIZER_LABELS.get(optimizer, optimizer),
                        "seed": int(seed),
                        "checkpoint": checkpoint,
                        "step": int(values["step"]),
                        "test_loss": float(values["loss"]),
                        "test_perplexity": float(values["perplexity"]),
                        "test_accuracy": float(values["accuracy"]),
                    }
                )
    return pd.DataFrame(rows)


def test_summary_table(test_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (optimizer, checkpoint), group in test_results.groupby(
        ["optimizer", "checkpoint"], sort=False
    ):
        for metric in ("test_loss", "test_perplexity", "test_accuracy"):
            statistics = mean_ci95(group[metric])
            rows.append(
                {
                    "optimizer": optimizer,
                    "optimizer_label": OPTIMIZER_LABELS.get(optimizer, optimizer),
                    "checkpoint": checkpoint,
                    "metric": metric,
                    **statistics,
                }
            )
    return pd.DataFrame(rows)


def validate_protocol_identity(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337, 2027, 4099),
) -> dict[str, object]:
    manifests = []
    for optimizer in optimizers:
        for seed in seeds:
            path = run_directory(results_root, optimizer, seed) / "manifest.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
    model_signatures = {
        json.dumps(manifest["config"]["model"], sort_keys=True)
        for manifest in manifests
    }
    data_signatures = {
        json.dumps(manifest["data_manifest"], sort_keys=True)
        for manifest in manifests
    }
    token_budgets = {manifest["planned_training_tokens"] for manifest in manifests}
    split_sizes = {
        json.dumps(manifest["split_sizes"], sort_keys=True) for manifest in manifests
    }
    result = {
        "runs": len(manifests),
        "same_model": len(model_signatures) == 1,
        "same_data_manifest": len(data_signatures) == 1,
        "same_training_token_budget": len(token_budgets) == 1,
        "same_split_sizes": len(split_sizes) == 1,
    }
    result["valid"] = all(
        bool(result[key])
        for key in (
            "same_model",
            "same_data_manifest",
            "same_training_token_budget",
            "same_split_sizes",
        )
    )
    if not result["valid"]:
        raise RuntimeError(f"protocol identity check failed: {result}")
    return result


def _metric_axis_label(metric: str) -> str:
    return metric.replace("_", " ").title()


def plot_optimizer_band(
    frame: pd.DataFrame,
    *,
    optimizer: str,
    metric: str,
    x: str = "epoch",
    sigma: float = 2.0,
    title: str | None = None,
):
    subset = frame[frame["optimizer"] == optimizer]
    band = bollinger_band(subset, metric, x=x, sigma=sigma)
    if band.empty:
        raise ValueError(f"no data for optimizer={optimizer} metric={metric}")
    figure, axis = plt.subplots(figsize=(9, 5))
    color = OPTIMIZER_COLORS[optimizer]
    axis.plot(band[x], band["mean"], color=color, label=OPTIMIZER_LABELS[optimizer])
    axis.fill_between(
        band[x], band["lower"], band["upper"], color=color, alpha=0.22
    )
    axis.set_xlabel(_metric_axis_label(x))
    axis.set_ylabel(_metric_axis_label(metric))
    axis.set_title(title or f"{OPTIMIZER_LABELS[optimizer]}: {metric}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure, axis, band


def plot_overlay_band(
    frame: pd.DataFrame,
    *,
    metric: str,
    x: str = "epoch",
    sigma: float = 2.0,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    title: str | None = None,
):
    figure, axis = plt.subplots(figsize=(9, 5))
    for optimizer in optimizers:
        subset = frame[frame["optimizer"] == optimizer]
        if subset.empty:
            continue
        band = bollinger_band(subset, metric, x=x, sigma=sigma)
        color = OPTIMIZER_COLORS[optimizer]
        axis.plot(
            band[x], band["mean"], color=color, label=OPTIMIZER_LABELS[optimizer]
        )
        axis.fill_between(
            band[x], band["lower"], band["upper"], color=color, alpha=0.16
        )
    axis.set_xlabel(_metric_axis_label(x))
    axis.set_ylabel(_metric_axis_label(metric))
    axis.set_title(title or f"Optimizer comparison: {metric}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure, axis


def plot_final_test_ci(
    summary: pd.DataFrame,
    *,
    metric: str,
    checkpoint: str = "final",
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
):
    subset = summary[
        (summary["metric"] == metric) & (summary["checkpoint"] == checkpoint)
    ].set_index("optimizer")
    ordered = [optimizer for optimizer in optimizers if optimizer in subset.index]
    means = [float(subset.loc[optimizer, "mean"]) for optimizer in ordered]
    errors = [
        float(subset.loc[optimizer, "ci95_half_width"]) for optimizer in ordered
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(ordered))
    axis.bar(
        positions,
        means,
        yerr=errors,
        capsize=5,
        color=[OPTIMIZER_COLORS[optimizer] for optimizer in ordered],
        alpha=0.85,
    )
    axis.set_xticks(
        positions, [OPTIMIZER_LABELS[optimizer] for optimizer in ordered]
    )
    axis.set_ylabel(_metric_axis_label(metric))
    axis.set_title(f"Final test {metric}: mean and 95% Student-t CI")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure, axis

# Prevent pytest from treating this public analysis helper as a test when it is
# imported into a test module.
test_summary_table.__test__ = False
