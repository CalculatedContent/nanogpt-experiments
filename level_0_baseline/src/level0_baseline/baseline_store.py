from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import (
    OPTIMIZER_COLORS,
    OPTIMIZER_LABELS,
    load_metrics,
    load_spectral_metrics,
    load_test_results,
    run_directory,
    test_summary_table,
)
from .config import DEFAULT_ROOT, SUPPORTED_OPTIMIZERS


STORE_SCHEMA_VERSION = 1

TRAJECTORY_METRICS = (
    "train_loss",
    "train_perplexity",
    "train_accuracy",
    "val_loss",
    "val_perplexity",
    "val_accuracy",
    "val_generalization_gap",
    "grad_norm_pre_clip",
    "grad_norm_post_clip",
    "weight_norm",
    "update_norm_since_eval",
    "update_to_weight_ratio",
    "tokens_per_sec",
)

EPOCH_METRICS = (
    "train_loss",
    "train_perplexity",
    "train_accuracy",
    "val_loss",
    "val_perplexity",
    "val_accuracy",
    "test_loss",
    "test_perplexity",
    "test_accuracy",
    "val_generalization_gap",
    "test_generalization_gap",
    "grad_norm_pre_clip",
    "grad_norm_post_clip",
    "weight_norm",
    "update_norm_since_eval",
    "update_to_weight_ratio",
    "tokens_per_sec",
)

SPECTRAL_METRICS = (
    "alpha_mean",
    "alpha_median",
    "alpha_weighted_mean",
    "alpha_weighted_median",
    "ERG_gap_mean",
    "ERG_gap_median",
    "D_mean",
    "D_median",
    "stable_rank_mean",
    "stable_rank_median",
    "mp_softrank_mean",
    "mp_softrank_median",
    "log_norm_mean",
    "log_norm_median",
    "log_spectral_norm_mean",
    "log_spectral_norm_median",
    "entropy_mean",
    "entropy_median",
    "num_pl_spikes_mean",
    "num_ERG_spikes_mean",
    "rank_loss_mean",
)


def default_baseline_store_root() -> Path:
    root = Path(os.getenv("NANOGPT_LEVEL0_ROOT", DEFAULT_ROOT))
    return Path(
        os.getenv(
            "NANOGPT_LEVEL0_BASELINE_STORE",
            root / "baseline_reference",
        )
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def _atomic_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def load_epoch_metrics(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337, 2027, 4099),
    require_complete: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for optimizer in optimizers:
        for seed in seeds:
            run_dir = run_directory(results_root, optimizer, seed)
            completion = run_dir / "run_complete.json"
            if require_complete and not completion.is_file():
                raise FileNotFoundError(
                    f"missing completion marker for optimizer={optimizer} seed={seed}"
                )
            path = run_dir / "epoch_metrics.csv"
            if not path.is_file():
                if require_complete:
                    raise FileNotFoundError(
                        f"missing preregistered epoch metrics: {path}"
                    )
                continue
            frame = pd.read_csv(path)
            frame["optimizer"] = optimizer
            frame["optimizer_label"] = OPTIMIZER_LABELS.get(
                optimizer, optimizer
            )
            frame["seed"] = int(seed)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    sort_columns = [
        column
        for column in ("optimizer", "seed", "nominal_epoch", "step")
        if column in result.columns
    ]
    result = result.sort_values(sort_columns)
    dedupe = [
        column
        for column in ("optimizer", "seed", "nominal_epoch")
        if column in result.columns
    ]
    return result.drop_duplicates(dedupe, keep="last")


def bollinger_summary_long(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    *,
    x: str,
    group: Sequence[str] = ("optimizer", "optimizer_label"),
    sigma: float = 2.0,
) -> pd.DataFrame:
    """Return long-form mean ± sigma sample-SD summaries."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *group,
                x,
                "metric",
                "n",
                "mean",
                "sd",
                "lower",
                "upper",
                "sigma",
            ]
        )
    missing_group = set(group).difference(frame.columns)
    if missing_group:
        raise KeyError(f"missing group columns: {sorted(missing_group)}")
    if x not in frame.columns:
        raise KeyError(f"missing x column: {x}")

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in frame.columns:
            continue
        subset = frame[[*group, x, metric]].copy()
        subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
        subset[x] = pd.to_numeric(subset[x], errors="coerce")
        subset = subset.dropna(subset=[x, metric])
        if subset.empty:
            continue
        for keys, values in subset.groupby([*group, x], sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            array = pd.to_numeric(values[metric], errors="coerce").dropna()
            n = int(len(array))
            mean = float(array.mean())
            sd = float(array.std(ddof=1)) if n > 1 else 0.0
            row = {
                column: value for column, value in zip([*group, x], keys)
            }
            row.update(
                {
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "sd": sd,
                    "lower": mean - float(sigma) * sd,
                    "upper": mean + float(sigma) * sd,
                    "sigma": float(sigma),
                }
            )
            rows.append(row)
    columns = [
        *group,
        x,
        "metric",
        "n",
        "mean",
        "sd",
        "lower",
        "upper",
        "sigma",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        [*group, "metric", x], ignore_index=True
    )


def _protocol_manifest_rows(
    results_root: str | Path, optimizer: str, seeds: Sequence[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = run_directory(results_root, optimizer, seed)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "optimizer": optimizer,
                "seed": int(seed),
                "run_dir": str(run_dir.resolve()),
                "protocol_fingerprint": manifest["protocol_fingerprint"],
                "planned_training_tokens": manifest["planned_training_tokens"],
                "planned_train_epochs": manifest["planned_train_epochs"],
                "tokens_per_optimizer_step": manifest[
                    "tokens_per_optimizer_step"
                ],
                "model": manifest["config"]["model"],
                "data_manifest": manifest["data_manifest"],
            }
        )
    return rows


def _write_optimizer_exports(
    *,
    store_root: Path,
    optimizer: str,
    metrics: pd.DataFrame,
    epoch_metrics: pd.DataFrame,
    spectral: pd.DataFrame,
    terminal_test: pd.DataFrame,
    sigma: float,
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    optimizer_root = store_root / "per_optimizer" / optimizer
    paths = {
        "trajectory_runs": optimizer_root / "trajectory_runs.csv",
        "epoch_runs": optimizer_root / "epoch_runs.csv",
        "spectral_runs": optimizer_root / "spectral_runs.csv",
        "terminal_test_runs": optimizer_root / "terminal_test_runs.csv",
        "trajectory_summary": optimizer_root
        / "trajectory_bollinger_summary.csv",
        "epoch_summary": optimizer_root / "epoch_bollinger_summary.csv",
        "spectral_summary": optimizer_root
        / "spectral_bollinger_summary.csv",
        "terminal_test_summary": optimizer_root
        / "terminal_test_student_t_summary.csv",
        "manifest": optimizer_root / "manifest.json",
    }
    _atomic_csv(metrics, paths["trajectory_runs"])
    _atomic_csv(epoch_metrics, paths["epoch_runs"])
    _atomic_csv(spectral, paths["spectral_runs"])
    _atomic_csv(terminal_test, paths["terminal_test_runs"])
    _atomic_csv(
        bollinger_summary_long(
            metrics, TRAJECTORY_METRICS, x="epoch", sigma=sigma
        ),
        paths["trajectory_summary"],
    )
    _atomic_csv(
        bollinger_summary_long(
            epoch_metrics,
            EPOCH_METRICS,
            x="nominal_epoch",
            sigma=sigma,
        ),
        paths["epoch_summary"],
    )
    _atomic_csv(
        bollinger_summary_long(
            spectral, SPECTRAL_METRICS, x="epoch", sigma=sigma
        ),
        paths["spectral_summary"],
    )
    _atomic_csv(
        test_summary_table(terminal_test),
        paths["terminal_test_summary"],
    )
    _atomic_json(
        {
            "schema_version": STORE_SCHEMA_VERSION,
            "optimizer": optimizer,
            "optimizer_label": OPTIMIZER_LABELS.get(optimizer, optimizer),
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": sorted(int(row["seed"]) for row in manifest_rows),
            "source_runs": manifest_rows,
            "bollinger_definition": "mean +/- 2 sample standard deviations across seeds",
            "test_epoch_policy": (
                "preregistered integer-epoch monitoring; test metrics are "
                "never used for optimizer updates or checkpoint selection"
            ),
            "files": {
                key: str(path.relative_to(store_root))
                for key, path in paths.items()
                if key != "manifest"
            },
        },
        paths["manifest"],
    )
    return paths


def _concat_available(
    store_root: Path, relative_path: str, optimizers: Sequence[str]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for optimizer in optimizers:
        path = store_root / "per_optimizer" / optimizer / relative_path
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def refresh_common_store(
    store_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
) -> dict[str, Path]:
    store_root = Path(store_root)
    available = [
        optimizer
        for optimizer in optimizers
        if (store_root / "per_optimizer" / optimizer / "manifest.json").is_file()
    ]
    missing = [optimizer for optimizer in optimizers if optimizer not in available]
    all_runs_root = store_root / "all_runs"
    summaries_root = store_root / "summaries"
    paths = {
        "trajectory_runs": all_runs_root / "trajectory_metrics.csv",
        "epoch_runs": all_runs_root / "epoch_checkpoint_metrics.csv",
        "spectral_runs": all_runs_root / "spectral_metrics.csv",
        "terminal_test_runs": all_runs_root / "terminal_test_metrics.csv",
        "trajectory_summary": summaries_root
        / "trajectory_bollinger_summary.csv",
        "epoch_summary": summaries_root / "epoch_bollinger_summary.csv",
        "spectral_summary": summaries_root
        / "spectral_bollinger_summary.csv",
        "terminal_test_summary": summaries_root
        / "terminal_test_student_t_summary.csv",
        "manifest": store_root / "store_manifest.json",
    }
    mappings = {
        "trajectory_runs": "trajectory_runs.csv",
        "epoch_runs": "epoch_runs.csv",
        "spectral_runs": "spectral_runs.csv",
        "terminal_test_runs": "terminal_test_runs.csv",
        "trajectory_summary": "trajectory_bollinger_summary.csv",
        "epoch_summary": "epoch_bollinger_summary.csv",
        "spectral_summary": "spectral_bollinger_summary.csv",
        "terminal_test_summary": "terminal_test_student_t_summary.csv",
    }
    for key, relative in mappings.items():
        _atomic_csv(
            _concat_available(store_root, relative, available),
            paths[key],
        )
    _atomic_json(
        {
            "schema_version": STORE_SCHEMA_VERSION,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "available_optimizers": available,
            "missing_optimizers": missing,
            "optimizer_labels": {
                optimizer: OPTIMIZER_LABELS.get(optimizer, optimizer)
                for optimizer in optimizers
            },
            "files": {
                key: str(path.relative_to(store_root))
                for key, path in paths.items()
                if key != "manifest"
            },
        },
        paths["manifest"],
    )
    return paths


def export_optimizer_core_results(
    results_root: str | Path,
    store_root: str | Path,
    *,
    optimizer: str,
    seeds: Sequence[int] = (1337, 2027, 4099),
    sigma: float = 2.0,
) -> dict[str, Path]:
    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"unsupported optimizer: {optimizer}")
    store_root = Path(store_root)
    metrics = load_metrics(
        results_root,
        optimizers=[optimizer],
        seeds=seeds,
        require_complete=True,
    )
    epoch_metrics = load_epoch_metrics(
        results_root,
        optimizers=[optimizer],
        seeds=seeds,
        require_complete=True,
    )
    spectral = load_spectral_metrics(
        results_root,
        optimizers=[optimizer],
        seeds=seeds,
        require_complete=True,
    )
    terminal_test = load_test_results(
        results_root,
        optimizers=[optimizer],
        seeds=seeds,
        require_complete=True,
    )
    manifests = _protocol_manifest_rows(results_root, optimizer, seeds)
    optimizer_paths = _write_optimizer_exports(
        store_root=store_root,
        optimizer=optimizer,
        metrics=metrics,
        epoch_metrics=epoch_metrics,
        spectral=spectral,
        terminal_test=terminal_test,
        sigma=sigma,
        manifest_rows=manifests,
    )
    refresh_common_store(store_root)
    return optimizer_paths


def load_common_baseline_store(
    store_root: str | Path,
    *,
    require_optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
) -> dict[str, Any]:
    store_root = Path(store_root)
    manifest_path = store_root / "store_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"baseline reference store has not been exported: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = set(manifest.get("available_optimizers", []))
    missing = [optimizer for optimizer in require_optimizers if optimizer not in available]
    if missing:
        raise RuntimeError(
            "baseline reference store is incomplete; run the missing optimizer "
            f"notebooks first: {missing}"
        )
    files = {
        key: store_root / relative
        for key, relative in manifest["files"].items()
    }
    return {
        "root": store_root,
        "manifest": manifest,
        **{key: pd.read_csv(path) for key, path in files.items()},
    }


def plot_bollinger_summary(
    summary: pd.DataFrame,
    *,
    metric: str,
    x: str,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    title: str | None = None,
):
    subset = summary[summary["metric"] == metric].copy()
    if subset.empty:
        raise ValueError(f"no exported summary rows for metric={metric}")
    figure, axis = plt.subplots(figsize=(9, 5))
    for optimizer in optimizers:
        group = subset[subset["optimizer"] == optimizer].sort_values(x)
        if group.empty:
            continue
        color = OPTIMIZER_COLORS[optimizer]
        axis.plot(
            group[x],
            group["mean"],
            color=color,
            label=OPTIMIZER_LABELS[optimizer],
        )
        axis.fill_between(
            group[x],
            group["lower"],
            group["upper"],
            color=color,
            alpha=0.16,
        )
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title(title or f"Baseline comparison: {metric}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure, axis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--store-root", default=str(default_baseline_store_root())
    )
    parser.add_argument(
        "--optimizers", default=",".join(SUPPORTED_OPTIMIZERS)
    )
    parser.add_argument("--seeds", default="1337,2027,4099")
    parser.add_argument("--sigma", type=float, default=2.0)
    args = parser.parse_args()
    optimizers = [
        value.strip() for value in args.optimizers.split(",") if value.strip()
    ]
    seeds = [
        int(value.strip()) for value in args.seeds.split(",") if value.strip()
    ]
    for optimizer in optimizers:
        export_optimizer_core_results(
            args.results_root,
            args.store_root,
            optimizer=optimizer,
            seeds=seeds,
            sigma=args.sigma,
        )
    print(json.dumps(refresh_common_store(args.store_root), default=str, indent=2))


if __name__ == "__main__":
    main()
