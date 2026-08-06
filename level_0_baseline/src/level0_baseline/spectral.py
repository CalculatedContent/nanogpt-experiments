from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn

from .model import GPT, transformer_matrix_items


SPECTRAL_METRICS = (
    "alpha",
    "alpha_weighted",
    "ERG_gap",
    "D",
    "stable_rank",
    "mp_softrank",
    "log_norm",
    "log_spectral_norm",
    "entropy",
    "Lambda",
    "num_pl_spikes",
    "num_ERG_spikes",
    "rank_loss",
)


class _WeightMatrixHolder(nn.Module):
    """CPU-only linear-layer view of the transformer block matrices."""

    def __init__(self, model: GPT):
        super().__init__()
        self.matrix_metadata: list[dict[str, object]] = []
        for name, matrix_type, block_index, weight in transformer_matrix_items(model):
            layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            layer.weight = nn.Parameter(
                weight.detach().float().cpu().clone(), requires_grad=False
            )
            self.add_module(name, layer)
            self.matrix_metadata.append(
                {
                    "matrix_name": name,
                    "matrix_type": matrix_type,
                    "block": block_index,
                }
            )


def attach_matrix_metadata(
    frame: pd.DataFrame, metadata: list[dict[str, object]]
) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    names = [str(item["matrix_name"]) for item in metadata]
    resolved: list[str | None] = [None] * len(result)
    for row_index, row in result.iterrows():
        text = " ".join(
            str(row.get(column, "")) for column in ("longname", "name")
        )
        for name in names:
            if name in text:
                resolved[row_index] = name
                break
    if any(name is None for name in resolved) and len(result) == len(metadata):
        order = list(range(len(result)))
        if "layer_id" in result.columns:
            numeric = pd.to_numeric(result["layer_id"], errors="coerce")
            if numeric.notna().all():
                order = list(numeric.sort_values().index)
        for metadata_index, row_index in enumerate(order):
            resolved[row_index] = names[metadata_index]
    if any(name is None for name in resolved):
        raise RuntimeError(
            "WeightWatcher rows could not be matched to all transformer matrices"
        )
    by_name = {str(item["matrix_name"]): item for item in metadata}
    result.insert(0, "matrix_name", resolved)
    result.insert(
        1,
        "matrix_type",
        [by_name[str(name)]["matrix_type"] for name in resolved],
    )
    result.insert(2, "block", [by_name[str(name)]["block"] for name in resolved])
    return result


def summarize_spectral_frame(
    frame: pd.DataFrame, *, step: int, tokens_seen: int, epoch: float
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "epoch": float(epoch),
        "n_layers": int(len(frame)),
    }
    for metric in SPECTRAL_METRICS:
        values = (
            pd.to_numeric(frame[metric], errors="coerce")
            if metric in frame.columns
            else pd.Series(dtype=float)
        )
        finite = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
        summary[f"{metric}_n"] = int(len(finite))
        if len(finite):
            summary[f"{metric}_mean"] = float(finite.mean())
            summary[f"{metric}_median"] = float(finite.median())
            summary[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
            )
            summary[f"{metric}_min"] = float(finite.min())
            summary[f"{metric}_max"] = float(finite.max())
        else:
            for statistic in ("mean", "median", "std", "min", "max"):
                summary[f"{metric}_{statistic}"] = float("nan")
    return summary


def _append_dataframe(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    else:
        combined = frame
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def run_weightwatcher(
    model: GPT,
    run_dir: Path,
    *,
    step: int,
    tokens_seen: int,
    train_tokens: int,
    analysis_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run WeightWatcher with ERG enabled and persist raw/layer/summary tables.

    No alpha or ERG fallback is synthesized. Missing values remain NaN and are
    visible through the per-metric valid-layer counts.
    """
    try:
        import weightwatcher as ww
    except ImportError as exc:
        raise RuntimeError(
            "WeightWatcher is required; run scripts/setup_mac.sh or install the analysis extra"
        ) from exc

    spectral_root = run_dir / "spectral"
    raw_root = spectral_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    holder = _WeightMatrixHolder(model)
    watcher = ww.WeightWatcher(model=holder)
    kwargs: dict[str, Any] = {
        "plot": False,
        "randomize": bool(analysis_cfg.get("randomize", False)),
        "min_evals": int(analysis_cfg.get("min_evals", 20)),
    }
    if bool(analysis_cfg.get("ERG", True)):
        kwargs["ERG"] = True

    try:
        details = watcher.analyze(**kwargs)
        if details is None or len(details) == 0:
            raise RuntimeError("WeightWatcher returned no transformer-matrix rows")
        frame = attach_matrix_metadata(pd.DataFrame(details), holder.matrix_metadata)
        epoch = tokens_seen / max(1, int(train_tokens))
        frame.insert(0, "step", int(step))
        frame.insert(1, "tokens_seen", int(tokens_seen))
        frame.insert(2, "epoch", float(epoch))
        raw_path = raw_root / f"weightwatcher_step_{step:07d}.csv"
        frame.to_csv(raw_path, index=False)
        _append_dataframe(spectral_root / "layers.csv", frame)

        summary = summarize_spectral_frame(
            frame, step=step, tokens_seen=tokens_seen, epoch=epoch
        )
        _append_dataframe(
            spectral_root / "summary.csv", pd.DataFrame([summary])
        )
        status = {
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "completed": True,
            "raw_path": str(raw_path),
            "alpha_valid_layers": int(summary["alpha_n"]),
            "ERG_gap_valid_layers": int(summary["ERG_gap_n"]),
        }
        (spectral_root / f"status_step_{step:07d}.json").write_text(
            json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary
    except Exception as exc:
        status = {
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "completed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        spectral_root.mkdir(parents=True, exist_ok=True)
        (spectral_root / f"status_step_{step:07d}.json").write_text(
            json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
        )
        if bool(analysis_cfg.get("weightwatcher_strict", True)):
            raise
        print(
            f"[level0-spectral] WARNING step={step} {type(exc).__name__}: {exc}",
            flush=True,
        )
        return status
