from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .controller import LayerwiseController
from .model import GPT


METRIC_FIELDS = [
    "step",
    "tokens_seen",
    "elapsed_sec",
    "learning_rate",
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
    "baseline_train_loss",
    "baseline_val_loss",
    "loss_gap_vs_baseline",
    "grad_norm",
    "weight_norm",
    "active_layer_count",
    "global_pause_until",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BaselineReference:
    def __init__(self, run: Path, adaptive_cfg: dict[str, Any], seed: int):
        manifest_path = run / "manifest.json"
        metrics_path = run / "metrics.csv"
        if not manifest_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"baseline reference is incomplete: {run}")
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("seed", -1)) != seed:
            raise RuntimeError("baseline reference seed does not match adaptive seed")
        baseline_cfg = manifest["config"]
        for section in ("model", "training", "analysis"):
            if baseline_cfg[section] != adaptive_cfg[section]:
                raise RuntimeError(
                    f"baseline/adaptive protocol mismatch in {section}"
                )
        frame = pd.read_csv(metrics_path)
        frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
        for column in ("train_loss", "val_loss", "learning_rate"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        self.by_step = {
            int(row.step): row
            for row in frame.dropna(subset=["step"]).itertuples(index=False)
        }
        self.run = run
        self.metrics_path = metrics_path
        self.manifest = manifest

    def at(self, step: int):
        row = self.by_step.get(step)
        if row is None:
            raise RuntimeError(
                f"matching baseline metrics are missing at evaluation step {step}"
            )
        return row


def save_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict[str, Any],
    best_validation_loss: float,
    best_validation_step: int,
    controller: LayerwiseController | None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "config": cfg,
            "best_validation_loss": float(best_validation_loss),
            "best_validation_step": int(best_validation_step),
            "controller": controller.snapshot() if controller is not None else None,
        },
        path,
    )
