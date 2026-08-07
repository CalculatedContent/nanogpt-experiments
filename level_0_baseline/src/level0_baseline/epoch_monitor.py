from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from .model import GPT


def epoch_monitor_step_map(
    *,
    train_tokens: int,
    tokens_per_step: int,
    max_steps: int,
    target_epochs: float,
    interval_epochs: float = 1.0,
    include_final: bool = True,
) -> dict[int, float]:
    """Map optimizer steps to preregistered epoch-monitoring points.

    The nominal epoch grid is defined before training. Each epoch is mapped to
    the nearest optimizer step, with the final step included explicitly.
    """
    if train_tokens < 1:
        raise ValueError("train_tokens must be positive")
    if tokens_per_step < 1:
        raise ValueError("tokens_per_step must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if target_epochs <= 0:
        raise ValueError("target_epochs must be positive")
    if interval_epochs <= 0:
        raise ValueError("interval_epochs must be positive")

    nominal_epochs: list[float] = []
    count = int(math.floor(target_epochs / interval_epochs + 1e-12))
    for index in range(1, count + 1):
        nominal_epochs.append(index * interval_epochs)
    if include_final and (
        not nominal_epochs
        or not math.isclose(
            nominal_epochs[-1], target_epochs, rel_tol=0.0, abs_tol=1e-9
        )
    ):
        nominal_epochs.append(float(target_epochs))

    step_to_epoch: dict[int, float] = {}
    for nominal_epoch in nominal_epochs:
        step = int(round(nominal_epoch * train_tokens / tokens_per_step))
        step = min(max_steps, max(1, step))
        # If two nominal points map to the same step at tiny test scales, retain
        # the later point because it is closer to the completed training state.
        step_to_epoch[step] = float(nominal_epoch)
    if include_final:
        step_to_epoch[max_steps] = float(target_epochs)
    return dict(sorted(step_to_epoch.items()))


def epoch_checkpoint_path(
    run_dir: str | Path, *, nominal_epoch: float, step: int
) -> Path:
    run_dir = Path(run_dir)
    epoch_text = f"{nominal_epoch:06.3f}".replace(".", "p")
    return (
        run_dir
        / "epoch_checkpoints"
        / f"checkpoint_epoch_{epoch_text}_step_{int(step):07d}.pt"
    )


def save_epoch_model_checkpoint(
    run_dir: str | Path,
    *,
    model: GPT,
    step: int,
    nominal_epoch: float,
    actual_epoch: float,
    cfg: dict[str, Any],
    optimizer_name: str,
    seed: int,
    protocol_fingerprint: str,
) -> Path:
    """Write a compact model-only checkpoint for post-run epoch auditing."""
    path = epoch_checkpoint_path(
        run_dir, nominal_epoch=nominal_epoch, step=step
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model": model.state_dict(),
        "step": int(step),
        "nominal_epoch": float(nominal_epoch),
        "actual_epoch": float(actual_epoch),
        "config": cfg,
        "optimizer_name": str(optimizer_name),
        "seed": int(seed),
        "protocol_fingerprint": str(protocol_fingerprint),
        "purpose": "preregistered_epoch_monitoring_model_only",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path
