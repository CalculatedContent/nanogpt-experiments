from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch

from .model import GPT, GPTConfig
from .optim import OptimizerHandle
from .runtime import evaluate_probe


def rng_state(train_generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if (
        torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
    ):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(
    state: dict[str, Any], train_generator: torch.Generator
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    train_generator.set_state(state["train_generator"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if (
        "torch_mps" in state
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["torch_mps"])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    *,
    model: GPT,
    handles: list[OptimizerHandle],
    step: int,
    cfg: dict[str, Any],
    optimizer_name: str,
    profile: dict[str, Any],
    best_validation_loss: float,
    best_validation_step: int,
    fingerprint: str,
    train_generator: torch.Generator,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "model": model.state_dict(),
        "optimizer_states": [
            {
                "role": handle.role,
                "state_dict": handle.optimizer.state_dict(),
                "peak_lr": handle.peak_lr,
                "min_lr": handle.min_lr,
            }
            for handle in handles
        ],
        "step": int(step),
        "config": cfg,
        "optimizer_name": optimizer_name,
        "optimizer_profile": profile,
        "best_validation_loss": float(best_validation_loss),
        "best_validation_step": int(best_validation_step),
        "protocol_fingerprint": fingerprint,
        "rng_state": rng_state(train_generator),
        "elapsed_seconds": float(elapsed_seconds),
    }


def save_checkpoint(path: Path, **kwargs: Any) -> None:
    atomic_torch_save(checkpoint_payload(**kwargs), path)


def prepare_metrics(
    path: Path, *, fields: Sequence[str], resume_step: int | None
) -> None:
    if resume_step is None or not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(fields)).writeheader()
        return
    import pandas as pd

    existing = pd.read_csv(path)
    if "step" not in existing.columns:
        raise RuntimeError("existing metrics.csv has no step column")
    existing = existing[existing["step"] < int(resume_step)]
    existing.to_csv(path, index=False)


def load_checkpoint_for_resume(
    path: Path,
    *,
    model: GPT,
    handles: list[OptimizerHandle],
    expected_fingerprint: str,
    train_generator: torch.Generator,
) -> tuple[int, float, int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            "resume checkpoint protocol fingerprint does not match this run"
        )
    model.load_state_dict(payload["model"])
    states = payload["optimizer_states"]
    if len(states) != len(handles):
        raise RuntimeError("resume checkpoint optimizer count mismatch")
    for handle, state in zip(handles, states, strict=True):
        if state["role"] != handle.role:
            raise RuntimeError("resume checkpoint optimizer role mismatch")
        handle.optimizer.load_state_dict(state["state_dict"])
    restore_rng_state(payload["rng_state"], train_generator)
    return (
        int(payload["step"]),
        float(payload["best_validation_loss"]),
        int(payload["best_validation_step"]),
        float(payload.get("elapsed_seconds", 0.0)),
    )


def test_metrics_for_checkpoint(
    checkpoint: Path,
    *,
    model_cfg: GPTConfig,
    test_probe: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, float, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    selected = GPT(model_cfg).to(device)
    selected.load_state_dict(payload["model"])
    metrics = evaluate_probe(selected, test_probe, device)
    return metrics[0], metrics[1], metrics[2], int(payload["step"])
