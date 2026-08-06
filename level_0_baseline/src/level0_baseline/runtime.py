from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from .model import GPTConfig


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_runtime(device: torch.device, cfg: dict[str, Any]) -> None:
    precision = str(cfg["training"].get("matmul_precision", "high"))
    torch.set_float32_matmul_precision(precision)
    if bool(cfg["training"].get("deterministic_algorithms", False)):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "mps" and bool(cfg["runtime"].get("mps_fallback", True)):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if (
        torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
    ):
        torch.mps.manual_seed(seed)


def random_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("data split is too short for the configured block size")
    starts = torch.randint(
        len(data) - block_size - 1,
        (batch_size,),
        generator=generator,
    ).tolist()
    x = torch.stack(
        [
            torch.from_numpy(
                np.asarray(data[start : start + block_size], dtype=np.int64)
            )
            for start in starts
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    data[start + 1 : start + 1 + block_size], dtype=np.int64
                )
            )
            for start in starts
        ]
    )
    return x, y


def fixed_probe(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    n_batches: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        random_batch(data, batch_size, block_size, generator)
        for _ in range(n_batches)
    ]


@torch.inference_mode()
def evaluate_probe(
    model: nn.Module,
    probe: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    for x_cpu, y_cpu in probe:
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        logits, loss = model(x, y)
        if loss is None:
            raise RuntimeError("evaluation forward pass did not return a loss")
        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(-1) == y).sum().detach().cpu())
        total += y.numel()
    model.train(was_training)
    mean_loss = float(np.mean(losses))
    return mean_loss, math.exp(min(20.0, mean_loss)), correct / max(total, 1)


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms), ord=2)


def model_weight_norm(model: nn.Module) -> float:
    squared = sum(
        float((parameter.detach().float() ** 2).sum().cpu())
        for parameter in model.parameters()
    )
    return math.sqrt(squared)


def parameter_snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [
        parameter.detach().float().cpu().clone() for parameter in model.parameters()
    ]


def update_norm_between(
    previous: list[torch.Tensor] | None, current: list[torch.Tensor]
) -> float:
    if previous is None:
        return 0.0
    if len(previous) != len(current):
        raise RuntimeError("parameter snapshot length changed")
    squared = 0.0
    for old, new in zip(previous, current, strict=True):
        squared += float(((new - old) ** 2).sum())
    return math.sqrt(squared)


def mps_memory_megabytes(device: torch.device) -> tuple[float, float]:
    if device.type != "mps" or not hasattr(torch, "mps"):
        return float("nan"), float("nan")
    current = (
        float(torch.mps.current_allocated_memory()) / (1024**2)
        if hasattr(torch.mps, "current_allocated_memory")
        else float("nan")
    )
    driver = (
        float(torch.mps.driver_allocated_memory()) / (1024**2)
        if hasattr(torch.mps, "driver_allocated_memory")
        else float("nan")
    )
    return current, driver


def load_data(data_root: Path, model_cfg: GPTConfig):
    metadata_path = data_root / "meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"missing {metadata_path}; run level0-prepare-data first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("tokenizer") != "gpt2":
        raise RuntimeError("expected GPT-2 BPE Level Zero data")
    if int(metadata.get("vocab_size", -1)) != model_cfg.vocab_size:
        raise RuntimeError(
            "data/model vocabulary mismatch: "
            f"data={metadata.get('vocab_size')} model={model_cfg.vocab_size}"
        )
    dtype = np.dtype(str(metadata.get("dtype", "uint16")))
    arrays: dict[str, np.memmap] = {}
    for split in ("train", "val", "test"):
        path = data_root / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(f"missing prepared split: {path}")
        arrays[split] = np.memmap(path, dtype=dtype, mode="r")
        if len(arrays[split]) <= model_cfg.block_size + 1:
            raise RuntimeError(f"{split} split is too short")
    return metadata, arrays


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    return f"{minutes:d}m{secs:02d}s"
