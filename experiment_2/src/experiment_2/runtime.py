from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .model import GPT, GPTConfig, projected_modules


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def learning_rate_at(update_index: int, training: dict[str, Any]) -> float:
    warmup_steps = int(training["warmup_steps"])
    max_steps = int(training["max_steps"])
    peak = float(training["learning_rate"])
    minimum = float(training["min_lr"])
    if update_index < warmup_steps:
        return peak * (update_index + 1) / max(1, warmup_steps)
    progress = (update_index - warmup_steps) / max(
        1, max_steps - warmup_steps - 1
    )
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + cosine * (peak - minimum)


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
                np.asarray(data[start + 1 : start + 1 + block_size], dtype=np.int64)
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
    model: GPT,
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
        assert loss is not None
        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(-1) == y).sum().detach().cpu())
        total += y.numel()
    model.train(was_training)
    mean_loss = float(np.mean(losses))
    return mean_loss, math.exp(min(20.0, mean_loss)), correct / max(total, 1)


def gradient_norm(parameters) -> torch.Tensor:
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms), ord=2)


def model_weight_norm(model: nn.Module) -> float:
    return math.sqrt(
        sum(
            float((parameter.detach().float() ** 2).sum())
            for parameter in model.parameters()
        )
    )


def load_data(data_root: Path, model_cfg: GPTConfig):
    metadata_path = data_root / "meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing {metadata_path}; prepare the BPE corpus first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("tokenizer") != "gpt2":
        raise RuntimeError("Experiment 2 requires GPT-2 BPE data")
    if int(metadata.get("vocab_size", -1)) != model_cfg.vocab_size:
        raise RuntimeError(
            "data/model vocabulary mismatch: "
            f"data={metadata.get('vocab_size')} model={model_cfg.vocab_size}"
        )
    dtype = np.dtype(str(metadata.get("dtype", "uint16")))
    arrays = {}
    for split in ("train", "val", "test"):
        path = data_root / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(f"missing prepared split: {path}")
        arrays[split] = np.memmap(path, dtype=dtype, mode="r")
        if len(arrays[split]) <= model_cfg.block_size + 1:
            raise RuntimeError(f"{split} split is too short")
    return metadata, arrays


def snapshot_projected_weights(model: GPT) -> dict[str, torch.Tensor]:
    return {
        name: module.weight.detach().clone()
        for name, _, _, module in projected_modules(model)
    }
