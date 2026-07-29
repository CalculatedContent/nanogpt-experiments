from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT = Path("/tmp/nanogpt-level0-bpe")


def roots() -> dict[str, Path]:
    root = Path(os.getenv("NANOGPT_LEVEL0_ROOT", DEFAULT_ROOT))
    return {
        "root": root,
        "data": Path(os.getenv("NANOGPT_LEVEL0_DATA_ROOT", root / "data")),
        "results": Path(
            os.getenv("NANOGPT_LEVEL0_RESULTS_ROOT", root / "results")
        ),
        "cache": Path(os.getenv("NANOGPT_LEVEL0_CACHE_ROOT", root / "cache")),
    }


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    env_map = {
        "NANOGPT_LEVEL0_SEED": ("training", "seed", int),
        "NANOGPT_LEVEL0_OPTIMIZER": ("training", "optimizer", str),
        "NANOGPT_LEVEL0_MAX_STEPS": ("training", "max_steps", int),
        "NANOGPT_LEVEL0_BATCH_SIZE": ("training", "batch_size", int),
        "NANOGPT_LEVEL0_GRAD_ACCUM_STEPS": (
            "training",
            "grad_accum_steps",
            int,
        ),
        "NANOGPT_LEVEL0_LR": ("training", "learning_rate", float),
        "NANOGPT_LEVEL0_WEIGHT_DECAY": (
            "training",
            "weight_decay",
            float,
        ),
        "NANOGPT_LEVEL0_EVAL_INTERVAL": (
            "training",
            "eval_interval",
            int,
        ),
        "NANOGPT_LEVEL0_WW_INTERVAL": (
            "analysis",
            "weightwatcher_interval",
            int,
        ),
    }
    for name, (section, key, cast) in env_map.items():
        if name in os.environ:
            cfg[section][key] = cast(os.environ[name])
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    model = cfg["model"]
    training = cfg["training"]
    analysis = cfg["analysis"]
    for key in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
        if int(model[key]) < 1:
            raise ValueError(f"model.{key} must be positive")
    if int(model["n_embd"]) % int(model["n_head"]) != 0:
        raise ValueError("model.n_embd must be divisible by model.n_head")
    for key in (
        "batch_size",
        "grad_accum_steps",
        "max_steps",
        "eval_interval",
        "eval_batches",
        "checkpoint_interval",
        "warmup_steps",
    ):
        if int(training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if float(training["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if float(training["min_lr"]) < 0:
        raise ValueError("training.min_lr must be nonnegative")
    if float(training["min_lr"]) > float(training["learning_rate"]):
        raise ValueError("training.min_lr cannot exceed learning_rate")
    if int(training["warmup_steps"]) >= int(training["max_steps"]):
        raise ValueError("warmup_steps must be less than max_steps")
    if int(analysis["weightwatcher_interval"]) < 1:
        raise ValueError("analysis.weightwatcher_interval must be positive")
