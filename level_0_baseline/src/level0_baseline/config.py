from __future__ import annotations

import copy
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
        "results": Path(os.getenv("NANOGPT_LEVEL0_RESULTS_ROOT", root / "results")),
        "cache": Path(os.getenv("NANOGPT_LEVEL0_CACHE_ROOT", root / "cache")),
    }


_ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
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
    "NANOGPT_LEVEL0_EVAL_INTERVAL": ("training", "eval_interval", int),
    "NANOGPT_LEVEL0_LOG_INTERVAL": ("training", "log_interval", int),
    "NANOGPT_LEVEL0_WEIGHTWATCHER_INTERVAL": (
        "analysis",
        "weightwatcher_interval",
        int,
    ),
}


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def validate_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a mapping")
    for section in ("model", "training", "analysis"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"missing configuration section: {section}")

    model = cfg["model"]
    training = cfg["training"]
    analysis = cfg["analysis"]

    for key in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
        model[key] = _positive_int(model[key], f"model.{key}")
    if model["vocab_size"] > 65_535:
        raise ValueError("model.vocab_size must fit in uint16 token files")
    if model["n_embd"] % model["n_head"] != 0:
        raise ValueError("model.n_embd must be divisible by model.n_head")
    dropout = float(model.get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must satisfy 0 <= dropout < 1")
    model["dropout"] = dropout
    model["bias"] = bool(model.get("bias", False))

    for key in (
        "batch_size",
        "grad_accum_steps",
        "max_steps",
        "eval_interval",
        "eval_batches",
        "log_interval",
        "checkpoint_interval",
        "warmup_steps",
    ):
        training[key] = _positive_int(training[key], f"training.{key}")
    if training["warmup_steps"] >= training["max_steps"]:
        raise ValueError("training.warmup_steps must be smaller than max_steps")

    for key in ("learning_rate", "min_lr", "weight_decay", "grad_clip"):
        training[key] = float(training[key])
    if training["learning_rate"] <= 0:
        raise ValueError("training.learning_rate must be positive")
    if not 0 <= training["min_lr"] <= training["learning_rate"]:
        raise ValueError("training.min_lr must be between 0 and learning_rate")
    if training["weight_decay"] < 0:
        raise ValueError("training.weight_decay must be nonnegative")
    if training["grad_clip"] < 0:
        raise ValueError("training.grad_clip must be nonnegative")

    training["beta1"] = float(training["beta1"])
    training["beta2"] = float(training["beta2"])
    training["epsilon"] = float(training.get("epsilon", 1e-8))
    if training["epsilon"] <= 0:
        raise ValueError("training.epsilon must be positive")
    if not 0 <= training["beta1"] < 1 or not 0 <= training["beta2"] < 1:
        raise ValueError("training betas must be in [0, 1)")
    training["seed"] = int(training["seed"])
    training["optimizer"] = str(training["optimizer"]).lower()
    if training["optimizer"] not in {"adamw", "muon"}:
        raise ValueError("training.optimizer must be adamw or muon")
    training["compile"] = bool(training.get("compile", False))

    analysis["weightwatcher"] = bool(analysis.get("weightwatcher", True))
    analysis["weightwatcher_interval"] = _positive_int(
        analysis["weightwatcher_interval"],
        "analysis.weightwatcher_interval",
    )
    # Alpha tracking must be deterministic. Randomized ESDs are a separate trap
    # diagnostic and are deliberately not used in this isolated baseline.
    analysis["randomize"] = False


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    cfg = copy.deepcopy(loaded)
    for name, (section, key, cast) in _ENV_OVERRIDES.items():
        if name in os.environ:
            cfg[section][key] = cast(os.environ[name])
    validate_config(cfg)
    return cfg
