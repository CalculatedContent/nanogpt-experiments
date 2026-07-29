from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Callable

import yaml

DEFAULT_ROOT = Path("/tmp/nanogpt-level0-gpt2")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


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


def _set_nested(cfg: dict[str, Any], section: str, key: str, value: Any) -> None:
    if section not in cfg or not isinstance(cfg[section], dict):
        raise ValueError(f"configuration is missing section {section!r}")
    cfg[section][key] = value


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")
    cfg: dict[str, Any] = copy.deepcopy(loaded)

    env_map: dict[str, tuple[str, str, Callable[[str], Any]]] = {
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
        "NANOGPT_LEVEL0_MIN_LR": ("training", "min_lr", float),
        "NANOGPT_LEVEL0_WARMUP_STEPS": ("training", "warmup_steps", int),
        "NANOGPT_LEVEL0_WEIGHT_DECAY": ("training", "weight_decay", float),
        "NANOGPT_LEVEL0_EVAL_INTERVAL": ("training", "eval_interval", int),
        "NANOGPT_LEVEL0_EVAL_BATCHES": ("training", "eval_batches", int),
        "NANOGPT_LEVEL0_EVAL_BATCH_SIZE": (
            "training",
            "eval_batch_size",
            int,
        ),
        "NANOGPT_LEVEL0_TEST_EVAL_BATCHES": (
            "training",
            "test_eval_batches",
            int,
        ),
        "NANOGPT_LEVEL0_CHECKPOINT_INTERVAL": (
            "training",
            "checkpoint_interval",
            int,
        ),
        "NANOGPT_LEVEL0_LAYER_LR_DECAY": (
            "training",
            "layer_lr_decay",
            float,
        ),
        "NANOGPT_LEVEL0_N_LAYER": ("model", "n_layer", int),
        "NANOGPT_LEVEL0_N_HEAD": ("model", "n_head", int),
        "NANOGPT_LEVEL0_N_EMBD": ("model", "n_embd", int),
        "NANOGPT_LEVEL0_BLOCK_SIZE": ("model", "block_size", int),
        "NANOGPT_LEVEL0_WEIGHTWATCHER": (
            "analysis",
            "weightwatcher",
            _parse_bool,
        ),
        "NANOGPT_LEVEL0_WW_INTERVAL": (
            "analysis",
            "weightwatcher_interval",
            int,
        ),
    }
    for name, (section, key, cast) in env_map.items():
        if name in os.environ:
            _set_nested(cfg, section, key, cast(os.environ[name]))

    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ("data", "model", "training", "analysis"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"configuration is missing section {section!r}")

    data = cfg["data"]
    model = cfg["model"]
    training = cfg["training"]
    analysis = cfg["analysis"]

    if data.get("tokenizer") != "gpt2":
        raise ValueError("the corrected Level 0 baseline requires tokenizer: gpt2")
    if data.get("dtype") != "uint16":
        raise ValueError("the corrected Level 0 baseline requires dtype: uint16")
    for key in ("train_tokens", "val_tokens", "test_tokens"):
        if int(data.get(key, 0)) <= 0:
            raise ValueError(f"data.{key} must be positive")

    for key in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
        if int(model.get(key, 0)) <= 0:
            raise ValueError(f"model.{key} must be positive")
    if int(model["n_embd"]) % int(model["n_head"]) != 0:
        raise ValueError("model.n_embd must be divisible by model.n_head")
    if int(model["vocab_size"]) < 50257:
        raise ValueError("model.vocab_size must cover the GPT-2 tokenizer")
    dropout = float(model.get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must satisfy 0 <= dropout < 1")

    positive_ints = (
        "batch_size",
        "grad_accum_steps",
        "max_steps",
        "eval_interval",
        "eval_batches",
        "eval_batch_size",
        "test_eval_batches",
        "checkpoint_interval",
    )
    for key in positive_ints:
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(training.get("warmup_steps", 0)) < 0:
        raise ValueError("training.warmup_steps must be nonnegative")
    if int(training["warmup_steps"]) >= int(training["max_steps"]):
        raise ValueError("training.warmup_steps must be smaller than max_steps")
    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if not 0 < float(training.get("min_lr", 0.0)) <= float(
        training["learning_rate"]
    ):
        raise ValueError("training.min_lr must be in (0, learning_rate]")
    if float(training.get("weight_decay", -1.0)) < 0:
        raise ValueError("training.weight_decay must be nonnegative")
    if float(training.get("epsilon", 0.0)) <= 0:
        raise ValueError("training.epsilon must be positive")
    if float(training.get("grad_clip", -1.0)) < 0:
        raise ValueError("training.grad_clip must be nonnegative")
    layer_lr_decay = float(training.get("layer_lr_decay", 1.0))
    if not 0 < layer_lr_decay <= 1:
        raise ValueError("training.layer_lr_decay must be in (0, 1]")
    if str(training.get("optimizer", "")).lower() not in {"adamw", "muon"}:
        raise ValueError("training.optimizer must be adamw or muon")

    interval = int(analysis.get("weightwatcher_interval", 0))
    if bool(analysis.get("weightwatcher", False)) and interval <= 0:
        raise ValueError("analysis.weightwatcher_interval must be positive")
