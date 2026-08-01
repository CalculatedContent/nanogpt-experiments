from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT = Path("/tmp/nanogpt-level0-wwpgd")
DEFAULT_BASELINE_ROOT = Path("/tmp/nanogpt-level0-bpe")


def roots() -> dict[str, Path]:
    root = Path(os.getenv("NANOGPT_LEVEL0_WWPGD_ROOT", DEFAULT_ROOT))
    baseline_root = Path(os.getenv("NANOGPT_LEVEL0_ROOT", DEFAULT_BASELINE_ROOT))
    shared_data = Path(
        os.getenv(
            "NANOGPT_LEVEL0_WWPGD_DATA_ROOT",
            os.getenv("NANOGPT_LEVEL0_DATA_ROOT", baseline_root / "data"),
        )
    )
    shared_cache = Path(
        os.getenv(
            "NANOGPT_LEVEL0_WWPGD_CACHE_ROOT",
            os.getenv("NANOGPT_LEVEL0_CACHE_ROOT", baseline_root / "cache"),
        )
    )
    return {
        "root": root,
        "data": shared_data,
        "results": Path(
            os.getenv("NANOGPT_LEVEL0_WWPGD_RESULTS_ROOT", root / "results")
        ),
        "cache": shared_cache,
    }


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    env_map = {
        "NANOGPT_LEVEL0_WWPGD_SEED": ("training", "seed", int),
        "NANOGPT_LEVEL0_WWPGD_MAX_STEPS": ("training", "max_steps", int),
        "NANOGPT_LEVEL0_WWPGD_BATCH_SIZE": ("training", "batch_size", int),
        "NANOGPT_LEVEL0_WWPGD_GRAD_ACCUM_STEPS": (
            "training",
            "grad_accum_steps",
            int,
        ),
        "NANOGPT_LEVEL0_WWPGD_LR": ("training", "learning_rate", float),
        "NANOGPT_LEVEL0_WWPGD_WEIGHT_DECAY": (
            "training",
            "weight_decay",
            float,
        ),
        "NANOGPT_LEVEL0_WWPGD_EVAL_INTERVAL": (
            "training",
            "eval_interval",
            int,
        ),
        "NANOGPT_LEVEL0_WWPGD_WW_INTERVAL": (
            "analysis",
            "weightwatcher_interval",
            int,
        ),
        "NANOGPT_LEVEL0_WWPGD_PROJECTION_INTERVAL": (
            "wwpgd",
            "interval",
            int,
        ),
        "NANOGPT_LEVEL0_WWPGD_TARGET_ALPHA": (
            "wwpgd",
            "target_alpha",
            float,
        ),
        "NANOGPT_LEVEL0_WWPGD_BLEND_ETA": (
            "wwpgd",
            "blend_eta",
            float,
        ),
        "NANOGPT_LEVEL0_WWPGD_CAYLEY_ETA": (
            "wwpgd",
            "cayley_eta",
            float,
        ),
        "NANOGPT_LEVEL0_WWPGD_MAX_RELATIVE_CHANGE": (
            "wwpgd",
            "max_relative_frobenius_change",
            float,
        ),
        "NANOGPT_LEVEL0_WWPGD_MAX_CONSECUTIVE_FAILURES": (
            "wwpgd",
            "max_consecutive_failures",
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
    wwpgd = cfg["wwpgd"]

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
    if str(training["optimizer"]).lower() != "adamw":
        raise ValueError("the isolated WWPGD experiment requires AdamW")
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

    if not bool(wwpgd.get("enabled", False)):
        raise ValueError("wwpgd.enabled must be true")
    if wwpgd.get("extension") != "wwpgd":
        raise ValueError("wwpgd.extension must be wwpgd")
    if wwpgd.get("apply_mode") != "event_projection":
        raise ValueError("wwpgd.apply_mode must be event_projection")
    if int(wwpgd["interval"]) < 1:
        raise ValueError("wwpgd.interval must be positive")
    if not float(wwpgd["target_alpha"]) > 1.0:
        raise ValueError("wwpgd.target_alpha must be greater than one")
    for key in ("blend_eta", "cayley_eta"):
        if not 0.0 <= float(wwpgd[key]) <= 1.0:
            raise ValueError(f"wwpgd.{key} must be in [0, 1]")
    if int(wwpgd["min_tail"]) < 1:
        raise ValueError("wwpgd.min_tail must be positive")
    if str(wwpgd["candidate_device"]).lower() != "cpu":
        raise ValueError(
            "this isolated MPS-safe experiment requires candidate_device=cpu"
        )
    limit = wwpgd.get("max_relative_frobenius_change")
    if limit is not None and float(limit) <= 0:
        raise ValueError("wwpgd.max_relative_frobenius_change must be positive")
    if int(wwpgd.get("log_interval", 1)) < 1:
        raise ValueError("wwpgd.log_interval must be positive")
    if int(wwpgd.get("max_consecutive_failures", 5)) < 1:
        raise ValueError("wwpgd.max_consecutive_failures must be positive")
