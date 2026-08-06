from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT = Path("/tmp/nanogpt-level0-baselines")
SUPPORTED_OPTIMIZERS = ("sgd_momentum", "adamw", "muon")


def roots() -> dict[str, Path]:
    root = Path(os.getenv("NANOGPT_LEVEL0_ROOT", DEFAULT_ROOT))
    return {
        "root": root,
        "data": Path(
            os.getenv("NANOGPT_LEVEL0_DATA_ROOT", root / "data")
        ),
        "results": Path(
            os.getenv("NANOGPT_LEVEL0_RESULTS_ROOT", root / "results")
        ),
        "cache": Path(
            os.getenv("NANOGPT_LEVEL0_CACHE_ROOT", root / "cache")
        ),
    }


def _set_nested(
    cfg: dict[str, Any], path: tuple[str, ...], value: Any
) -> None:
    target = cfg
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("configuration root must be a mapping")

    env_map: dict[str, tuple[tuple[str, ...], Any]] = {
        "NANOGPT_LEVEL0_SEED": (("training", "seed"), int),
        "NANOGPT_LEVEL0_OPTIMIZER": (
            ("training", "optimizer"),
            str,
        ),
        "NANOGPT_LEVEL0_MAX_STEPS": (
            ("training", "max_steps"),
            int,
        ),
        "NANOGPT_LEVEL0_BATCH_SIZE": (
            ("training", "batch_size"),
            int,
        ),
        "NANOGPT_LEVEL0_GRAD_ACCUM_STEPS": (
            ("training", "grad_accum_steps"),
            int,
        ),
        "NANOGPT_LEVEL0_EVAL_INTERVAL": (
            ("training", "eval_interval"),
            int,
        ),
        "NANOGPT_LEVEL0_EVAL_BATCHES": (
            ("training", "eval_batches"),
            int,
        ),
        "NANOGPT_LEVEL0_CHECKPOINT_INTERVAL": (
            ("training", "checkpoint_interval"),
            int,
        ),
        "NANOGPT_LEVEL0_WW_INTERVAL": (
            ("analysis", "weightwatcher_interval"),
            int,
        ),
        "NANOGPT_LEVEL0_WEIGHTWATCHER": (
            ("analysis", "weightwatcher"),
            _parse_bool,
        ),
        "NANOGPT_LEVEL0_COMPILE": (
            ("training", "compile"),
            _parse_bool,
        ),
        "NANOGPT_LEVEL0_EPOCH_MONITORING": (
            ("epoch_monitoring", "enabled"),
            _parse_bool,
        ),
    }
    for name, (path_parts, cast) in env_map.items():
        if name in os.environ:
            _set_nested(cfg, path_parts, cast(os.environ[name]))

    validate_config(cfg)
    return cfg


def canonical_seeds(cfg: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(seed) for seed in cfg["training"]["seeds"])


def optimizer_profile(
    cfg: dict[str, Any], name: str | None = None
) -> dict[str, Any]:
    selected = str(name or cfg["training"]["optimizer"]).lower()
    if selected not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"unsupported optimizer {selected!r}; "
            f"choose from {SUPPORTED_OPTIMIZERS}"
        )
    profile = deepcopy(cfg["optimizer_profiles"][selected])
    profile["name"] = selected
    validate_optimizer_profile(profile)
    return profile


def warmup_steps_for(
    profile: dict[str, Any], max_steps: int
) -> int:
    if max_steps < 2:
        return 0
    fraction = float(profile["warmup_fraction"])
    return min(max_steps - 1, max(1, int(round(max_steps * fraction))))


def protocol_fingerprint(
    cfg: dict[str, Any],
    *,
    optimizer: str,
    seed: int,
    data_manifest: dict[str, Any],
) -> str:
    payload = {
        "protocol": cfg.get("protocol", {}),
        "model": cfg["model"],
        "training": cfg["training"],
        "epoch_monitoring": cfg.get("epoch_monitoring", {}),
        "optimizer_profile": optimizer_profile(cfg, optimizer),
        "analysis": cfg["analysis"],
        "optimizer": optimizer,
        "seed": int(seed),
        "data_manifest": data_manifest,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_optimizer_profile(profile: dict[str, Any]) -> None:
    family = str(profile.get("family", ""))
    if family not in {"sgd", "adamw", "muon"}:
        raise ValueError(f"unsupported optimizer family: {family!r}")
    warmup_fraction = float(profile.get("warmup_fraction", -1))
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError(
            "optimizer warmup_fraction must be in [0, 1)"
        )
    if profile.get("decay") != "cosine":
        raise ValueError(
            "only cosine decay is supported in the confirmatory suite"
        )

    if family in {"sgd", "adamw"}:
        peak = float(profile["learning_rate"])
        floor = float(profile["min_lr"])
        if peak <= 0 or floor < 0 or floor > peak:
            raise ValueError(
                "optimizer learning_rate/min_lr values are inconsistent"
            )
    if family == "muon":
        for peak_key, floor_key in (
            ("hidden_learning_rate", "hidden_min_lr"),
            ("auxiliary_learning_rate", "auxiliary_min_lr"),
        ):
            peak = float(profile[peak_key])
            floor = float(profile[floor_key])
            if peak <= 0 or floor < 0 or floor > peak:
                raise ValueError(
                    f"Muon {peak_key}/{floor_key} values "
                    "are inconsistent"
                )
        if int(profile["newton_schulz_steps"]) < 1:
            raise ValueError(
                "newton_schulz_steps must be positive"
            )


def validate_config(cfg: dict[str, Any]) -> None:
    for section in (
        "model",
        "training",
        "epoch_monitoring",
        "optimizer_profiles",
        "analysis",
        "runtime",
        "sampling",
    ):
        if section not in cfg:
            raise ValueError(
                f"missing configuration section: {section}"
            )

    model = cfg["model"]
    training = cfg["training"]
    monitoring = cfg["epoch_monitoring"]
    analysis = cfg["analysis"]
    for key in (
        "vocab_size",
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
    ):
        if int(model[key]) < 1:
            raise ValueError(f"model.{key} must be positive")
    if int(model["n_embd"]) % int(model["n_head"]) != 0:
        raise ValueError(
            "model.n_embd must be divisible by model.n_head"
        )

    for key in (
        "batch_size",
        "grad_accum_steps",
        "max_steps",
        "eval_interval",
        "eval_batches",
        "checkpoint_interval",
    ):
        if int(training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if float(training["grad_clip"]) < 0:
        raise ValueError(
            "training.grad_clip must be nonnegative"
        )
    if not training.get("seeds"):
        raise ValueError(
            "training.seeds must contain at least one seed"
        )
    if len(
        {int(seed) for seed in training["seeds"]}
    ) != len(training["seeds"]):
        raise ValueError("training.seeds must be unique")
    if (
        str(training["optimizer"]).lower()
        not in SUPPORTED_OPTIMIZERS
    ):
        raise ValueError("training.optimizer is unsupported")

    if float(monitoring["interval_epochs"]) <= 0:
        raise ValueError(
            "epoch_monitoring.interval_epochs must be positive"
        )
    if bool(monitoring.get("use_for_checkpoint_selection", False)):
        raise ValueError(
            "test epoch monitoring cannot be used for "
            "checkpoint selection"
        )

    profiles = cfg["optimizer_profiles"]
    for optimizer in SUPPORTED_OPTIMIZERS:
        if optimizer not in profiles:
            raise ValueError(
                f"missing optimizer profile: {optimizer}"
            )
        validate_optimizer_profile(
            {**profiles[optimizer], "name": optimizer}
        )

    if int(analysis["weightwatcher_interval"]) < 1:
        raise ValueError(
            "analysis.weightwatcher_interval must be positive"
        )
    if float(analysis["bollinger_sigma"]) <= 0:
        raise ValueError(
            "analysis.bollinger_sigma must be positive"
        )
