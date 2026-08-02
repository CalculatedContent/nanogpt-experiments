from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT = Path("/tmp/nanogpt-experiment2")
DEFAULT_DATA_ROOT = Path("/tmp/nanogpt-level0-bpe/data")


def roots() -> dict[str, Path]:
    root = Path(os.getenv("NANOGPT_EXPERIMENT2_ROOT", DEFAULT_ROOT))
    return {
        "root": root,
        "data": Path(os.getenv("NANOGPT_EXPERIMENT2_DATA_ROOT", DEFAULT_DATA_ROOT)),
        "results": Path(os.getenv("NANOGPT_EXPERIMENT2_RESULTS_ROOT", root / "results")),
    }


def _env_map() -> dict[str, tuple[str, str, type]]:
    return {
        "NANOGPT_EXPERIMENT2_VOCAB_SIZE": ("model", "vocab_size", int),
        "NANOGPT_EXPERIMENT2_BLOCK_SIZE": ("model", "block_size", int),
        "NANOGPT_EXPERIMENT2_N_LAYER": ("model", "n_layer", int),
        "NANOGPT_EXPERIMENT2_N_HEAD": ("model", "n_head", int),
        "NANOGPT_EXPERIMENT2_N_EMBD": ("model", "n_embd", int),
        "NANOGPT_EXPERIMENT2_SEED": ("training", "seed", int),
        "NANOGPT_EXPERIMENT2_MAX_STEPS": ("training", "max_steps", int),
        "NANOGPT_EXPERIMENT2_BATCH_SIZE": ("training", "batch_size", int),
        "NANOGPT_EXPERIMENT2_GRAD_ACCUM_STEPS": (
            "training",
            "grad_accum_steps",
            int,
        ),
        "NANOGPT_EXPERIMENT2_LR": ("training", "learning_rate", float),
        "NANOGPT_EXPERIMENT2_WEIGHT_DECAY": ("training", "weight_decay", float),
        "NANOGPT_EXPERIMENT2_EVAL_INTERVAL": ("training", "eval_interval", int),
        "NANOGPT_EXPERIMENT2_EVAL_BATCHES": ("training", "eval_batches", int),
        "NANOGPT_EXPERIMENT2_CHECKPOINT_INTERVAL": (
            "training",
            "checkpoint_interval",
            int,
        ),
        "NANOGPT_EXPERIMENT2_WARMUP_STEPS": ("training", "warmup_steps", int),
        "NANOGPT_EXPERIMENT2_WW_INTERVAL": (
            "analysis",
            "weightwatcher_interval",
            int,
        ),
        "NANOGPT_EXPERIMENT2_START_STEP": ("controller", "start_step", int),
        "NANOGPT_EXPERIMENT2_CONTROL_INTERVAL": (
            "controller",
            "control_interval",
            int,
        ),
        "NANOGPT_EXPERIMENT2_PROJECTION_INTERVAL": (
            "controller",
            "projection_interval",
            int,
        ),
        "NANOGPT_EXPERIMENT2_MAX_ACTIVE_LAYERS": (
            "controller",
            "max_active_layers",
            int,
        ),
        "NANOGPT_EXPERIMENT2_PROBE_BATCHES": (
            "controller",
            "probe_batches",
            int,
        ),
        "NANOGPT_EXPERIMENT2_MAX_PROBE_LOSS_INCREASE": (
            "controller",
            "max_probe_loss_increase",
            float,
        ),
        "NANOGPT_EXPERIMENT2_TARGET_ALPHA": (
            "controller",
            "target_alpha",
            float,
        ),
        "NANOGPT_EXPERIMENT2_ENTER_TOLERANCE": (
            "controller",
            "enter_tolerance",
            float,
        ),
        "NANOGPT_EXPERIMENT2_EXIT_TOLERANCE": (
            "controller",
            "exit_tolerance",
            float,
        ),
        "NANOGPT_EXPERIMENT2_BLEND_ETA": ("wwpgd", "blend_eta", float),
        "NANOGPT_EXPERIMENT2_CAYLEY_ETA": ("wwpgd", "cayley_eta", float),
        "NANOGPT_EXPERIMENT2_MAX_RELATIVE_CHANGE": (
            "wwpgd",
            "max_relative_frobenius_change",
            float,
        ),
    }


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for name, (section, key, cast) in _env_map().items():
        if name in os.environ:
            cfg[section][key] = cast(os.environ[name])
    if "NANOGPT_EXPERIMENT2_TARGET_ALPHA" in os.environ:
        cfg["wwpgd"]["target_alpha"] = float(
            os.environ["NANOGPT_EXPERIMENT2_TARGET_ALPHA"]
        )
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ("experiment", "model", "training", "analysis", "wwpgd", "controller"):
        if section not in cfg:
            raise ValueError(f"missing config section: {section}")

    experiment = cfg["experiment"]
    model = cfg["model"]
    training = cfg["training"]
    analysis = cfg["analysis"]
    wwpgd = cfg["wwpgd"]
    controller = cfg["controller"]

    if experiment.get("name") != "experiment_2":
        raise ValueError("experiment.name must be experiment_2")
    if experiment.get("scale") not in {"level0", "level1"}:
        raise ValueError("experiment.scale must be level0 or level1")

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
    if int(training["warmup_steps"]) >= int(training["max_steps"]):
        raise ValueError("warmup_steps must be less than max_steps")
    if float(training["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= float(training["min_lr"]) <= float(training["learning_rate"]):
        raise ValueError("min_lr must be in [0, learning_rate]")

    if int(analysis["weightwatcher_interval"]) < 1:
        raise ValueError("analysis.weightwatcher_interval must be positive")
    if int(analysis["weightwatcher_interval"]) != int(controller["control_interval"]):
        raise ValueError("weightwatcher_interval must equal controller.control_interval")

    if not bool(wwpgd.get("enabled", False)):
        raise ValueError("wwpgd.enabled must be true")
    if wwpgd.get("apply_mode") != "event_projection":
        raise ValueError("wwpgd.apply_mode must be event_projection")
    if str(wwpgd.get("candidate_device", "")).lower() != "cpu":
        raise ValueError("wwpgd.candidate_device must be cpu")
    if float(wwpgd["target_alpha"]) <= 1.0:
        raise ValueError("wwpgd.target_alpha must be greater than one")
    for key in ("blend_eta", "cayley_eta"):
        if not 0.0 <= float(wwpgd[key]) <= 1.0:
            raise ValueError(f"wwpgd.{key} must be in [0, 1]")
    if int(wwpgd["min_tail"]) < 1:
        raise ValueError("wwpgd.min_tail must be positive")
    if int(wwpgd["max_consecutive_failures"]) < 1:
        raise ValueError("wwpgd.max_consecutive_failures must be positive")
    trust = wwpgd.get("max_relative_frobenius_change")
    if trust is not None and float(trust) <= 0:
        raise ValueError("max_relative_frobenius_change must be positive")

    for key in (
        "control_interval",
        "projection_interval",
        "max_active_layers",
        "start_step",
        "layer_harm_patience",
        "layer_cooldown_steps",
        "global_loss_gap_patience",
        "global_pause_steps",
        "probe_batches",
    ):
        if int(controller[key]) < 1:
            raise ValueError(f"controller.{key} must be positive")
    if int(controller["control_interval"]) % int(training["eval_interval"]) != 0:
        raise ValueError("controller.control_interval must be a multiple of eval_interval")
    if int(controller["control_interval"]) % int(controller["projection_interval"]) != 0:
        raise ValueError("control_interval must be a multiple of projection_interval")
    matrix_count = int(model["n_layer"]) * 6
    if int(controller["max_active_layers"]) > matrix_count:
        raise ValueError("max_active_layers exceeds projected matrix count")

    target = float(controller["target_alpha"])
    if not target > 1.0:
        raise ValueError("controller.target_alpha must be greater than one")
    if not abs(target - float(wwpgd["target_alpha"])) < 1e-12:
        raise ValueError("controller and WWPGD target_alpha must match")
    enter = float(controller["enter_tolerance"])
    exit_tolerance = float(controller["exit_tolerance"])
    if not (0 < exit_tolerance < enter):
        raise ValueError("require 0 < exit_tolerance < enter_tolerance")
    if not -1.0 <= float(controller["min_alignment_cosine"]) <= 1.0:
        raise ValueError("min_alignment_cosine must be in [-1, 1]")
    if float(controller["max_projection_to_adamw_ratio"]) <= 0:
        raise ValueError("max_projection_to_adamw_ratio must be positive")
    if float(controller["min_candidate_alpha_improvement"]) < 0:
        raise ValueError("min_candidate_alpha_improvement must be nonnegative")
    if float(controller["max_probe_loss_increase"]) < 0:
        raise ValueError("max_probe_loss_increase must be nonnegative")
    if not 0.0 <= float(controller["credit_ema_beta"]) < 1.0:
        raise ValueError("credit_ema_beta must be in [0, 1)")
    for key in (
        "layer_harm_margin",
        "global_loss_gap_margin",
        "recovery_gap_margin",
    ):
        if float(controller[key]) < 0:
            raise ValueError(f"controller.{key} must be nonnegative")
