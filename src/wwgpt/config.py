from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SEEDS = [1337, 2027, 4099, 7919, 104729]
TOKEN_MULTIPLIERS = [5, 10, 20, 40]


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 2
    n_head: int = 1
    n_embd: int = 64
    block_size: int = 256
    vocab_size: int = 8192
    dropout: float = 0.0
    bias: bool = True
    activation: str = "gelu"
    tie_weights: bool = True


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    gradient_accumulation: int = 1
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    warmup_steps: int = 10
    max_steps: int | None = None
    grad_clip: float = 1.0
    eval_interval: int = 10
    checkpoint_interval: int = 50
    spectral_interval: int = 10
    eval_batches: int = 4


@dataclass(frozen=True)
class WWPGDConfig:
    enabled: bool = False
    target_alpha: float = 2.0
    strength: float = 0.02
    projection_schedule: list[float] = field(default_factory=lambda: [0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.82, 0.92])
    warmup_steps: int = 0
    ramp_steps: int = 10
    layer_scope: str = "blocks"
    include_embeddings: bool = False
    include_output: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wwpgd: WWPGDConfig = field(default_factory=WWPGDConfig)
    seeds: list[int] = field(default_factory=lambda: DEFAULT_SEEDS.copy())
    token_multipliers: list[int] = field(default_factory=lambda: TOKEN_MULTIPLIERS.copy())
    parameter_count_convention: str = "total"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_revision: str = "main"


def ladder() -> dict[int, ModelConfig]:
    return {
        0: ModelConfig(n_layer=2, n_head=1, n_embd=64),
        1: ModelConfig(n_layer=4, n_head=2, n_embd=128),
        2: ModelConfig(n_layer=6, n_head=3, n_embd=192),
        3: ModelConfig(n_layer=8, n_head=4, n_embd=256),
        4: ModelConfig(n_layer=10, n_head=5, n_embd=320),
    }


def load_config(path: Path | None = None, level: int = 0) -> ExperimentConfig:
    cfg = ExperimentConfig(model=ladder()[level])
    if path is None:
        return cfg
    data = yaml.safe_load(path.read_text())
    model = ModelConfig(**{**asdict(cfg.model), **data.get("model", {})})
    train = TrainConfig(**{**asdict(cfg.train), **data.get("train", {})})
    wwpgd = WWPGDConfig(**{**asdict(cfg.wwpgd), **data.get("wwpgd", {})})
    rest: dict[str, Any] = {k: v for k, v in data.items() if k not in {"model", "train", "wwpgd"}}
    return ExperimentConfig(model=model, train=train, wwpgd=wwpgd, **rest)
