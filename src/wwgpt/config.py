from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SEEDS = [1337, 2027, 4099, 7919, 104729]
TOKEN_MULTIPLIERS = [20, 40, 80, 160]
SCIENTIFIC_SCHEMA_VERSION = 3
MODEL_ARCHITECTURE_VERSION = "separate_qkv_bias_free_untied_head_v1"


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 2
    n_head: int = 1
    n_embd: int = 64
    block_size: int = 256
    vocab_size: int = 8192
    dropout: float = 0.0
    bias: bool = False
    activation: str = "gelu"
    tie_weights: bool = False
    mlp_mult: int = 4
    model_architecture_version: str = MODEL_ARCHITECTURE_VERSION


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    gradient_accumulation: int = 1
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    warmup_steps: int | None = None
    max_steps: int | None = None
    grad_clip: float = 0.0
    eval_interval: int = 10
    checkpoint_interval: int = 50
    spectral_interval: int = 10
    eval_batches: int = 20
    lr_schedule: str = "warmup_cosine"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.10
    layer_lr: str = "llrd"
    llrd_gamma: float | None = None
    llrd_min_multiplier: float = 0.50
    matrix_lr_multipliers: dict[str, float] = field(default_factory=dict)
    muon_learning_rate: float = 2e-2
    muon_momentum: float = 0.95
    newton_schulz_steps: int = 5
    stable_learning_rate: float = 3e-4
    stable_betas: tuple[float, float] = (0.9, 0.99)
    stable_epsilon: float = 1e-6
    stable_triton: bool = False
    max_train_tokens: int | None = None
    evaluation_sampling: str = "random_per_eval"
    wwpgd_interval: int | None = None


@dataclass(frozen=True)
class WWPGDConfig:
    enabled: bool = False
    extension: str = "none"
    target_alpha: float = 2.0
    strength: float = 0.02
    projection_schedule: list[float] = field(default_factory=lambda: [0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.82, 0.92])
    warmup_steps: int = 0
    ramp_steps: int = 10
    layer_scope: str = "blocks"
    include_embeddings: bool = False
    include_output: bool = False
    project_embeddings: bool = False
    project_output_head: bool = False
    min_tail: int = 5
    blend_eta: float = 0.5
    cayley_eta: float = 0.25
    use_detx: bool = True
    warmup_events: int = 0
    ramp_events: int = 5


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wwpgd: WWPGDConfig = field(default_factory=WWPGDConfig)
    seeds: list[int] = field(default_factory=lambda: DEFAULT_SEEDS.copy())
    token_multipliers: list[int] = field(default_factory=lambda: TOKEN_MULTIPLIERS.copy())
    parameter_count_convention: str = "total"
    base_optimizer: str = "adamw"
    extensions: list[str] = field(default_factory=lambda: ["none", "wwpgd"])
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_revision: str = "main"


def level_model_config(level: int) -> ModelConfig:
    if level == 0:
        return ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=256, dropout=0.0, mlp_mult=4)
    return ModelConfig(n_layer=2 * level, n_head=level + 1, n_embd=64 * (level + 1), block_size=256)


def ladder() -> dict[int, ModelConfig]:
    return {i: level_model_config(i) for i in range(5)}


def validate_model_config(cfg: ModelConfig) -> None:
    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if cfg.n_embd // cfg.n_head != 64:
        raise ValueError("schema-v3 requires attention head dimension 64")


def load_config(path: Path | None = None, level: int = 0) -> ExperimentConfig:
    cfg = ExperimentConfig(model=level_model_config(level))
    if path is None:
        return cfg
    data = yaml.safe_load(path.read_text())
    model = ModelConfig(**{**asdict(cfg.model), **data.get("model", {})})
    train = TrainConfig(**{**asdict(cfg.train), **data.get("train", {})})
    wwpgd = WWPGDConfig(**{**asdict(cfg.wwpgd), **data.get("wwpgd", {})})
    rest: dict[str, Any] = {k: v for k, v in data.items() if k not in {"model", "train", "wwpgd"}}
    return ExperimentConfig(model=model, train=train, wwpgd=wwpgd, **rest)
