from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SEEDS = [1337, 2027, 4099, 7919, 104729]
TOKEN_MULTIPLIERS = [20, 40, 80, 160]
SCIENTIFIC_SCHEMA_VERSION = 3
MODEL_ARCHITECTURE_VERSION = "separate_qkv_bias_free_untied_head_v1"
VALID_BASE_OPTIMIZERS = {"adamw", "muon", "stableadamw"}
VALID_EXTENSIONS = {"none", "wwpgd"}



@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 64
    block_size: int = 256
    vocab_size: int = 8192
    dropout: float = 0.0
    bias: bool = False
    linear_bias: bool = False
    layernorm_bias: bool = True
    init_mode: str = "nanogpt_normal_0p02"
    profile_name: str = "scaling_level0"
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
    lr_decay_steps: int | None = None
    max_steps: int | None = None
    grad_clip: float = 0.0
    eval_interval: int = 10
    checkpoint_interval: int = 50
    spectral_interval: int = 10
    eval_batches: int = 20
    lr_schedule: str = "warmup_cosine"
    warmup_ratio: float = 0.01
    min_lr_ratio: float = 0.10
    layer_lr: str = "flat"
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
    training_sampling: str = "random_window"
    wwpgd_interval: int | None = None

    def __post_init__(self) -> None:
        if self.lr_schedule not in {"constant", "warmup_cosine", "warmup_linear"}:
            raise ValueError(f"unknown lr_schedule {self.lr_schedule}")
        if self.layer_lr not in {"flat", "llrd", "manual"}:
            raise ValueError(f"unknown layer_lr {self.layer_lr}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must satisfy 0.0 <= warmup_ratio < 1.0")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must satisfy 0.0 <= min_lr_ratio <= 1.0")
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0 when supplied")
        if self.lr_decay_steps is not None and self.lr_decay_steps < 1:
            raise ValueError("lr_decay_steps must be >= 1 when supplied")


@dataclass(frozen=True)
class WWPGDConfig:
    enabled: bool = False
    extension: str = "none"
    target_alpha: float = 2.0
    # Deprecated compatibility field for loading old artifacts; new scientific runs use q/eta scheduling.
    strength: float = 1.0
    q: float = 1.0
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
    ramp_events: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    scientific_schema_version: int = SCIENTIFIC_SCHEMA_VERSION
    model_architecture_version: str = MODEL_ARCHITECTURE_VERSION
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
    data_mode: str = "fineweb_custom_bpe_scaling"
    tokenizer: str | None = None
    composite_spectral_analysis_enabled: bool = False


def historical_level0_model_config() -> ModelConfig:
    return ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=64, dropout=0.0, mlp_mult=4, init_mode="pytorch_default", profile_name="historical_reproduction_level0")


def scaling_level0_model_config() -> ModelConfig:
    return ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=256, dropout=0.0, mlp_mult=4, init_mode="nanogpt_normal_0p02", profile_name="scaling_level0")


def level_model_config(level: int) -> ModelConfig:
    if level == 0:
        return scaling_level0_model_config()
    return ModelConfig(n_layer=2 * level, n_head=level + 1, n_embd=64 * (level + 1), block_size=256)


def ladder() -> dict[int, ModelConfig]:
    return {i: level_model_config(i) for i in range(5)}


def validate_model_config(cfg: ModelConfig) -> None:
    if cfg.n_layer < 1:
        raise ValueError("model.n_layer must be >= 1")
    if cfg.n_head < 1:
        raise ValueError("model.n_head must be >= 1")
    if cfg.n_embd < 1:
        raise ValueError("model.n_embd must be >= 1")
    if cfg.block_size < 1:
        raise ValueError("model.block_size must be >= 1")
    if cfg.vocab_size < 1:
        raise ValueError("model.vocab_size must be >= 1")
    if cfg.mlp_mult < 1:
        raise ValueError("model.mlp_mult must be >= 1")
    if not 0.0 <= cfg.dropout <= 1.0:
        raise ValueError("model.dropout must satisfy 0.0 <= dropout <= 1.0")
    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if cfg.n_embd // cfg.n_head != 64:
        raise ValueError("schema-v3 requires attention head dimension 64")


def validate_train_config(cfg: TrainConfig) -> None:
    if cfg.batch_size < 1:
        raise ValueError("train.batch_size must be >= 1")
    if cfg.gradient_accumulation < 1:
        raise ValueError("train.gradient_accumulation must be >= 1")
    if cfg.learning_rate <= 0.0:
        raise ValueError("train.learning_rate must be > 0")
    if cfg.weight_decay < 0.0:
        raise ValueError("train.weight_decay must be >= 0")
    if cfg.wwpgd_interval is not None and cfg.wwpgd_interval < 1:
        raise ValueError("train.wwpgd_interval must be >= 1 when supplied")
    if cfg.lr_schedule not in {"constant", "warmup_cosine", "warmup_linear"}:  # stlr is intentionally retired.
        raise ValueError(f"unknown lr_schedule {cfg.lr_schedule}")
    if cfg.layer_lr not in {"flat", "llrd", "manual"}:
        raise ValueError(f"unknown layer_lr {cfg.layer_lr}")
    if not 0.0 <= cfg.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must satisfy 0.0 <= warmup_ratio < 1.0")
    if not 0.0 <= cfg.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must satisfy 0.0 <= min_lr_ratio <= 1.0")
    if cfg.warmup_steps is not None and cfg.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0 when supplied")
    if cfg.lr_decay_steps is not None and cfg.lr_decay_steps < 1:
        raise ValueError("lr_decay_steps must be >= 1 when supplied")
    if cfg.lr_schedule == "warmup_cosine" and cfg.warmup_steps is not None and cfg.lr_decay_steps is not None and cfg.lr_decay_steps <= cfg.warmup_steps:
        raise ValueError("train.lr_decay_steps must be greater than train.warmup_steps when warmup_cosine is enabled")


def validate_wwpgd_config(cfg: WWPGDConfig) -> None:
    if not 0.0 <= cfg.blend_eta <= 1.0:
        raise ValueError("wwpgd.blend_eta must satisfy 0.0 <= blend_eta <= 1.0")
    if cfg.warmup_steps < 0:
        raise ValueError("wwpgd.warmup_steps must be >= 0")
    if cfg.ramp_steps < 0:
        raise ValueError("wwpgd.ramp_steps must be >= 0")
    if cfg.warmup_events < 0:
        raise ValueError("wwpgd.warmup_events must be >= 0")
    if cfg.ramp_events < 0:
        raise ValueError("wwpgd.ramp_events must be >= 0")
    if cfg.min_tail < 1:
        raise ValueError("wwpgd.min_tail must be >= 1")
    if cfg.extension not in VALID_EXTENSIONS:
        raise ValueError(f"unknown wwpgd.extension {cfg.extension}")


def validate_experiment_config(cfg: ExperimentConfig) -> None:
    validate_model_config(cfg.model)
    validate_train_config(cfg.train)
    validate_wwpgd_config(cfg.wwpgd)
    if len(cfg.seeds) < 1:
        raise ValueError("seeds must contain at least one seed")
    if cfg.base_optimizer not in VALID_BASE_OPTIMIZERS:
        raise ValueError(f"unknown base_optimizer {cfg.base_optimizer}")
    invalid_extensions = [ext for ext in cfg.extensions if ext not in VALID_EXTENSIONS]
    if invalid_extensions:
        raise ValueError(f"unknown extension(s): {', '.join(invalid_extensions)}")


def _reject_unknown_keys(section: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        prefix = f"{section}." if section else ""
        keys = ", ".join(f"{prefix}{key}" for key in unknown)
        raise ValueError(f"unknown configuration key(s): {keys}")


def load_config(path: Path | None = None, level: int = 0) -> ExperimentConfig:
    cfg = ExperimentConfig(model=level_model_config(level))
    if path is None:
        validate_experiment_config(cfg)
        return cfg
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    model_keys = set(ModelConfig.__dataclass_fields__)
    train_keys = set(TrainConfig.__dataclass_fields__)
    wwpgd_keys = set(WWPGDConfig.__dataclass_fields__)
    experiment_keys = set(ExperimentConfig.__dataclass_fields__) - {"model", "train", "wwpgd"}
    _reject_unknown_keys("", data, experiment_keys | {"model", "train", "wwpgd"})
    for section in ("model", "train", "wwpgd"):
        if section in data and not isinstance(data[section], dict):
            raise ValueError(f"configuration section {section} must be a mapping")
    _reject_unknown_keys("model", data.get("model", {}), model_keys)
    _reject_unknown_keys("train", data.get("train", {}), train_keys)
    _reject_unknown_keys("wwpgd", data.get("wwpgd", {}), wwpgd_keys)
    model = ModelConfig(**{**asdict(cfg.model), **data.get("model", {})})
    train = TrainConfig(**{**asdict(cfg.train), **data.get("train", {})})
    wwpgd = WWPGDConfig(**{**asdict(cfg.wwpgd), **data.get("wwpgd", {})})
    rest: dict[str, Any] = {k: v for k, v in data.items() if k in experiment_keys}
    loaded = ExperimentConfig(model=model, train=train, wwpgd=wwpgd, **rest)
    validate_experiment_config(loaded)
    return loaded
