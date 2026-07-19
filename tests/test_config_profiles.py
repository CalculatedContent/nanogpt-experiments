from pathlib import Path


def test_default_yaml_loads_for_levels_0_through_4():
    from wwgpt.config import load_config

    path = Path("configs/default.yaml")
    for level in range(5):
        cfg = load_config(path, level=level)
        assert cfg.model.n_embd % cfg.model.n_head == 0


def test_level_zero_ladder_exact_scaling_profile():
    from wwgpt.config import level_model_config

    cfg = level_model_config(0)
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.block_size) == (1, 1, 64, 256)


def _historical_train_tuple(cfg):
    return (
        cfg.train.batch_size,
        cfg.train.gradient_accumulation,
        cfg.train.learning_rate,
        cfg.train.weight_decay,
        cfg.train.grad_clip,
        cfg.train.eval_batches,
        cfg.train.lr_schedule,
        cfg.train.layer_lr,
        cfg.train.warmup_ratio,
        cfg.train.min_lr_ratio,
    )


def test_reproduction_tiny_historical_hyperparameters():
    from wwgpt.config import load_config

    cfg = load_config(Path("configs/reproduction_tiny.yaml"), level=0)
    assert (cfg.model.n_layer, cfg.model.n_head, cfg.model.n_embd, cfg.model.block_size) == (1, 1, 64, 64)
    assert cfg.model.mlp_mult == 4
    assert cfg.model.tie_weights is False
    assert cfg.model.init_mode == "pytorch_default"
    assert cfg.model.layernorm_bias is True
    assert cfg.model.linear_bias is False
    assert cfg.train.max_steps == 100000
    assert cfg.train.eval_interval == 50
    assert _historical_train_tuple(cfg) == (16, 1, 0.001, 0.005, 0.0, 20, "warmup_cosine", "manual", 0.03, 0.25)


def test_reproduction_fineweb_historical_hyperparameters():
    from wwgpt.config import load_config

    cfg = load_config(Path("configs/reproduction_fineweb.yaml"), level=0)
    assert cfg.tokenizer == "gpt2"
    assert cfg.dataset_name == "HuggingFaceFW/fineweb-edu"
    assert cfg.dataset_config == "sample-10BT"
    assert (cfg.model.n_layer, cfg.model.n_head, cfg.model.n_embd, cfg.model.block_size) == (1, 1, 64, 64)
    assert cfg.model.mlp_mult == 4
    assert cfg.model.tie_weights is False
    assert cfg.model.init_mode == "pytorch_default"
    assert cfg.model.layernorm_bias is True
    assert cfg.model.linear_bias is False
    assert cfg.train.max_steps == 130000
    assert cfg.train.eval_interval == 1000
    assert _historical_train_tuple(cfg) == (16, 1, 0.001, 0.005, 0.0, 20, "warmup_cosine", "manual", 0.03, 0.25)


def test_resolved_wwpgd_defaults_are_exact():
    from wwgpt.config import load_config

    cfg = load_config(Path("configs/default.yaml"), level=0)
    assert cfg.wwpgd.q == 1.0
    assert cfg.wwpgd.blend_eta == 0.5
    assert cfg.wwpgd.cayley_eta == 0.25
    assert cfg.wwpgd.min_tail == 5
    assert cfg.wwpgd.warmup_events == 0
    assert cfg.wwpgd.ramp_events == 5
    assert cfg.wwpgd.use_detx is True


def test_required_profile_dependencies_import():
    import optimi
    import tiktoken
    import ww_pgd

    assert optimi is not None
    assert tiktoken is not None
    assert ww_pgd is not None


def test_composite_analysis_disabled_by_default():
    from wwgpt.config import ExperimentConfig, load_config

    assert ExperimentConfig().composite_spectral_analysis_enabled is False
    assert load_config(Path("configs/default.yaml"), level=0).composite_spectral_analysis_enabled is False
