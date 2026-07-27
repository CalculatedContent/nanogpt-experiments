from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from wwgpt.config import ModelConfig, TrainConfig, load_config
from wwgpt.model import GPT
from wwgpt.optim import (
    build_optimizer_bundle,
    learning_rate_scale_factor,
    resolve_learning_rates,
)
from wwgpt.ww import (
    ExternalWWTailConfigSpec,
    build_stock_wwpgd_candidate,
    resolve_candidate_execution_device,
)


def test_standard_lr_scale_rules() -> None:
    cfg = TrainConfig(batch_size=16, gradient_accumulation=1, lr_reference_tokens_per_step=4096)
    assert learning_rate_scale_factor(cfg, 256) == pytest.approx(1.0)
    linear = replace(cfg, batch_size=32, lr_scale_rule="linear_batch")
    sqrt = replace(cfg, batch_size=32, lr_scale_rule="sqrt_batch")
    assert learning_rate_scale_factor(linear, 256) == pytest.approx(2.0)
    assert learning_rate_scale_factor(sqrt, 256) == pytest.approx(2.0**0.5)


def test_optimizer_groups_record_resolved_lr_scaling() -> None:
    model = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=32, vocab_size=128))
    cfg = TrainConfig(
        batch_size=8, gradient_accumulation=2, learning_rate=1e-3,
        lr_scale_rule="linear_batch", lr_reference_tokens_per_step=256,
    )
    bundle, _ = build_optimizer_bundle(model, cfg, "adamw")
    resolution = resolve_learning_rates(cfg, 32)
    assert resolution["lr_scale_factor"] == pytest.approx(2.0)
    assert bundle.learning_rate_resolution == resolution
    assert all(group["effective_base_lr"] == pytest.approx(2e-3) for group in bundle.optimizers[0].param_groups)


def test_unknown_matrix_lr_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown matrix_lr_multipliers"):
        TrainConfig(matrix_lr_multipliers={"typo_role": 1.0})


def test_auto_candidate_device_offloads_mps_and_xla(monkeypatch) -> None:
    model = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=16, vocab_size=64))
    monkeypatch.setattr("wwgpt.ww._model_device", lambda _model: torch.device("mps"))
    assert resolve_candidate_execution_device(model, "auto") == "cpu"
    monkeypatch.setattr("wwgpt.ww._model_device", lambda _model: torch.device("cuda"))
    assert resolve_candidate_execution_device(model, "auto") == "live"


def test_cpu_candidate_offload_does_not_mutate_live_model(monkeypatch) -> None:
    import wwgpt.ww as ww

    model = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=16, vocab_size=64))
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    class DummyConfig:
        pass

    monkeypatch.setattr(ww, "_external_config_object", lambda *_args, **_kwargs: DummyConfig())
    monkeypatch.setattr(ww, "_assert_stock_wwpgd_api", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ww, "_external_wwpgd_module",
        lambda: type("Module", (), {"ww_pgd_project": object()})(),
    )

    def fake_run(execution_model, _config, **_kwargs):
        rows = []
        with torch.no_grad():
            for name, weight in ww.projected_matrix_modules(execution_model):
                weight.add_(0.001)
                rows.append({"longname": name, "alpha": 2.5, "D": 0.05, "xmin": 0.1, "detX_num": 8, "num_evals": weight.shape[0]})
        return {"ww_logs": [pd.DataFrame(rows)], "diagnostic_logs": [], "native_internal_diagnostics": False}

    monkeypatch.setattr(ww, "run_pip_wwpgd_candidate", fake_run)
    candidate = build_stock_wwpgd_candidate(
        model, cfg=ExternalWWTailConfigSpec(candidate_device="cpu")
    )
    assert candidate.candidate_offloaded is True
    assert candidate.candidate_execution_device == "cpu"
    assert any(candidate.stock_candidate_changed.values())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_level_configs_declare_safe_local_defaults() -> None:
    for level in (0, 1, 2):
        cfg = load_config(Path(f"configs/level{level}_adaptive_alpha.yaml"), level)
        assert cfg.train.lr_scale_rule == "fixed"
        assert cfg.train.lr_reference_tokens_per_step == 4096
        assert cfg.wwpgd.candidate_device == "auto"
