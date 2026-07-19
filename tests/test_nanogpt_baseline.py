import json
import math
from pathlib import Path

import pytest
import torch

from wwgpt.config import ExperimentConfig, ModelConfig, TrainConfig, load_config
from wwgpt.model import GPT
from wwgpt.optim import build_optimizer_bundle, nanogpt_cosine_lr
from wwgpt.train import resolved_baseline_hyperparameters, run_single


def test_baseline_defaults_match_standard_nanogpt_profile():
    cfg = TrainConfig()
    assert cfg.learning_rate == pytest.approx(6e-4)
    assert cfg.weight_decay == pytest.approx(0.1)
    assert cfg.grad_clip == pytest.approx(1.0)
    assert cfg.betas == pytest.approx((0.9, 0.95))
    assert cfg.lr_schedule == "warmup_cosine"
    assert cfg.layer_lr == "flat"
    assert cfg.matrix_lr_multipliers == {}
    model_cfg = ModelConfig()
    assert model_cfg.tie_weights is True
    assert model_cfg.dropout == pytest.approx(0.0)

    yaml_cfg = load_config(Path("configs/default.yaml"), level=0)
    assert yaml_cfg.train.learning_rate == pytest.approx(6e-4)
    assert yaml_cfg.train.weight_decay == pytest.approx(0.1)
    assert yaml_cfg.train.grad_clip == pytest.approx(1.0)
    assert yaml_cfg.model.tie_weights is True


def test_tied_weights_share_storage_and_optimizer_deduplicates_parameter():
    model = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=8, vocab_size=32, tie_weights=True))
    assert model.lm_head.weight is model.wte.weight
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()
    named = dict(model.named_parameters())
    assert "wte.weight" in named
    assert "lm_head.weight" not in named
    signature = [row["parameter_name"] for row in __import__("wwgpt.optim", fromlist=["optimizer_group_signature"]).optimizer_group_signature(build_optimizer_bundle(model, TrainConfig(), "adamw")[0])]
    assert signature.count("wte.weight") == 1
    assert "lm_head.weight" not in signature


def test_residual_projection_initialization_uses_nanogpt_scaled_std():
    torch.manual_seed(123)
    cfg = ModelConfig(n_layer=4, n_head=1, n_embd=64, block_size=8, vocab_size=32)
    model = GPT(cfg)
    expected = 0.02 / math.sqrt(2 * cfg.n_layer)
    assert model.blocks[0].attn.proj.weight.std().item() == pytest.approx(expected, rel=0.25)
    assert model.blocks[0].mlp[2].weight.std().item() == pytest.approx(expected, rel=0.25)
    assert model.blocks[0].attn.query.weight.std().item() == pytest.approx(0.02, rel=0.25)


def test_causal_attention_prevents_future_token_information_flow():
    torch.manual_seed(7)
    model = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=4, vocab_size=16, dropout=0.0))
    model.eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 9, 10]])
    with torch.no_grad():
        logits_a, _ = model(prefix)
        logits_b, _ = model(changed_future)
    assert torch.allclose(logits_a[:, :2], logits_b[:, :2], atol=1e-6)
    assert not torch.allclose(logits_a[:, 2:], logits_b[:, 2:])


def test_attention_keeps_separate_qkv_matrices_and_standard_dropout_modules():
    model = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=8, vocab_size=32, dropout=0.2))
    attn = model.blocks[0].attn
    assert attn.query is not attn.key and attn.query is not attn.value and attn.key is not attn.value
    assert attn.query.weight.shape == attn.key.weight.shape == attn.value.weight.shape == (64, 64)
    assert attn.attn_dropout.p == pytest.approx(0.2)
    assert attn.resid_dropout.p == pytest.approx(0.2)
    assert model.drop.p == pytest.approx(0.2)


def test_resolved_baseline_hyperparameters_are_recorded_in_smoke_metadata(tmp_path):
    cfg = ExperimentConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=8, vocab_size=16),
        train=TrainConfig(batch_size=2, max_steps=1, eval_interval=1),
    )
    run = run_single(tmp_path, "adamw", 123, cfg, list(range(128)), list(range(128)), "pair")
    manifest = json.loads((run / "manifest.json").read_text())
    expected = resolved_baseline_hyperparameters(cfg)
    assert manifest["resolved_baseline_hyperparameters"] == json.loads(json.dumps(expected))
    assert manifest["resolved_baseline_hyperparameters"]["learning_rate"] == pytest.approx(6e-4)


def test_warmup_then_cosine_decay_has_no_hard_coded_step_values():
    cfg = TrainConfig(warmup_steps=2, lr_decay_steps=10, min_lr_ratio=0.1)
    assert nanogpt_cosine_lr(0, peak_lr=cfg.learning_rate, warmup_steps=2, lr_decay_steps=10, min_lr_ratio=0.1) == pytest.approx(cfg.learning_rate / 3)
    assert nanogpt_cosine_lr(2, peak_lr=cfg.learning_rate, warmup_steps=2, lr_decay_steps=10, min_lr_ratio=0.1) == pytest.approx(cfg.learning_rate)
    assert nanogpt_cosine_lr(9, peak_lr=cfg.learning_rate, warmup_steps=2, lr_decay_steps=10, min_lr_ratio=0.1) == pytest.approx(cfg.learning_rate * 0.1)
