from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
import torch

from wwgpt.config import ModelConfig, TrainConfig, WWPGDConfig
from wwgpt.model import GPT
from wwgpt.optim import build_optimizer_bundle, optimizer_group_signature
from wwgpt.train import WWPGDExtension
from wwgpt.ww import (
    apply_external_wwpgd,
    external_projected_layer_names,
    projected_matrix_modules,
    resolved_external_wwpgd_config,
    weightwatcher_details,
)


@dataclass
class FakeExternalConfig:
    enable_tail_pgd: bool
    q: float
    blend_eta: float
    cayley_eta: float
    min_tail: int
    use_detx: bool
    warmup_epochs: int
    ramp_epochs: int
    verbose: bool


def install_fake_ww_pgd(monkeypatch, calls):
    mod = types.ModuleType("ww_pgd")
    mod.WWTailConfig = FakeExternalConfig

    def ww_pgd_project(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return [{"layer_name": name, "changed": True} for name in kwargs.get("layer_names", [])]

    mod.ww_pgd_project = ww_pgd_project
    monkeypatch.setitem(sys.modules, "ww_pgd", mod)
    return mod


def tiny_model():
    return GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=8, vocab_size=32))


def test_resolved_external_configuration_exact_values():
    cfg = resolved_external_wwpgd_config()
    assert cfg.enable_tail_pgd is True
    assert cfg.q == 1.0
    assert cfg.blend_eta == 0.5
    assert cfg.cayley_eta == 0.25
    assert cfg.min_tail == 5
    assert cfg.use_detx is True
    assert cfg.warmup_epochs == 0
    assert cfg.ramp_epochs == 0
    assert cfg.verbose is False


def test_no_strength_multiplier_external_blend_eta(monkeypatch):
    calls = []
    install_fake_ww_pgd(monkeypatch, calls)
    rows = apply_external_wwpgd(tiny_model(), actual_step=1)
    cfg = calls[0]["args"][1]
    assert cfg.blend_eta == 0.5
    assert all(row["blend_eta"] == 0.5 for row in rows)


def test_projection_interval_and_due_steps(monkeypatch):
    calls = []
    install_fake_ww_pgd(monkeypatch, calls)
    ext = WWPGDExtension(cfg=WWPGDConfig(), interval=3)
    m = tiny_model()
    monkeypatch.setattr("wwgpt.train.weightwatcher_details", lambda model: pd.DataFrame())
    assert ext.after_optimizer_step(model=m, optimizer_step=1, total_optimizer_steps=6, tokens_seen=8) == []
    assert ext.after_optimizer_step(model=m, optimizer_step=2, total_optimizer_steps=6, tokens_seen=16) == []
    assert len(calls) == 0
    for step in (3, 6):
        details, rows = ext.after_optimizer_step(model=m, optimizer_step=step, total_optimizer_steps=6, tokens_seen=step * 8)
        assert rows
    assert len(calls) == 2


def test_base_step_occurs_before_projection(monkeypatch):
    order = []
    install_fake_ww_pgd(monkeypatch, order)
    m = tiny_model()
    bundle, _ = build_optimizer_bundle(m, TrainConfig(layer_lr="flat"), "adamw")
    monkeypatch.setattr("wwgpt.train.weightwatcher_details", lambda model: pd.DataFrame())
    orig_step = bundle.step

    def step():
        order.append("base_step")
        orig_step()

    bundle.step = step
    ext = WWPGDExtension(cfg=WWPGDConfig(), interval=1)
    bundle.step()
    ext.after_optimizer_step(model=m, optimizer_step=1, total_optimizer_steps=1, tokens_seen=8)
    assert order[0] == "base_step"
    assert isinstance(order[1], dict)


def test_base_and_projected_arms_identical_before_first_projection(monkeypatch):
    calls = []
    install_fake_ww_pgd(monkeypatch, calls)
    torch.manual_seed(123)
    base = tiny_model()
    projected = tiny_model()
    projected.load_state_dict(base.state_dict())
    assert all(torch.equal(base.state_dict()[k], projected.state_dict()[k]) for k in base.state_dict())
    ext = WWPGDExtension(cfg=WWPGDConfig(), interval=10)
    monkeypatch.setattr("wwgpt.train.weightwatcher_details", lambda model: pd.DataFrame())
    assert ext.after_optimizer_step(model=projected, optimizer_step=9, total_optimizer_steps=20, tokens_seen=72) == []
    assert len(calls) == 0
    assert all(torch.equal(base.state_dict()[k], projected.state_dict()[k]) for k in base.state_dict())


def test_optimizer_group_and_weight_decay_signatures_identical():
    m1 = tiny_model(); m2 = tiny_model(); m2.load_state_dict(m1.state_dict())
    cfg = TrainConfig(layer_lr="manual")
    b1, _ = build_optimizer_bundle(m1, cfg, "adamw")
    b2, _ = build_optimizer_bundle(m2, cfg, "adamw")
    assert optimizer_group_signature(b1) == optimizer_group_signature(b2)


def test_external_layer_selector_resolves_exact_level0_matrices():
    names = external_projected_layer_names(tiny_model())
    assert names == [
        "blocks.0.attn.key",
        "blocks.0.attn.query",
        "blocks.0.attn.value",
        "blocks.0.attn.proj",
        "blocks.0.mlp.0",
        "blocks.0.mlp.2",
    ]


def test_external_layer_selector_excludes_embeddings_layernorm_head():
    names = external_projected_layer_names(tiny_model())
    assert "wte" not in names and "wpe" not in names and "lm_head" not in names
    assert not any("ln_" in n or n == "ln_f" for n in names)


def test_raw_weightwatcher_returns_separate_k_q_v(monkeypatch):
    captured = {}
    class FakeWatcher:
        def __init__(self, model):
            captured["model"] = model
        def analyze(self, **kwargs):
            captured["kwargs"] = kwargs
            return pd.DataFrame({"name": ["key", "query", "value"], "longname": ["blocks.0.attn.key", "blocks.0.attn.query", "blocks.0.attn.value"]})
    fake = types.ModuleType("weightwatcher")
    fake.WeightWatcher = FakeWatcher
    monkeypatch.setitem(sys.modules, "weightwatcher", fake)
    df = weightwatcher_details(tiny_model())
    assert captured["kwargs"] == {"detX": True, "randomize": False, "plot": False}
    assert {"blocks.0.attn.key", "blocks.0.attn.query", "blocks.0.attn.value"}.issubset(set(df["longname"]))


def test_src_wwgpt_has_no_wwpgd_svd_calls():
    text = "\n".join(p.read_text() for p in Path("src/wwgpt").glob("*.py"))
    assert "torch.linalg.svd(" not in text
    assert "torch.linalg.svdvals" not in text


def test_manifest_records_requested_and_resolved_external_config():
    from wwgpt.ww import external_wwpgd_manifest_fields

    cfg = WWPGDConfig(q=1.0, blend_eta=0.5, cayley_eta=0.25, min_tail=5, use_detx=True, warmup_events=0, ramp_events=0)
    fields = external_wwpgd_manifest_fields(True, cfg)

    assert fields["blend_eta"] == 0.5
    assert fields["q"] == 1.0
    assert fields["cayley_eta"] == 0.25
    assert fields["min_tail"] == 5
    assert fields["warmup"] == 0
    assert fields["ramp"] == 0
    assert fields["requested_external_wwpgd_config"]["blend_eta"] == 0.5
    assert fields["requested_external_wwpgd_config"]["ramp_events"] == 0
    assert fields["resolved_external_wwpgd_config"] == {
        "enable_tail_pgd": True,
        "q": 1.0,
        "blend_eta": 0.5,
        "cayley_eta": 0.25,
        "min_tail": 5,
        "use_detx": True,
        "warmup_epochs": 0,
        "ramp_epochs": 0,
        "verbose": False,
    }


def test_real_extension_passes_resolved_experiment_config_to_installed_package(monkeypatch):
    import ww_pgd

    captured = {}

    def capture_projector(model, cfg, **kwargs):
        captured["cfg"] = cfg
        captured["kwargs"] = kwargs
        return [{"layer_name": name, "changed": False} for name in external_projected_layer_names(model)]

    monkeypatch.setattr(ww_pgd, "ww_pgd_project", capture_projector)
    monkeypatch.setattr("wwgpt.train.weightwatcher_details", lambda model: pd.DataFrame())
    cfg = WWPGDConfig(q=1.0, blend_eta=0.5, cayley_eta=0.25, min_tail=5, use_detx=True, warmup_events=0, ramp_events=0)
    ext = WWPGDExtension(cfg=cfg, interval=1)
    details, rows = ext.after_optimizer_step(model=tiny_model(), optimizer_step=1, total_optimizer_steps=1, tokens_seen=8)

    external_cfg = captured["cfg"]
    assert external_cfg.__class__ is ww_pgd.WWTailConfig
    assert external_cfg.blend_eta == 0.5
    assert external_cfg.q == 1.0
    assert external_cfg.cayley_eta == 0.25
    assert external_cfg.min_tail == 5
    assert external_cfg.warmup_epochs == 0
    assert external_cfg.ramp_epochs == 0
    assert all(row["blend_eta"] == 0.5 and row["q"] == 1.0 and row["ramp"] == 0 for row in rows)
