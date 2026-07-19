from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from wwgpt.config import ExperimentConfig, ModelConfig, TrainConfig, WWPGDConfig, load_config
from wwgpt.data import TokenData, prepare_fineweb_gpt2_reproduction, prepare_scientific_data, prepare_tiny_shakespeare_char_reproduction
from wwgpt.model import GPT
from wwgpt.optim import optimizer_group_signature, build_optimizer_bundle
from wwgpt.train import run_scientific_single
from wwgpt.utils import sha256_bytes


def _fake_tiktoken(monkeypatch):
    mod = types.ModuleType("tiktoken")
    class Encoding:
        n_vocab = 50257
        def encode_ordinary(self, text):
            return [(i % 255) for i, _ in enumerate(text, start=1)]
    mod.get_encoding = lambda name: Encoding()
    monkeypatch.setitem(sys.modules, "tiktoken", mod)


def _fake_ww_pgd(monkeypatch, calls):
    mod = types.ModuleType("ww_pgd")
    class WWTailConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    def ww_pgd_project(model, cfg, *, layer_names=None, **kwargs):
        calls.append({"cfg": cfg, "layer_names": tuple(layer_names or ()), "kwargs": kwargs})
        return [{"layer_name": n, "changed": True} for n in (layer_names or [])]
    mod.WWTailConfig = WWTailConfig
    mod.ww_pgd_project = ww_pgd_project
    monkeypatch.setitem(sys.modules, "ww_pgd", mod)


def test_three_data_modes_and_custom_bpe_keeps_tokenizer_training_docs(tmp_path, monkeypatch):
    _fake_tiktoken(monkeypatch)
    tiny = prepare_tiny_shakespeare_char_reproduction(tmp_path, text="abcdefghij" * 10)
    assert tiny.data_manifest["data_mode"] == "tiny_shakespeare_char_reproduction"
    assert len(tiny.train) == 90 and len(tiny.val) == 10
    assert tiny.tokenizer_manifest["tokenizer_type"] == "character"

    cfg = replace(load_config(Path("configs/reproduction_fineweb.yaml"), level=0), seeds=[123])
    docs = [f"document {i} with deterministic split content" for i in range(100)]
    gpt2 = prepare_fineweb_gpt2_reproduction(tmp_path, cfg, docs=docs)
    assert gpt2.data_manifest["data_mode"] == "fineweb_gpt2_reproduction"
    assert gpt2.data_manifest["dataset_revision"] == "593b3a867298afb8ce42625a270ef20ddcad28f9"
    assert 50256 in gpt2.train + gpt2.val
    assert gpt2.tokenizer_manifest["tokenizer_type"] == "tiktoken_gpt2"

    scale_cfg = ExperimentConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=4, vocab_size=64),
        train=TrainConfig(batch_size=1, max_steps=None, eval_batches=1, eval_interval=1),
    )
    cfg_path = tmp_path / "scale.yaml"
    cfg_path.write_text("model:\n  n_layer: 1\n  n_head: 1\n  n_embd: 64\n  block_size: 4\n  vocab_size: 64\ntrain:\n  batch_size: 1\n  eval_batches: 1\n")
    train_docs = [(f"training document number {i} has unique content {i} " * 80) for i in range(180)]
    custom = prepare_scientific_data(tmp_path, 0, 1, config_path=cfg_path, docs=train_docs, min_validation_tokens=1)
    assert custom.data_manifest["data_mode"] == "fineweb_custom_bpe_scaling"
    assert custom.tokenizer_manifest["not_reproduction_of_uploaded_fineweb_experiment"] is True
    assert custom.data_manifest["train_document_count"] >= 128
    assert len(custom.train) >= custom.data_manifest["realized_tokens"] + 1


def _fixture_data() -> TokenData:
    train = list(range(16)) * 32
    val = list(reversed(range(16))) * 16
    manifest = {"realized_tokens": 8, "data_mode": "offline_token_fixture", "dataset_name": "offline_token_fixture", "dataset_config": "fixture", "dataset_revision": "fixture", "validation_document_count": 1}
    tok = {"tokenizer_hash": sha256_bytes(b"offline-token-fixture"), "tokenizer_type": "offline", "vocab_size": 16}
    return TokenData(train, val, 16, sha256_bytes(bytes(train + val)), None, manifest, tok)


def _spectral_kqv(model, **kwargs):
    return [{"longname": n, "name": n.rsplit(".", 1)[-1], **kwargs} for n in ["blocks.0.attn.key", "blocks.0.attn.query", "blocks.0.attn.value"]]


@pytest.mark.parametrize("base", ["adamw", "muon", "stableadamw"])
def test_six_arm_pair_plumbing_offline_token_fixture(tmp_path, monkeypatch, base):
    calls = []
    _fake_ww_pgd(monkeypatch, calls)
    monkeypatch.setattr("wwgpt.train.weightwatcher_details", lambda model: pd.DataFrame())
    monkeypatch.setattr("wwgpt.train.spectral_summary", _spectral_kqv)
    cfg = ExperimentConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=4, vocab_size=16),
        train=TrainConfig(batch_size=1, max_steps=2, eval_interval=1, eval_batches=1, wwpgd_interval=1, layer_lr="flat", lr_schedule="constant"),
        wwpgd=WWPGDConfig(extension="none"),
        base_optimizer=base,
        extensions=["none", "wwpgd"],
    )
    torch.manual_seed(7)
    init_model = GPT(cfg.model)
    init_state = init_model.state_dict()
    init_hash = sha256_bytes(b"".join(init_state[k].cpu().numpy().tobytes() for k in sorted(init_state)))
    data = _fixture_data()
    base_dir = run_scientific_single(tmp_path, base, 7, cfg, data, f"pair-{base}", init_state, init_hash, 0, 1, device="cpu")
    ww_dir = run_scientific_single(tmp_path, f"{base}_wwpgd", 7, cfg, data, f"pair-{base}", init_state, init_hash, 0, 1, device="cpu")

    bm = json.loads((base_dir / "manifest.json").read_text())
    wm = json.loads((ww_dir / "manifest.json").read_text())
    assert bm["initialization_hash"] == wm["initialization_hash"]
    assert bm["training_probe_hash"] == wm["training_probe_hash"]
    assert bm["validation_probe_hash"] == wm["validation_probe_hash"]
    assert bm["target_train_tokens"] == wm["target_train_tokens"]
    assert bm["base_optimizer"] == wm["base_optimizer"] == base
    assert bm["extension"] == "none" and wm["extension"] == "wwpgd"

    sig_base = optimizer_group_signature(build_optimizer_bundle(GPT(cfg.model), cfg.train, base)[0])
    sig_ww = optimizer_group_signature(build_optimizer_bundle(GPT(cfg.model), cfg.train, base)[0])
    assert sig_base == sig_ww
    assert tuple(x["weight_decay"] for x in sig_base) == tuple(x["weight_decay"] for x in sig_ww)
    assert len(calls) == 2
    assert [c["kwargs"].get("global_step") or c["kwargs"].get("actual_step") for c in calls] == [1, 2]

    metrics = pd.read_csv(ww_dir / "metrics.csv")
    assert {"train_loss", "val_loss", "train_perplexity", "val_perplexity", "train_top1_accuracy", "val_top1_accuracy"}.issubset(metrics.columns)
    raw = pd.read_csv(ww_dir / "spectral.csv")
    assert {"blocks.0.attn.key", "blocks.0.attn.query", "blocks.0.attn.value"}.issubset(set(raw["longname"]))
    assert not (base_dir / "composite_spectral.csv").exists()
    assert not (ww_dir / "composite_spectral.csv").exists()


def test_real_external_adamw_wwpgd_tiny_model_if_installed():
    pytest.importorskip("ww_pgd")
    from wwgpt.ww import apply_external_wwpgd
    model = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=4, vocab_size=16))
    rows = apply_external_wwpgd(model, actual_step=1, actual_tokens_seen=4)
    assert isinstance(rows, list)
