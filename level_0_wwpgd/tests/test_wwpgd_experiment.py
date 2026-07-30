from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "level_0_baseline" / "src"))
sys.path.insert(0, str(REPO / "level_0_wwpgd" / "src"))

from level0_wwpgd.config import load_config  # noqa: E402
from level0_wwpgd.model import GPT, GPTConfig  # noqa: E402
from level0_wwpgd.train import main as train_main  # noqa: E402
from level0_wwpgd.wwpgd_extension import WWPGDExtension, projected_modules  # noqa: E402


def test_protocol_matches_baseline_except_wwpgd_section():
    baseline = yaml.safe_load((REPO / "level_0_baseline/configs/level0.yaml").read_text())
    intervention = yaml.safe_load((REPO / "level_0_wwpgd/configs/level0.yaml").read_text())
    assert intervention["model"] == baseline["model"]
    assert intervention["training"] == baseline["training"]
    assert intervention["analysis"] == baseline["analysis"]
    assert intervention["wwpgd"]["apply_mode"] == "event_projection"
    assert intervention["wwpgd"]["interval"] == 1


def test_config_rejects_non_adamw_and_non_event_projection(tmp_path):
    cfg = yaml.safe_load((REPO / "level_0_wwpgd/configs/level0.yaml").read_text())
    cfg["training"]["optimizer"] = "muon"
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="requires AdamW"):
        load_config(path)
    cfg["training"]["optimizer"] = "adamw"
    cfg["wwpgd"]["apply_mode"] = "cached_endpoint_relaxation"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="event_projection"):
        load_config(path)


def test_projection_inventory_is_transformer_only():
    model = GPT(GPTConfig(vocab_size=64, block_size=8, n_layer=2, n_head=1, n_embd=64))
    items = projected_modules(model)
    assert len(items) == 12
    names = {name for name, _, _, _ in items}
    assert not any("embedding" in name or "lm_head" in name or "ln" in name for name in names)


def test_two_step_training_calls_extension_once_per_update(tmp_path, monkeypatch):
    data_root = tmp_path / "data"; results_root = tmp_path / "results"; data_root.mkdir()
    rng = np.random.default_rng(7)
    for split, size in (("train", 2000), ("val", 800), ("test", 800)):
        rng.integers(0, 64, size=size, dtype=np.uint16).tofile(data_root / f"{split}.bin")
    (data_root / "meta.json").write_text(json.dumps({"tokenizer":"gpt2","vocab_size":64,"dtype":"uint16","splits":{"train":2000,"val":800,"test":800}}))
    cfg = yaml.safe_load((REPO / "level_0_wwpgd/configs/level0.yaml").read_text())
    cfg["model"].update(vocab_size=64, block_size=8, n_layer=1, n_head=1, n_embd=16)
    cfg["training"].update(batch_size=2, grad_accum_steps=1, max_steps=2, eval_interval=1, eval_batches=1, checkpoint_interval=1, warmup_steps=1, seed=13)
    cfg["analysis"]["weightwatcher"] = False
    config_path = tmp_path / "config.yaml"; config_path.write_text(yaml.safe_dump(cfg))

    class FakeExtension:
        instances = []
        def __init__(self, model, config):
            self.model=model; self.config=config; self.call_count=0; self.projected_matrix_count=0; self.runtime_seconds=0.0; self.__class__.instances.append(self)
        def manifest_fields(self): return {"wwpgd_implementation":"fake-test"}
        def after_optimizer_step(self, *, optimizer_step, total_optimizer_steps, tokens_seen):
            self.call_count += 1; self.projected_matrix_count += 1
            return [{"optimizer_step":optimizer_step,"tokens_seen":tokens_seen,"projection_event":self.call_count-1,"layer_name":"blocks.0.attn.q_proj","matrix_type":"W_Q","block":0,"target_alpha":2.0,"changed":True,"relative_frobenius_change_applied":1e-4}]

    import level0_wwpgd.train as train_module
    monkeypatch.setattr(train_module, "WWPGDExtension", FakeExtension)
    monkeypatch.setattr(sys, "argv", ["level0-wwpgd-train","--config",str(config_path),"--data-root",str(data_root),"--results-root",str(results_root),"--device","cpu"])
    train_main()
    run = results_root / "adamw_wwpgd_seed_13"
    completion = json.loads((run / "run_complete.json").read_text())
    projection = pd.read_csv(run / "wwpgd_projection.csv")
    assert completion["completed"] is True
    assert completion["wwpgd_call_count"] == 2
    assert len(projection) == 2
    assert int(projection.optimizer_step.max()) == 2


@pytest.mark.integration
def test_real_stock_wwpgd_one_event_changes_only_block_matrices():
    pytest.importorskip("ww_pgd")
    model = GPT(GPTConfig(vocab_size=256, block_size=8, n_layer=1, n_head=1, n_embd=64))
    protected_before = {
        "token_embedding": model.token_embedding.weight.detach().clone(),
        "position_embedding": model.position_embedding.weight.detach().clone(),
        "ln_f": model.ln_f.weight.detach().clone(),
    }
    config = yaml.safe_load((REPO / "level_0_wwpgd/configs/level0.yaml").read_text())["wwpgd"]
    extension = WWPGDExtension(model, config)
    rows = extension.after_optimizer_step(optimizer_step=1, total_optimizer_steps=2, tokens_seen=512)
    assert len(rows) == 6
    assert extension.call_count == 1
    assert {row["candidate_epoch"] for row in rows} == {0}
    assert {row["candidate_num_epochs"] for row in rows} == {2}
    assert {row["candidate_analyzed_matrix_count"] for row in rows} == {6}
    assert torch.equal(model.token_embedding.weight, protected_before["token_embedding"])
    assert torch.equal(model.position_embedding.weight, protected_before["position_embedding"])
    assert torch.equal(model.ln_f.weight, protected_before["ln_f"])
    assert {row["matrix_type"] for row in rows} == {"W_Q","W_K","W_V","W_O","W_MLP_IN","W_MLP_OUT"}


def test_same_seed_initialization_matches_baseline():
    from level0_baseline.model import GPT as BaselineGPT
    from level0_baseline.model import GPTConfig as BaselineGPTConfig

    kwargs = dict(vocab_size=256, block_size=8, n_layer=2, n_head=1, n_embd=64)
    torch.manual_seed(12345)
    baseline = BaselineGPT(BaselineGPTConfig(**kwargs))
    torch.manual_seed(12345)
    intervention = GPT(GPTConfig(**kwargs))
    assert baseline.state_dict().keys() == intervention.state_dict().keys()
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, intervention.state_dict()[name]), name
