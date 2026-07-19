import pandas as pd
import pytest

from wwgpt.train import WWPGDExtension
from wwgpt.ww import measured_projection_spectral_rows
from wwgpt.checkpointing import save_checkpoint, complete_test_checkpoint_state


class DummyCfg:
    q=1.0; target_alpha=2.0; strength=0.1; min_tail=1; blend_eta=.5; cayley_eta=.25; use_detx=True; warmup_events=0; ramp_events=0


def test_immediate_projection_pairing_uses_layer_specific_pre_rows(monkeypatch):
    pre = pd.DataFrame([
        {"longname":"blocks.0.attn.key","alpha":1.1,"xmin":1.0},
        {"longname":"blocks.0.attn.query","alpha":2.2,"xmin":1.0},
    ])
    post = pd.DataFrame([
        {"longname":"blocks.0.attn.key","alpha":1.4,"xmin":1.0},
        {"longname":"blocks.0.attn.query","alpha":2.6,"xmin":1.0},
    ])
    rows = measured_projection_spectral_rows(pre, post, [
        {"layer_name":"blocks.0.attn.key","projection_event":0},
        {"layer_name":"blocks.0.attn.query","projection_event":0},
    ], 2.5)
    assert [r["alpha_before"] for r in rows] == [1.1, 2.2]
    assert [r["alpha_after"] for r in rows] == [1.4, 2.6]


def test_wwpgd_extension_one_pre_call_and_fraction(monkeypatch):
    calls = {"pre":0}
    details = pd.DataFrame([{"longname":"blocks.0.attn.key","alpha":1.1,"xmin":1.0}])
    def fake_details(model):
        calls["pre"] += 1
        return details
    def fake_apply(model, *, event_index, scheduled_token_fraction, actual_step, actual_tokens_seen, cfg):
        return [{"projection_event":event_index,"layer_name":"blocks.0.attn.key","scheduled_token_fraction":scheduled_token_fraction}]
    monkeypatch.setattr("wwgpt.train.weightwatcher_details", fake_details)
    monkeypatch.setattr("wwgpt.train.apply_external_wwpgd", fake_apply)
    ext = WWPGDExtension(DummyCfg(), interval=1)
    pre, rows1 = ext.after_optimizer_step(model=object(), optimizer_step=1, total_optimizer_steps=4, tokens_seen=999, collect_pre_details=True)
    pre, rows2 = ext.after_optimizer_step(model=object(), optimizer_step=4, total_optimizer_steps=4, tokens_seen=999, collect_pre_details=True)
    assert calls["pre"] == 2
    assert rows1[0]["scheduled_token_fraction"] == 0.25
    assert rows2[0]["scheduled_token_fraction"] == 1.0
    assert 0 <= rows1[0]["scheduled_token_fraction"] <= 1
    assert 0 <= rows2[0]["scheduled_token_fraction"] <= 1


def test_save_checkpoint_rejects_incomplete_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="checkpoint missing required keys"):
        save_checkpoint(tmp_path, {"current_step":1, "next_step":2, "compatibility":{}})
    save_checkpoint(tmp_path, complete_test_checkpoint_state(current_step=1, next_step=2))


def test_immediate_projection_post_call_count(monkeypatch):
    calls = {"post":0}
    pre = pd.DataFrame([{"longname":"blocks.0.attn.key","alpha":1.0,"xmin":1.0}])
    post = pd.DataFrame([{"longname":"blocks.0.attn.key","alpha":1.5,"xmin":1.0}])
    def fake_post(model):
        calls["post"] += 1
        return post
    monkeypatch.setattr("wwgpt.ww.weightwatcher_details", fake_post)
    proj = [{"layer_name":"blocks.0.attn.key","projection_event":0}]
    # immediate=false path consumes the pre details for projection only and performs no post analysis.
    assert calls["post"] == 0
    rows = measured_projection_spectral_rows(pre, object(), projection_rows=proj, target_alpha=2.0, step=1, tokens_seen=1, optimizer="adamw_wwpgd", seed=0, pair_id="p", projection_event=0)
    assert calls["post"] == 1
    assert rows[0]["alpha_before"] == 1.0 and rows[0]["alpha_after"] == 1.5

import math
import numpy as np
import torch
from wwgpt.config import ExperimentConfig, ModelConfig, TrainConfig, WWPGDConfig
from wwgpt.data import TokenData
from wwgpt.train import _perplexity_from_cross_entropy, run_scientific_single
from wwgpt.utils import sha256_bytes


def test_metric_formulas_do_not_clip_perplexity_and_gap_semantics():
    assert _perplexity_from_cross_entropy(2.0) == pytest.approx(math.exp(2.0))
    assert math.isinf(_perplexity_from_cross_entropy(1000.0))
    train_ce = 1.25
    val_ce = 1.75
    test_ce = 2.0
    assert val_ce - train_ce == pytest.approx(0.5)
    assert test_ce - train_ce == pytest.approx(0.75)


def test_test_evaluation_runs_once_on_final_selected_checkpoint(tmp_path, monkeypatch):
    cfg = ExperimentConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=4, vocab_size=8),
        train=TrainConfig(batch_size=2, gradient_accumulation=1, max_steps=2, eval_interval=1, checkpoint_interval=10, spectral_interval=99, eval_batches=1, grad_clip=0.0),
        wwpgd=WWPGDConfig(enabled=False, extension="none"),
    )
    manifest = {
        "storage_format": "raw_memmap_v1",
        "dataset_name": "unit",
        "dataset_config": "unit",
        "dataset_revision": "unit",
        "realized_tokens": 16,
        "validation_document_count": 1,
    }
    data = TokenData(
        train=np.full(64, 1, dtype=np.int64),
        val=np.full(64, 2, dtype=np.int64),
        test=np.full(64, 3, dtype=np.int64),
        vocab_size=8,
        corpus_hash="corpus",
        data_manifest=manifest,
        tokenizer_manifest={"tokenizer_hash": "tok"},
    )
    torch.manual_seed(0)
    init_state = {k: v.detach().clone() for k, v in __import__("wwgpt.model", fromlist=["GPT"]).GPT(cfg.model).state_dict().items()}
    calls = {"train": 0, "val": 0, "test": 0}

    def fake_eval(model, probe_x, probe_y, device):
        split = int(np.asarray(probe_y).max())
        if split == 1:
            calls["train"] += 1
            loss = 1.0
        elif split == 2:
            calls["val"] += 1
            loss = 0.5 if calls["val"] == 1 else 0.9
        else:
            calls["test"] += 1
            loss = 0.7
        return {"loss": loss, "perplexity": math.exp(loss), "bits_per_token": loss / math.log(2), "top1_accuracy": 0.25, "top5_accuracy": 0.5, "token_error": 0.75}, loss

    monkeypatch.setattr("wwgpt.train._evaluate_probe_batches", fake_eval)
    monkeypatch.setattr("wwgpt.train.spectral_summary", lambda *a, **k: [])
    run_dir = run_scientific_single(tmp_path, "adamw", 7, cfg, data, "pair", init_state, sha256_bytes(b"init"), 0, 1)
    rows = list(__import__("csv").DictReader((run_dir / "metrics.csv").open()))
    assert calls == {"train": 2, "val": 2, "test": 1}
    assert rows[0]["test_loss"] == "nan"
    assert float(rows[-1]["test_loss"]) == pytest.approx(0.7)
    assert int(float(rows[-1]["selected_checkpoint_step"])) == 1
