import pandas as pd
import pytest

from wwgpt.train import WWPGDExtension
from wwgpt.ww import measured_projection_spectral_rows
from wwgpt.checkpointing import save_checkpoint, complete_test_checkpoint_state


class DummyCfg:
    q=1.0; target_alpha=2.0; strength=0.1; min_tail=1; blend_eta=.5; cayley_eta=.25; use_detx=True; warmup_events=0; ramp_events=1


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
    ext = WWPGDExtension(DummyCfg(), interval=2)
    assert ext.after_optimizer_step(model=object(), optimizer_step=1, total_optimizer_steps=4, tokens_seen=999) == []
    pre, rows1 = ext.after_optimizer_step(model=object(), optimizer_step=2, total_optimizer_steps=4, tokens_seen=999)
    pre, rows2 = ext.after_optimizer_step(model=object(), optimizer_step=4, total_optimizer_steps=4, tokens_seen=999)
    assert calls["pre"] == 2
    assert rows1[0]["scheduled_token_fraction"] == 0.5
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
