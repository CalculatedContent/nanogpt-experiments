from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from wwgpt.config import ExperimentConfig, ModelConfig, TrainConfig, WWPGDConfig
from wwgpt.model import GPT
from wwgpt.train import WWPGDExtension, run_scientific_single
from wwgpt.cli import _resolve_ww_interval_aliases


def tiny_cfg(steps=6, interval=1):
    return ExperimentConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=600),
        train=TrainConfig(batch_size=2, max_steps=steps, eval_interval=steps+10, spectral_interval=steps+10, checkpoint_interval=steps+10, eval_batches=1, wwpgd_interval=interval),
        wwpgd=WWPGDConfig(enabled=True, extension="wwpgd"),
    )


def tiny_data(tokens=256):
    return SimpleNamespace(train=list(range(tokens)), val=list(range(tokens, tokens*2)), corpus_hash="c", data_manifest={"dataset_name":"d","dataset_config":"c","dataset_revision":"r","realized_tokens":tokens}, tokenizer_manifest={"tokenizer_hash":"t"}, test=None)


def init_state(cfg):
    torch.manual_seed(1)
    m = GPT(cfg.model)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}, "init"


def test_extension_interval_cadences_and_skips(monkeypatch):
    calls = []
    pre_calls = []
    monkeypatch.setattr("wwgpt.train.weightwatcher_details", lambda model: pre_calls.append("pre") or object())
    monkeypatch.setattr("wwgpt.train.apply_external_wwpgd", lambda *args, **kw: calls.append(kw) or [{"projection_event": kw["event_index"], "actual_step": kw["actual_step"], "actual_tokens_seen": kw["actual_tokens_seen"], "layer_name": "fake"}])

    ext = WWPGDExtension(WWPGDConfig(), interval=2)
    rows_by_step = [ext.after_optimizer_step(model=object(), optimizer_step=s, total_optimizer_steps=6, tokens_seen=s*10, collect_pre_details=True)[1] for s in range(1, 7)]
    assert [r[0]["actual_step"] for r in rows_by_step if r] == [2, 4, 6]
    assert [r[0]["projection_event"] for r in rows_by_step if r] == [0, 1, 2]
    assert calls[0]["scheduled_token_fraction"] == pytest.approx(2/6)
    assert len(pre_calls) == 3
    assert rows_by_step[0] == []

    calls.clear()
    ext8 = WWPGDExtension(WWPGDConfig(), interval=8)
    for s in range(1, 25):
        ext8.after_optimizer_step(model=object(), optimizer_step=s, total_optimizer_steps=24, tokens_seen=s)
    assert [c["actual_step"] for c in calls] == [8, 16, 24]
    assert [c["event_index"] for c in calls] == [0, 1, 2]


def test_extension_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="positive integer"):
        WWPGDExtension(WWPGDConfig(), interval=0)


def test_training_interval_manifest_and_counts(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("wwgpt.train.spectral_summary", lambda *a, **k: [])
    monkeypatch.setattr("wwgpt.train.apply_external_wwpgd", lambda *a, **kw: calls.append(kw["actual_step"]) or [{"projection_event": kw["event_index"], "actual_step": kw["actual_step"], "actual_tokens_seen": kw["actual_tokens_seen"], "layer_name": "fake"}])
    cfg = tiny_cfg(steps=6, interval=2)
    state, h = init_state(cfg)
    run = run_scientific_single(tmp_path, "adamw_wwpgd", 3, cfg, tiny_data(), "pair", state, h, 0, 1, device="cpu")
    complete = json.loads((run / "run_complete.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    assert calls == [2, 4, 6]
    assert complete["wwpgd_call_count"] == 3
    assert complete["completed_projection_event_indexes"] == [0, 1, 2]
    assert complete["next_projection_event_index"] == 3
    assert manifest["wwpgd_interval"] == 2
    assert manifest["expected_projection_optimizer_steps"] == [2, 4, 6]
    assert manifest["total_projection_events"] == 3


def test_run_function_cli_override_precedes_config(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("wwgpt.train.spectral_summary", lambda *a, **k: [])
    monkeypatch.setattr("wwgpt.train.apply_external_wwpgd", lambda *a, **kw: calls.append(kw["actual_step"]) or [{"projection_event": kw["event_index"], "actual_step": kw["actual_step"], "layer_name": "fake"}])
    cfg = tiny_cfg(steps=8, interval=2)
    state, h = init_state(cfg)
    run_scientific_single(tmp_path, "adamw_wwpgd", 3, cfg, tiny_data(), "pair", state, h, 0, 1, device="cpu", ww_interval=4)
    assert calls == [4, 8]


def test_alias_conflict_fails():
    args = SimpleNamespace(ww_interval=2, wwpgd_interval=4)
    with pytest.raises(SystemExit, match="conflicting WW-PGD interval aliases"):
        _resolve_ww_interval_aliases(args)
