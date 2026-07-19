from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import torch

from wwgpt.config import ExperimentConfig, ModelConfig, TrainConfig, WWPGDConfig
from wwgpt.model import GPT
from wwgpt.train import run_scientific_single


def _tiny_data(tokens: int = 256):
    train = list(range(3, tokens + 3))
    val = list(range(tokens + 3, tokens * 2 + 3))
    return SimpleNamespace(
        train=train,
        val=val,
        corpus_hash="tiny-corpus",
        data_manifest={
            "dataset_name": "unit",
            "dataset_config": "tiny",
            "dataset_revision": "test",
            "realized_tokens": len(train),
            "validation_document_count": 1,
        },
        tokenizer_manifest={"tokenizer_hash": "tiny-tokenizer"},
    )


def _tiny_cfg(steps: int) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=600),
        train=TrainConfig(
            batch_size=2,
            max_steps=steps,
            eval_interval=steps + 10,
            spectral_interval=steps + 10,
            checkpoint_interval=steps + 10,
            eval_batches=1,
            wwpgd_interval=1,
        ),
        wwpgd=WWPGDConfig(enabled=True, extension="wwpgd"),
    )


def _init_state(cfg: ExperimentConfig):
    torch.manual_seed(123)
    model = GPT(cfg.model)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return state, "tiny-init"


def test_wwpgd_runs_once_per_successful_optimizer_step(monkeypatch, tmp_path: Path):
    steps = 3
    calls = []

    def fake_apply(model, *, event_index, scheduled_token_fraction, actual_step, actual_tokens_seen, cfg):
        calls.append(actual_step)
        return [
            {"projection_event": event_index, "layer_name": "blocks.0.attn.key"},
            {"projection_event": event_index, "layer_name": "blocks.0.attn.value"},
        ]

    monkeypatch.setattr("wwgpt.train.apply_external_wwpgd", fake_apply)
    monkeypatch.setattr("wwgpt.train.spectral_summary", lambda *args, **kwargs: [])

    cfg = _tiny_cfg(steps)
    init_state, init_hash = _init_state(cfg)
    run_dir = run_scientific_single(
        tmp_path,
        "adamw",
        7,
        cfg,
        _tiny_data(),
        "pair_tiny",
        init_state,
        init_hash,
        0,
        1,
        device="cpu",
        immediate_projection_spectral=False,
    )

    complete = json.loads((run_dir / "run_complete.json").read_text())
    assert calls == [1, 2, 3]
    assert complete["optimizer_step_count"] == steps
    assert complete["wwpgd_call_count"] == steps
    assert complete["projected_matrix_count"] == steps * 2


def test_baseline_arm_records_zero_wwpgd_calls(monkeypatch, tmp_path: Path):
    steps = 2
    calls = []
    monkeypatch.setattr("wwgpt.train.apply_external_wwpgd", lambda *args, **kwargs: calls.append(kwargs.get("actual_step")) or [])
    monkeypatch.setattr("wwgpt.train.spectral_summary", lambda *args, **kwargs: [])

    cfg = replace(_tiny_cfg(steps), wwpgd=WWPGDConfig(enabled=False, extension="none"))
    init_state, init_hash = _init_state(cfg)
    run_dir = run_scientific_single(
        tmp_path,
        "adamw",
        7,
        cfg,
        _tiny_data(),
        "pair_tiny",
        init_state,
        init_hash,
        0,
        1,
        device="cpu",
        immediate_projection_spectral=False,
    )

    complete = json.loads((run_dir / "run_complete.json").read_text())
    assert calls == []
    assert complete["optimizer_step_count"] == steps
    assert complete["wwpgd_call_count"] == 0
    assert complete["projected_matrix_count"] == 0
