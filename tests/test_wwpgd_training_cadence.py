from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
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
        test=None,
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


def test_checkpoint_resume_is_deterministic_and_complete(monkeypatch, tmp_path: Path):
    steps = 4
    cfg = replace(
        _tiny_cfg(steps),
        train=replace(_tiny_cfg(steps).train, checkpoint_interval=2, eval_interval=1, spectral_interval=99),
    )
    init_state, init_hash = _init_state(cfg)
    monkeypatch.setattr("wwgpt.train.spectral_summary", lambda *args, **kwargs: [])
    monkeypatch.setattr("wwgpt.train.apply_external_wwpgd", lambda *args, **kwargs: [{"projection_event": kwargs["event_index"], "actual_step": kwargs["actual_step"], "actual_tokens_seen": kwargs["actual_tokens_seen"], "layer_name": "fake"}])

    full = run_scientific_single(tmp_path / "full", "adamw_wwpgd", 7, cfg, _tiny_data(), "pair_tiny", init_state, init_hash, 0, 1, device="cpu")

    import wwgpt.train as train_mod
    real_save = train_mod.save_checkpoint
    interrupted = {"run": None}

    def save_then_interrupt(run_dir, state):
        path = real_save(run_dir, state)
        if int(state["current_step"]) == 2:
            interrupted["run"] = Path(run_dir)
            raise KeyboardInterrupt("unit-test interruption after checkpoint")
        return path

    monkeypatch.setattr(train_mod, "save_checkpoint", save_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="unit-test interruption"):
        run_scientific_single(tmp_path / "resumed", "adamw_wwpgd", 7, cfg, _tiny_data(), "pair_tiny", init_state, init_hash, 0, 1, device="cpu")
    assert interrupted["run"] is not None

    monkeypatch.setattr(train_mod, "save_checkpoint", real_save)
    resumed = run_scientific_single(tmp_path / "resumed", "adamw_wwpgd", 7, cfg, _tiny_data(), "pair_tiny", init_state, init_hash, 0, 1, device="cpu", resume=True)

    full_ckpt = torch.load(full / "checkpoints" / "checkpoint_step_000004.pt", map_location="cpu", weights_only=False)
    resumed_ckpt = torch.load(resumed / "checkpoints" / "checkpoint_step_000004.pt", map_location="cpu", weights_only=False)
    for required in ["model_state_dict", "optimizer_state_dict", "base_optimizer_state_dict", "scheduler_state_dict", "wwpgd_state", "gradient_scaler_state_dict", "python_random_state", "numpy_random_state", "torch_cpu_rng_state", "accelerator_rng_states", "training_reader_state", "tokens_processed", "optimizer_step_count", "best_validation_loss", "resolved_config", "optimizer_fingerprint", "data_hash", "tokenizer_hash"]:
        assert required in resumed_ckpt
    assert resumed_ckpt["tokens_processed"] == full_ckpt["tokens_processed"] == steps * cfg.train.batch_size * cfg.model.block_size * cfg.train.gradient_accumulation
    assert resumed_ckpt["optimizer_step_count"] == full_ckpt["optimizer_step_count"] == steps
    assert resumed_ckpt["wwpgd_state"]["call_count"] == steps
    for name, tensor in full_ckpt["model_state_dict"].items():
        assert torch.equal(tensor, resumed_ckpt["model_state_dict"][name]), name

    ignored = {"elapsed_time", "wall_clock_time", "tokens_per_second", "examples_per_second", "projection_overhead", "weightwatcher_overhead", "peak_memory"}
    def comparable(rows):
        return [{k: ("NaN" if isinstance(v, float) and v != v else v) for k, v in row.items() if k not in ignored} for row in rows]
    assert comparable(full_ckpt["metrics_rows"]) == comparable(resumed_ckpt["metrics_rows"])

    bad = dict(resumed_ckpt)
    bad["compatibility"] = dict(bad["compatibility"], optimizer_fingerprint="wrong")
    with pytest.raises(RuntimeError, match="checkpoint compatibility validation failed.*optimizer_fingerprint"):
        train_mod.assert_checkpoint_compatible(bad, resumed_ckpt["compatibility"])
