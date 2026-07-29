from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from level0_baseline.config import validate_config
from level0_baseline.data import (
    _progress_message,
    prepare_token_splits,
)
from level0_baseline.model import GPT, GPTConfig
from level0_baseline.optim import make_optimizers
from level0_baseline.train import (
    evaluate,
    fixed_eval_starts,
    random_batch,
    run_experiment,
)


class FakeTokenizer:
    name = "fake"
    eot_token = 31
    n_vocab = 32

    def encode_ordinary(self, text: str) -> list[int]:
        return [ord(character) % 31 for character in text]


def tiny_config() -> dict:
    return {
        "model": {
            "vocab_size": 64,
            "block_size": 8,
            "n_layer": 2,
            "n_head": 4,
            "n_embd": 32,
            "dropout": 0.0,
            "bias": False,
        },
        "training": {
            "batch_size": 2,
            "grad_accum_steps": 1,
            "max_steps": 2,
            "eval_interval": 1,
            "eval_batches": 2,
            "log_interval": 1,
            "checkpoint_interval": 1,
            "learning_rate": 0.001,
            "min_lr": 0.0001,
            "warmup_steps": 1,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "grad_clip": 1.0,
            "optimizer": "adamw",
            "muon_learning_rate": 0.02,
            "muon_aux_adamw_learning_rate": 0.001,
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "seed": 1337,
            "compile": False,
        },
        "analysis": {
            "weightwatcher": False,
            "weightwatcher_interval": 1,
            "randomize": False,
        },
    }


def write_tiny_data(root: Path, vocab_size: int = 64) -> None:
    root.mkdir(parents=True, exist_ok=True)
    split_tokens = {"train": 512, "val": 128, "test": 128}
    for offset, (split, count) in enumerate(split_tokens.items()):
        values = (np.arange(count, dtype=np.uint16) + offset) % vocab_size
        values.astype("<u2").tofile(root / f"{split}.bin")
    metadata = {
        "format_version": 2,
        "tokenizer": "tiktoken:fake",
        "tokenizer_vocab_size": 32,
        "model_vocab_size": vocab_size,
        "eot_token": 31,
        "dtype": "<u2",
        "split_tokens": split_tokens,
    }
    (root / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_default_model_is_realistic_macbook_scale():
    model = GPT(GPTConfig())
    assert model.num_parameters() > 10_000_000
    assert model.num_parameters() < 25_000_000
    x = torch.randint(0, model.cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, model.cfg.vocab_size)
    assert torch.isfinite(loss)
    assert len(list(model.spectral_matrices())) == 24


def test_adamw_step_uses_standard_decay_groups():
    cfg = tiny_config()
    model = GPT(GPTConfig(**cfg["model"]))
    optimizers = make_optimizers(model, cfg, device_type="cpu")
    assert len(optimizers) == 1
    assert len(optimizers[0].param_groups) == 2
    x = torch.randint(0, 64, (2, 8))
    _, loss = model(x, x)
    assert loss is not None
    loss.backward()
    optimizers[0].step()


def test_progress_message_reports_phase_eta_and_stall():
    message = _progress_message(
        written_tokens=50,
        required_tokens=100,
        documents=4,
        elapsed_seconds=10,
        stalled_seconds=3,
        phase="tokenizing",
        split="train",
    )
    assert "phase=tokenizing" in message
    assert "split=train" in message
    assert "documents=4" in message
    assert "percent= 50.0%" in message
    assert "eta=10s" in message
    assert "no_new_tokens_for=3s" in message


def test_bpe_split_preparation_is_fixed_and_nonoverlapping(tmp_path):
    texts = iter(["abcdefghij", "klmnopqrst", "uvwxyz"])
    metadata = prepare_token_splits(
        texts,
        tmp_path,
        OrderedDict(train=8, val=5, test=4),
        FakeTokenizer(),
        model_vocab_size=64,
        dataset_metadata={"name": "unit"},
    )
    assert metadata["split_tokens"] == {"train": 8, "val": 5, "test": 4}
    assert metadata["discarded_boundary_tokens"] > 0
    assert (tmp_path / "train.bin").stat().st_size == 16
    assert (tmp_path / "val.bin").stat().st_size == 10
    assert (tmp_path / "test.bin").stat().st_size == 8
    on_disk = json.loads((tmp_path / "meta.json").read_text())
    assert on_disk["tokenizer"] == "tiktoken:fake"
    assert on_disk["model_vocab_size"] == 64


def test_evaluation_does_not_advance_training_rng(tmp_path):
    data_path = tmp_path / "train.bin"
    (np.arange(512, dtype=np.uint16) % 64).astype("<u2").tofile(data_path)
    data = np.memmap(data_path, dtype="<u2", mode="r")
    model = GPT(GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=4, n_embd=32))
    starts = fixed_eval_starts(
        len(data), batch_size=2, block_size=8, eval_batches=2, seed=99
    )

    first = torch.Generator().manual_seed(123)
    expected_x, _ = random_batch(data, 2, 8, torch.device("cpu"), first)

    second = torch.Generator().manual_seed(123)
    evaluate(model, data, starts, 8, torch.device("cpu"))
    actual_x, _ = random_batch(data, 2, 8, torch.device("cpu"), second)
    assert torch.equal(expected_x, actual_x)


def test_two_step_run_writes_final_and_selected_test_metrics(tmp_path):
    cfg = tiny_config()
    validate_config(cfg)
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    write_tiny_data(data_root)

    run_dir = run_experiment(
        cfg,
        data_root=data_root,
        results_root=results_root,
        device=torch.device("cpu"),
        disable_weightwatcher=True,
    )

    complete = json.loads((run_dir / "run_complete.json").read_text())
    assert complete["status"] == "complete"
    assert complete["final_step"] == 2
    assert complete["test_evaluation_policy"] == (
        "final_and_validation_selected_checkpoint_only"
    )
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert metrics.step.tolist() == [0, 1, 2]
    assert "test_loss" not in metrics.columns
    final = json.loads((run_dir / "final_metrics.json").read_text())
    selected = json.loads(
        (run_dir / "selected_checkpoint_metrics.json").read_text()
    )
    assert np.isfinite(final["test_loss"])
    assert np.isfinite(selected["test_loss"])
    assert (run_dir / "checkpoint_final.pt").is_file()
    assert (run_dir / "checkpoint_best.pt").is_file()
