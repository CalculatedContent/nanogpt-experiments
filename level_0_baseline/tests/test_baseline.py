from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from level0_baseline.config import load_config
from level0_baseline.data import (
    _progress_message,
    validate_prepared_data,
    write_token_splits,
)
from level0_baseline.model import GPT, GPTConfig
from level0_baseline.optim import make_optimizers
from level0_baseline.train import (
    _SpectralSnapshot,
    make_fixed_probe,
    run_training,
)


class FakeTokenizer:
    name = "gpt2"
    n_vocab = 32
    eot_token = 0

    @staticmethod
    def encode_ordinary(text: str) -> list[int]:
        return [1 + (ord(character) % 31) for character in text]


def tiny_training_config() -> dict:
    return {
        "data": {
            "dataset_name": "unit",
            "dataset_config": "unit",
            "dataset_split": "train",
            "dataset_revision": "unit",
            "tokenizer": "gpt2",
            "dtype": "uint16",
            "train_tokens": 256,
            "val_tokens": 128,
            "test_tokens": 128,
        },
        "model": {
            "vocab_size": 32,
            "block_size": 8,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
            "tie_weights": True,
        },
        "training": {
            "batch_size": 2,
            "grad_accum_steps": 2,
            "max_steps": 2,
            "eval_interval": 1,
            "eval_batches": 2,
            "eval_batch_size": 2,
            "test_eval_batches": 2,
            "checkpoint_interval": 1,
            "learning_rate": 0.001,
            "muon_learning_rate": 0.02,
            "muon_aux_adamw_learning_rate": 0.001,
            "min_lr": 0.0001,
            "warmup_steps": 1,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "grad_clip": 1.0,
            "layer_lr_decay": 1.0,
            "optimizer": "adamw",
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "seed": 7,
            "compile": False,
        },
        "analysis": {
            "weightwatcher": False,
            "weightwatcher_interval": 1,
            "randomize": False,
        },
    }


def prepare_tiny_data(path: Path) -> None:
    write_token_splits(
        iter(["abcdefghijklmnopqrstuvwxyz"] * 64),
        path,
        tokenizer=FakeTokenizer(),
        train_tokens=256,
        val_tokens=128,
        test_tokens=128,
        dataset_metadata={
            "dataset_name": "unit",
            "dataset_config": "unit",
            "dataset_split": "train",
            "dataset_revision": "unit",
        },
    )


def test_default_config_is_realistic_gpt2_bpe_baseline():
    config = load_config(PACKAGE_ROOT / "configs" / "level0.yaml")
    assert config["data"]["tokenizer"] == "gpt2"
    assert config["data"]["dtype"] == "uint16"
    assert config["model"] == {
        "vocab_size": 50257,
        "block_size": 256,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "dropout": 0.0,
        "bias": False,
        "tie_weights": True,
    }
    assert config["training"]["batch_size"] == 4
    assert config["training"]["grad_accum_steps"] == 8
    assert config["training"]["layer_lr_decay"] == 1.0
    assert config["analysis"]["randomize"] is False


def test_model_size_and_weightwatcher_scope_are_not_toy_byte_baseline():
    model = GPT(GPTConfig())
    assert model.num_parameters() == 7_253_248
    assert len(model.blocks) == 4
    assert len(model.spectral_layers()) == 24
    names = [name for name, _ in model.spectral_layers()]
    assert "block_00_W_Q" in names
    assert "block_03_W_MLP_OUT" in names
    snapshot = _SpectralSnapshot(model)
    assert len(snapshot.layers) == 24
    assert "token_embedding" not in snapshot.layers


def test_adamw_uses_decoupled_decay_and_explicit_lr_groups():
    config = tiny_training_config()
    model = GPT(GPTConfig(**config["model"]))
    optimizer = make_optimizers(model, config)[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert {float(group["weight_decay"]) for group in optimizer.param_groups} == {
        0.0,
        0.01,
    }
    assert all(float(group["lr_multiplier"]) == 1.0 for group in optimizer.param_groups)


def test_progress_message_reports_tokens_elapsed_eta_and_stall():
    message = _progress_message(
        collected_tokens=50,
        required_tokens=100,
        documents=4,
        elapsed_seconds=10,
        stalled_seconds=3,
    )
    assert "documents=4" in message
    assert "tokens=50/100" in message
    assert "percent= 50.0%" in message
    assert "elapsed=10s" in message
    assert "eta=10s" in message
    assert "no_new_tokens_for=3s" in message


def test_gpt2_token_split_preparation_is_atomic_and_valid(tmp_path, capsys):
    metadata = write_token_splits(
        iter(["abcdef"] * 10),
        tmp_path,
        tokenizer=FakeTokenizer(),
        train_tokens=8,
        val_tokens=4,
        test_tokens=4,
        dataset_metadata={
            "dataset_name": "unit",
            "dataset_config": "unit",
            "dataset_split": "train",
            "dataset_revision": "unit",
        },
        verbose=True,
        log_interval_seconds=60,
    )
    captured = capsys.readouterr()
    assert "[level0-prepare-data] starting" in captured.err
    assert "tokenization complete" in captured.err
    assert metadata["tokenizer"] == "gpt2"
    assert metadata["dtype"] == "uint16"
    assert metadata["document_disjoint_splits"] is True
    assert metadata["boundary_tokens_discarded"] > 0
    assert (tmp_path / "train.bin").stat().st_size == 16
    assert (tmp_path / "val.bin").stat().st_size == 8
    assert (tmp_path / "test.bin").stat().st_size == 8
    assert not list(tmp_path.glob("*.tmp"))
    validated = validate_prepared_data(tmp_path, verify_hashes=True)
    assert validated["split_tokens"] == {"train": 8, "val": 4, "test": 4}


def test_old_byte_data_is_rejected(tmp_path):
    (tmp_path / "meta.json").write_text(
        json.dumps({"tokenizer": "utf8-byte", "dtype": "uint8"})
    )
    with pytest.raises(ValueError, match="corrected GPT-2-BPE"):
        validate_prepared_data(tmp_path)


def test_fixed_probes_do_not_advance_training_rng():
    data = np.arange(256, dtype=np.uint16)
    training_generator = torch.Generator().manual_seed(123)
    state_before = training_generator.get_state().clone()
    first = make_fixed_probe(
        data,
        batch_size=2,
        block_size=8,
        batches=2,
        seed=456,
    )
    second = make_fixed_probe(
        data,
        batch_size=2,
        block_size=8,
        batches=2,
        seed=456,
    )
    assert torch.equal(training_generator.get_state(), state_before)
    assert all(
        torch.equal(a_tensor, b_tensor)
        for a_batch, b_batch in zip(first, second)
        for a_tensor, b_tensor in zip(a_batch, b_batch)
    )


def test_two_step_training_writes_complete_scientific_artifacts(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    prepare_tiny_data(data_root)
    run = run_training(
        tiny_training_config(),
        data_root=data_root,
        results_root=results_root,
        device="cpu",
    )

    completion = json.loads((run / "run_complete.json").read_text())
    selected = json.loads((run / "selected_checkpoint_metrics.json").read_text())
    final = json.loads((run / "final_metrics.json").read_text())
    metrics = pd.read_csv(run / "metrics.csv")
    manifest = json.loads((run / "manifest.json").read_text())

    assert completion["completed"] is True
    assert completion["optimizer_steps"] == 2
    assert completion["tokens_seen"] == 64
    assert completion["weightwatcher_measurements"] == 0
    assert len(metrics) == 3
    assert list(metrics["step"]) == [0, 1, 2]
    assert not any(column.startswith("test_") for column in metrics.columns)
    assert selected["selected_step"] in {0, 1, 2}
    assert np.isfinite(selected["test_loss"])
    assert np.isfinite(final["test_loss"])
    assert manifest["evaluation_protocol"]["test"] == (
        "final_and_validation_selected_only"
    )
    assert manifest["evaluation_protocol"][
        "training_rng_isolated_from_evaluation"
    ] is True
    assert (run / "checkpoint_best.pt").is_file()
    assert (run / "checkpoint_final.pt").is_file()


def test_nonempty_run_requires_explicit_overwrite(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    prepare_tiny_data(data_root)
    config = tiny_training_config()
    run_training(config, data_root=data_root, results_root=results_root, device="cpu")
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_training(config, data_root=data_root, results_root=results_root, device="cpu")
