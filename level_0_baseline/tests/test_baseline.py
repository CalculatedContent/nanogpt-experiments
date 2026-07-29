import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from level0_baseline.data import _progress_message, write_token_splits
from level0_baseline.model import GPT, GPTConfig, transformer_matrix_items
from level0_baseline.optim import make_optimizers
from level0_baseline.train import learning_rate_at, main as train_main


class FakeEncoder:
    n_vocab = 64
    eot_token = 0

    def encode_ordinary(self, text: str) -> list[int]:
        return [1 + (ord(character) % 62) for character in text]


def optimizer_config(name: str):
    return {
        "training": {
            "optimizer": name,
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "muon_learning_rate": 0.02,
            "muon_aux_adamw_learning_rate": 0.001,
        }
    }


def test_forward_and_transformer_matrix_inventory():
    model = GPT(
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_embd=16,
            n_head=1,
            n_layer=2,
        )
    )
    x = torch.randint(0, 64, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(loss)
    matrices = transformer_matrix_items(model)
    assert len(matrices) == 12
    assert {matrix_type for _, matrix_type, _, _ in matrices} == {
        "W_Q",
        "W_K",
        "W_V",
        "W_O",
        "W_MLP_IN",
        "W_MLP_OUT",
    }


def test_adamw_and_muon_steps():
    for optimizer_name in ("adamw", "muon"):
        model = GPT(GPTConfig(vocab_size=64, block_size=8, n_embd=16))
        optimizers = make_optimizers(model, optimizer_config(optimizer_name))
        _, loss = model(
            torch.randint(0, 64, (2, 8)),
            torch.randint(0, 64, (2, 8)),
        )
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        assert len(optimizers) == (1 if optimizer_name == "adamw" else 2)


def test_learning_rate_has_warmup_and_cosine_floor():
    training = {
        "learning_rate": 6e-4,
        "min_lr": 6e-5,
        "warmup_steps": 10,
        "max_steps": 100,
    }
    assert learning_rate_at(0, training) == pytest.approx(6e-5)
    assert learning_rate_at(9, training) == pytest.approx(6e-4)
    assert learning_rate_at(99, training) == pytest.approx(6e-5)
    assert learning_rate_at(50, training) < 6e-4


def test_progress_message_reports_eta_and_stall():
    message = _progress_message(
        collected_tokens=50,
        required_tokens=100,
        documents=4,
        elapsed_seconds=10,
        stalled_seconds=3,
    )
    assert "documents=4" in message
    assert "percent= 50.0%" in message
    assert "elapsed=10s" in message
    assert "eta=10s" in message
    assert "no_new_tokens_for=3s" in message


def test_verbose_bpe_split_preparation_writes_exact_uint16_files(tmp_path, capsys):
    metadata = write_token_splits(
        iter(["abcdef", "ghijkl", "mnopqr"]),
        FakeEncoder(),
        tmp_path,
        train_tokens=3,
        val_tokens=2,
        test_tokens=2,
        verbose=True,
        log_interval_seconds=60,
        dataset_metadata={"dataset_name": "unit"},
    )
    captured = capsys.readouterr()
    assert "[level0-prepare-data] starting" in captured.err
    assert "[level0-prepare-data] complete" in captured.err
    assert np.fromfile(tmp_path / "train.bin", dtype=np.uint16).size == 3
    assert np.fromfile(tmp_path / "val.bin", dtype=np.uint16).size == 2
    assert np.fromfile(tmp_path / "test.bin", dtype=np.uint16).size == 2
    disk_metadata = json.loads((tmp_path / "meta.json").read_text())
    assert disk_metadata["tokenizer"] == "gpt2"
    assert disk_metadata["dtype"] == "uint16"
    assert disk_metadata["splits"] == {"train": 3, "val": 2, "test": 2}
    assert disk_metadata["split_document_counts"] == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    assert disk_metadata["document_disjoint_splits"] is True
    assert metadata["dataset_name"] == "unit"


def test_two_step_cpu_training_completes_with_final_only_test(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir()
    rng = np.random.default_rng(7)
    for split, size in (("train", 2_000), ("val", 800), ("test", 800)):
        rng.integers(0, 64, size=size, dtype=np.uint16).tofile(
            data_root / f"{split}.bin"
        )
    (data_root / "meta.json").write_text(
        json.dumps(
            {
                "tokenizer": "gpt2",
                "vocab_size": 64,
                "dtype": "uint16",
                "splits": {"train": 2_000, "val": 800, "test": 800},
            }
        )
    )
    config = {
        "model": {
            "vocab_size": 64,
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
            "checkpoint_interval": 1,
            "learning_rate": 1e-3,
            "min_lr": 1e-4,
            "warmup_steps": 1,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "grad_clip": 1.0,
            "optimizer": "adamw",
            "muon_learning_rate": 0.02,
            "muon_aux_adamw_learning_rate": 1e-3,
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "seed": 13,
            "compile": False,
        },
        "analysis": {
            "weightwatcher": False,
            "weightwatcher_interval": 1,
            "randomize": False,
            "include_embeddings": False,
            "include_output_head": False,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "level0-train",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--results-root",
            str(results_root),
            "--device",
            "cpu",
        ],
    )
    train_main()
    run = results_root / "adamw_seed_13"
    completion = json.loads((run / "run_complete.json").read_text())
    metrics = pd.read_csv(run / "metrics.csv")
    assert completion["completed"] is True
    assert completion["optimizer_steps"] == 2
    assert int(metrics.iloc[-1]["step"]) == 2
    assert metrics.iloc[:-1]["test_loss"].isna().all()
    assert np.isfinite(metrics.iloc[-1]["test_loss"])
    assert (run / "checkpoint_best.pt").is_file()
    assert (run / "checkpoint_final.pt").is_file()
    assert (run / "selected_checkpoint_metrics.json").is_file()
