import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from level0_baseline.data import _progress_message, write_splits
from level0_baseline.model import GPT, GPTConfig
from level0_baseline.optim import make_optimizers


def cfg(opt):
    return {
        "training": {
            "optimizer": opt,
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "muon_learning_rate": 0.02,
            "muon_aux_adamw_learning_rate": 0.001,
        }
    }


def test_forward_and_accuracy_shape():
    model = GPT(GPTConfig(block_size=8, n_embd=16, n_head=1, n_layer=1))
    x = torch.randint(0, 256, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 256)
    assert torch.isfinite(loss)


def test_adamw_step():
    model = GPT(GPTConfig(block_size=8, n_embd=16))
    optimizers = make_optimizers(model, cfg("adamw"))
    _, loss = model(
        torch.randint(0, 256, (2, 8)),
        torch.randint(0, 256, (2, 8)),
    )
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()


def test_muon_partition_and_step():
    model = GPT(GPTConfig(block_size=8, n_embd=16))
    optimizers = make_optimizers(model, cfg("muon"))
    assert len(optimizers) == 2
    _, loss = model(
        torch.randint(0, 256, (2, 8)),
        torch.randint(0, 256, (2, 8)),
    )
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()


def test_progress_message_reports_elapsed_eta_and_stall():
    message = _progress_message(
        collected_bytes=50,
        required_bytes=100,
        documents=4,
        elapsed_seconds=10,
        stalled_seconds=3,
    )
    assert "documents=4" in message
    assert "percent= 50.0%" in message
    assert "elapsed=10s" in message
    assert "eta=10s" in message
    assert "no_new_bytes_for=3s" in message


def test_verbose_split_preparation_logs_and_writes_files(tmp_path, capsys):
    write_splits(
        iter(["abcdef"]),
        tmp_path,
        train_bytes=2,
        val_bytes=2,
        test_bytes=2,
        verbose=True,
        log_interval_seconds=60,
    )

    captured = capsys.readouterr()
    assert "[level0-prepare-data] starting" in captured.err
    assert "[level0-prepare-data] collection complete" in captured.err
    assert "[level0-prepare-data] complete" in captured.err
    assert (tmp_path / "train.bin").stat().st_size == 2
    assert (tmp_path / "val.bin").stat().st_size == 2
    assert (tmp_path / "test.bin").stat().st_size == 2
    metadata = json.loads((tmp_path / "meta.json").read_text())
    assert metadata["sizes"] == {"train": 2, "val": 2, "test": 2}
