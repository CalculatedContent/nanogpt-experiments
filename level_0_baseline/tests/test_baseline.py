import json
import sys
from copy import deepcopy
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from level0_baseline.analysis import (
    bollinger_band,
    mean_ci95,
    test_summary_table,
)
from level0_baseline.config import (
    load_config,
    optimizer_profile,
    warmup_steps_for,
)
from level0_baseline.data import _progress_message, write_token_splits
from level0_baseline.model import (
    GPT,
    GPTConfig,
    transformer_matrix_items,
)
from level0_baseline.optim import (
    cosine_learning_rate,
    make_optimizer_handles,
    optimizer_step,
)
from level0_baseline.spectral import summarize_spectral_frame
from level0_baseline.train import main as train_main


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


class FakeEncoder:
    n_vocab = 64
    eot_token = 0

    def encode_ordinary(self, text: str) -> list[int]:
        return [1 + (ord(character) % 62) for character in text]


def base_config():
    return load_config(EXPERIMENT_ROOT / "configs" / "level0.yaml")


def tiny_config(optimizer: str) -> dict:
    cfg = deepcopy(base_config())
    cfg["model"] = {
        "vocab_size": 64,
        "block_size": 8,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "dropout": 0.0,
        "bias": False,
        "tie_weights": True,
    }
    cfg["training"].update(
        {
            "seeds": [13],
            "seed": 13,
            "optimizer": optimizer,
            "batch_size": 2,
            "grad_accum_steps": 1,
            "max_steps": 2,
            "target_train_epochs": 0.02,
            "eval_interval": 1,
            "eval_batches": 1,
            "checkpoint_interval": 1,
            "keep_periodic_checkpoints": False,
            "grad_clip": 1.0,
            "compile": False,
        }
    )
    cfg["epoch_monitoring"].update(
        {
            "enabled": True,
            "interval_epochs": 1.0,
            "save_model_checkpoints": True,
            "test_monitoring_only": True,
            "use_for_checkpoint_selection": False,
        }
    )
    cfg["analysis"].update(
        {
            "weightwatcher": False,
            "weightwatcher_interval": 1,
            "weightwatcher_strict": True,
        }
    )
    return cfg


def test_forward_generation_and_transformer_matrix_inventory():
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
    generated = model.generate(
        x[:1, :3],
        4,
        temperature=0.8,
        top_k=10,
        generator=torch.Generator().manual_seed(7),
    )
    assert generated.shape == (1, 7)
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


@pytest.mark.parametrize(
    "optimizer_name", ["sgd_momentum", "adamw", "muon"]
)
def test_all_optimizer_steps_are_finite(optimizer_name):
    model = GPT(
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_embd=16,
            n_head=1,
            n_layer=1,
        )
    )
    profile = optimizer_profile(base_config(), optimizer_name)
    handles = make_optimizer_handles(model, profile)
    _, loss = model(
        torch.randint(0, 64, (2, 8)),
        torch.randint(0, 64, (2, 8)),
    )
    loss.backward()
    optimizer_step(handles)
    assert all(
        torch.isfinite(parameter).all()
        for parameter in model.parameters()
    )
    assert len(handles) == (2 if optimizer_name == "muon" else 1)


def test_profiles_have_distinct_warmups_and_cosine_floors():
    cfg = base_config()
    warmups = {
        name: warmup_steps_for(
            optimizer_profile(cfg, name), 6104
        )
        for name in ("sgd_momentum", "adamw", "muon")
    }
    assert warmups == {
        "sgd_momentum": 610,
        "adamw": 61,
        "muon": 305,
    }
    assert cosine_learning_rate(
        0,
        max_steps=100,
        warmup_steps=10,
        peak_lr=0.1,
        min_lr=0.01,
    ) == pytest.approx(0.01)
    assert cosine_learning_rate(
        9,
        max_steps=100,
        warmup_steps=10,
        peak_lr=0.1,
        min_lr=0.01,
    ) == pytest.approx(0.1)
    assert cosine_learning_rate(
        99,
        max_steps=100,
        warmup_steps=10,
        peak_lr=0.1,
        min_lr=0.01,
    ) == pytest.approx(0.01)


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


def test_bpe_split_preparation_is_exact_and_document_disjoint(
    tmp_path,
):
    metadata = write_token_splits(
        iter(["abcdef", "ghijkl", "mnopqr"]),
        FakeEncoder(),
        tmp_path,
        train_tokens=3,
        val_tokens=2,
        test_tokens=2,
        dataset_metadata={"dataset_name": "unit"},
    )
    assert (
        np.fromfile(
            tmp_path / "train.bin", dtype=np.uint16
        ).size
        == 3
    )
    assert (
        np.fromfile(
            tmp_path / "val.bin", dtype=np.uint16
        ).size
        == 2
    )
    assert (
        np.fromfile(
            tmp_path / "test.bin", dtype=np.uint16
        ).size
        == 2
    )
    disk_metadata = json.loads(
        (tmp_path / "meta.json").read_text()
    )
    assert disk_metadata["tokenizer"] == "gpt2"
    assert disk_metadata["document_disjoint_splits"] is True
    assert metadata["dataset_name"] == "unit"


def test_spectral_summary_preserves_missing_metrics_without_fallback():
    frame = pd.DataFrame(
        {
            "alpha": [2.1, 2.3, np.nan],
            "ERG_gap": [0.0, 2.0, np.nan],
            "D": [0.05, 0.10, 0.15],
        }
    )
    summary = summarize_spectral_frame(
        frame, step=10, tokens_seen=80, epoch=0.5
    )
    assert summary["alpha_n"] == 2
    assert summary["alpha_median"] == pytest.approx(2.2)
    assert summary["ERG_gap_n"] == 2
    assert summary["ERG_gap_mean"] == pytest.approx(1.0)
    assert summary["stable_rank_n"] == 0
    assert np.isnan(summary["stable_rank_median"])


def test_bollinger_and_student_t_statistics():
    frame = pd.DataFrame(
        {
            "optimizer": ["adamw"] * 6,
            "seed": [1, 2, 3, 1, 2, 3],
            "epoch": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "val_loss": [3.0, 4.0, 5.0, 2.0, 3.0, 4.0],
        }
    )
    band = bollinger_band(frame, "val_loss", sigma=2.0)
    first = band.iloc[0]
    assert first["mean"] == pytest.approx(4.0)
    assert first["sd"] == pytest.approx(1.0)
    assert first["lower"] == pytest.approx(2.0)
    assert first["upper"] == pytest.approx(6.0)
    ci = mean_ci95([1.0, 2.0, 3.0])
    assert ci["n"] == 3
    assert ci["mean"] == pytest.approx(2.0)
    assert ci["ci95_half_width"] == pytest.approx(
        4.3026527297 / np.sqrt(3)
    )


@pytest.mark.parametrize(
    "optimizer_name", ["sgd_momentum", "adamw", "muon"]
)
def test_two_step_cpu_training_completes_with_epoch_test_monitoring(
    tmp_path, monkeypatch, optimizer_name
):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir()
    rng = np.random.default_rng(7)
    split_sizes = {
        "train": 2_000,
        "val": 800,
        "test": 800,
    }
    for split, size in split_sizes.items():
        rng.integers(
            0, 64, size=size, dtype=np.uint16
        ).tofile(data_root / f"{split}.bin")
    (data_root / "meta.json").write_text(
        json.dumps(
            {
                "tokenizer": "gpt2",
                "vocab_size": 64,
                "dtype": "uint16",
                "splits": split_sizes,
                "document_disjoint_splits": True,
            }
        )
    )
    config_path = tmp_path / f"{optimizer_name}.yaml"
    config_path.write_text(
        yaml.safe_dump(tiny_config(optimizer_name))
    )
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
            "--optimizer",
            optimizer_name,
            "--seed",
            "13",
            "--device",
            "cpu",
        ],
    )
    train_main()
    run = results_root / optimizer_name / "seed_13"
    completion = json.loads(
        (run / "run_complete.json").read_text()
    )
    metrics = pd.read_csv(run / "metrics.csv")
    epoch_metrics = pd.read_csv(run / "epoch_metrics.csv")
    tests = json.loads(
        (run / "test_results.json").read_text()
    )
    assert completion["completed"] is True
    assert completion["optimizer_steps"] == 2
    assert int(metrics.iloc[-1]["step"]) == 2
    assert metrics.iloc[:-1]["test_loss"].isna().all()
    assert np.isfinite(metrics.iloc[-1]["test_loss"])
    assert len(epoch_metrics) == 1
    assert epoch_metrics.iloc[0]["nominal_epoch"] == pytest.approx(
        0.02
    )
    assert epoch_metrics.iloc[0]["test_monitoring_only"] == 1
    assert Path(
        epoch_metrics.iloc[0]["checkpoint_path"]
    ).is_file()
    assert (
        completion["test_epoch_monitoring_used_for_selection"]
        is False
    )
    assert tests["policy"].startswith(
        "final and validation-selected summary"
    )
    assert (run / "checkpoint_best.pt").is_file()
    assert (run / "checkpoint_final.pt").is_file()
    assert (run / "checkpoint_latest.pt").is_file()


def test_test_summary_uses_three_seed_t_intervals():
    frame = pd.DataFrame(
        {
            "optimizer": ["adamw"] * 3,
            "optimizer_label": ["AdamW"] * 3,
            "seed": [1, 2, 3],
            "checkpoint": ["final"] * 3,
            "step": [10] * 3,
            "test_loss": [2.0, 3.0, 4.0],
            "test_perplexity": [7.0, 8.0, 9.0],
            "test_accuracy": [0.2, 0.3, 0.4],
        }
    )
    summary = test_summary_table(frame)
    loss = summary[
        summary["metric"] == "test_loss"
    ].iloc[0]
    assert loss["n"] == 3
    assert loss["mean"] == pytest.approx(3.0)
    assert loss["ci95_half_width"] > loss["sd"]


def test_four_notebooks_are_valid_and_follow_shared_store_contract():
    paths = sorted(
        (EXPERIMENT_ROOT / "notebooks").glob("*.ipynb")
    )
    assert [path.name for path in paths] == [
        "01_sgd_momentum_baseline.ipynb",
        "02_adamw_baseline.ipynb",
        "03_muon_baseline.ipynb",
        "04_compare_baselines.ipynb",
    ]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        source = "\n".join(
            cell.source for cell in notebook.cells
        )
        assert (
            "Bollinger" in source
            or "Bollinger-style" in source
        )
        assert "ERG_gap" in source
        if path.name.startswith(("01_", "02_", "03_")):
            assert "generate_from_checkpoint" in source
            assert "export_optimizer_core_results" in source
            assert "epoch_metrics" in source
        else:
            assert "load_common_baseline_store" in source
            assert "plot_bollinger_summary" in source
            assert "load_metrics(" not in source
