import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from level0_baseline.baseline_store import (
    bollinger_summary_long,
    export_optimizer_core_results,
    load_common_baseline_store,
)
from level0_baseline.config import SUPPORTED_OPTIMIZERS
from level0_baseline.epoch_monitor import (
    epoch_checkpoint_path,
    epoch_monitor_step_map,
    save_epoch_model_checkpoint,
)
from level0_baseline.model import GPT, GPTConfig


def _write_synthetic_run(
    results_root: Path,
    *,
    optimizer: str,
    seed: int,
    offset: float,
) -> None:
    run_dir = results_root / optimizer / f"seed_{seed}"
    (run_dir / "spectral").mkdir(parents=True)
    (run_dir / "run_complete.json").write_text(
        json.dumps({"completed": True, "optimizer": optimizer, "seed": seed})
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_fingerprint": f"{optimizer}-{seed}",
                "planned_training_tokens": 1000,
                "planned_train_epochs": 2.0,
                "tokens_per_optimizer_step": 10,
                "config": {
                    "model": {
                        "vocab_size": 64,
                        "block_size": 8,
                        "n_layer": 1,
                        "n_head": 1,
                        "n_embd": 16,
                    }
                },
                "data_manifest": {
                    "tokenizer": "gpt2",
                    "splits": {"train": 500, "val": 100, "test": 100},
                },
            }
        )
    )
    trajectory = pd.DataFrame(
        {
            "step": [0, 50, 100],
            "tokens_seen": [0, 500, 1000],
            "epoch": [0.0, 1.0, 2.0],
            "train_loss": [8.0, 6.0 + offset, 5.0 + offset],
            "train_perplexity": [2980.0, 403.0, 148.0],
            "train_accuracy": [0.01, 0.12 + offset / 100, 0.20 + offset / 100],
            "val_loss": [8.1, 6.2 + offset, 5.2 + offset],
            "val_perplexity": [3294.0, 493.0, 181.0],
            "val_accuracy": [0.01, 0.10 + offset / 100, 0.18 + offset / 100],
            "test_loss": [np.nan, 6.3 + offset, 5.3 + offset],
            "test_perplexity": [np.nan, 544.0, 200.0],
            "test_accuracy": [np.nan, 0.09 + offset / 100, 0.17 + offset / 100],
            "val_generalization_gap": [0.1, 0.2, 0.2],
            "test_generalization_gap": [np.nan, 0.3, 0.3],
            "grad_norm_pre_clip": [np.nan, 1.2, 0.8],
            "grad_norm_post_clip": [np.nan, 1.0, 0.8],
            "weight_norm": [10.0, 11.0, 12.0],
            "update_norm_since_eval": [0.0, 0.3, 0.2],
            "update_to_weight_ratio": [0.0, 0.027, 0.017],
            "tokens_per_sec": [0.0, 1000.0, 1100.0],
        }
    )
    trajectory.to_csv(run_dir / "metrics.csv", index=False)
    epoch = trajectory.iloc[1:].copy()
    epoch["nominal_epoch"] = [1.0, 2.0]
    epoch["checkpoint_path"] = [
        str(run_dir / "epoch_checkpoints" / "one.pt"),
        str(run_dir / "epoch_checkpoints" / "two.pt"),
    ]
    epoch["test_monitoring_only"] = 1
    epoch.to_csv(run_dir / "epoch_metrics.csv", index=False)
    pd.DataFrame(
        {
            "step": [50, 100],
            "tokens_seen": [500, 1000],
            "epoch": [1.0, 2.0],
            "alpha_mean": [2.4 + offset, 2.2 + offset],
            "alpha_median": [2.3 + offset, 2.1 + offset],
            "ERG_gap_mean": [1.0, 0.5],
            "ERG_gap_median": [1.0, 0.0],
            "D_mean": [0.10, 0.08],
            "D_median": [0.09, 0.07],
            "stable_rank_mean": [5.0, 5.5],
            "stable_rank_median": [5.0, 5.5],
            "mp_softrank_mean": [0.5, 0.6],
            "mp_softrank_median": [0.5, 0.6],
        }
    ).to_csv(run_dir / "spectral" / "summary.csv", index=False)
    (run_dir / "test_results.json").write_text(
        json.dumps(
            {
                "final": {
                    "step": 100,
                    "loss": 5.3 + offset,
                    "perplexity": 200.0 + offset,
                    "accuracy": 0.17 + offset / 100,
                },
                "validation_selected": {
                    "step": 100,
                    "validation_loss": 5.2 + offset,
                    "loss": 5.3 + offset,
                    "perplexity": 200.0 + offset,
                    "accuracy": 0.17 + offset / 100,
                },
            }
        )
    )


def test_epoch_monitor_map_uses_integer_epochs_and_final_step():
    mapping = epoch_monitor_step_map(
        train_tokens=10_000_000,
        tokens_per_step=8192,
        max_steps=6104,
        target_epochs=5.0,
        interval_epochs=1.0,
    )
    assert mapping == {
        1221: 1.0,
        2441: 2.0,
        3662: 3.0,
        4883: 4.0,
        6104: 5.0,
    }


def test_tiny_epoch_monitor_map_collapses_to_final_step():
    mapping = epoch_monitor_step_map(
        train_tokens=2000,
        tokens_per_step=16,
        max_steps=2,
        target_epochs=0.02,
        interval_epochs=1.0,
    )
    assert mapping == {2: 0.02}


def test_model_only_epoch_checkpoint_is_auditable(tmp_path):
    model = GPT(
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=16,
        )
    )
    path = save_epoch_model_checkpoint(
        tmp_path,
        model=model,
        step=10,
        nominal_epoch=1.0,
        actual_epoch=1.01,
        cfg={"model": model.cfg.__dict__},
        optimizer_name="adamw",
        seed=13,
        protocol_fingerprint="abc",
    )
    assert path == epoch_checkpoint_path(
        tmp_path, nominal_epoch=1.0, step=10
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["purpose"] == "preregistered_epoch_monitoring_model_only"
    assert payload["step"] == 10
    assert payload["nominal_epoch"] == pytest.approx(1.0)
    assert "optimizer_states" not in payload


def test_bollinger_summary_is_mean_plus_or_minus_two_sample_sd():
    frame = pd.DataFrame(
        {
            "optimizer": ["adamw"] * 3,
            "optimizer_label": ["AdamW"] * 3,
            "seed": [1, 2, 3],
            "nominal_epoch": [1.0, 1.0, 1.0],
            "test_loss": [4.0, 5.0, 6.0],
        }
    )
    result = bollinger_summary_long(
        frame,
        ["test_loss"],
        x="nominal_epoch",
        sigma=2.0,
    ).iloc[0]
    assert result["mean"] == pytest.approx(5.0)
    assert result["sd"] == pytest.approx(1.0)
    assert result["lower"] == pytest.approx(3.0)
    assert result["upper"] == pytest.approx(7.0)
    assert result["n"] == 3


def test_optimizer_notebooks_publish_and_comparison_reads_common_store():
    notebook_root = EXPERIMENT_ROOT / "notebooks"
    for name in (
        "01_sgd_momentum_baseline.ipynb",
        "02_adamw_baseline.ipynb",
        "03_muon_baseline.ipynb",
    ):
        notebook = nbformat.read(notebook_root / name, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "export_optimizer_core_results" in source
        assert "BASELINE_STORE" in source
        assert "epoch_metrics" in source
    comparison = nbformat.read(
        notebook_root / "04_compare_baselines.ipynb", as_version=4
    )
    source = "\n".join(cell.source for cell in comparison.cells)
    assert "load_common_baseline_store" in source
    assert "plot_bollinger_summary" in source
    assert "load_metrics(" not in source
    assert "load_spectral_metrics(" not in source


def test_three_optimizer_exports_create_complete_common_store(tmp_path):
    results_root = tmp_path / "results"
    store_root = tmp_path / "baseline_reference"
    seeds = (1, 2, 3)
    for optimizer_index, optimizer in enumerate(SUPPORTED_OPTIMIZERS):
        for seed_index, seed in enumerate(seeds):
            _write_synthetic_run(
                results_root,
                optimizer=optimizer,
                seed=seed,
                offset=optimizer_index * 0.1 + seed_index * 0.01,
            )
        export_optimizer_core_results(
            results_root,
            store_root,
            optimizer=optimizer,
            seeds=seeds,
            sigma=2.0,
        )

    store = load_common_baseline_store(
        store_root,
        require_optimizers=SUPPORTED_OPTIMIZERS,
    )
    assert store["manifest"]["missing_optimizers"] == []
    assert set(store["manifest"]["available_optimizers"]) == set(
        SUPPORTED_OPTIMIZERS
    )
    assert set(store["epoch_runs"]["optimizer"]) == set(SUPPORTED_OPTIMIZERS)
    assert set(store["epoch_summary"]["metric"]) >= {
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "test_perplexity",
    }
    final_epoch = store["epoch_summary"][
        (store["epoch_summary"]["optimizer"] == "adamw")
        & (store["epoch_summary"]["metric"] == "test_loss")
        & (store["epoch_summary"]["nominal_epoch"] == 2.0)
    ].iloc[0]
    assert final_epoch["n"] == 3
    assert final_epoch["upper"] > final_epoch["mean"]
    assert (
        store_root
        / "summaries"
        / "terminal_test_student_t_summary.csv"
    ).is_file()
