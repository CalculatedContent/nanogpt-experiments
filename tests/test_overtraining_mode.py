from __future__ import annotations

import csv
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from wwgpt.analysis import build_run_inventory
from wwgpt.config import ExperimentConfig, ModelConfig, TrainConfig, WWPGDConfig
from wwgpt.data import TokenData, _budget_identity
from wwgpt.model import GPT
from wwgpt.run_health import generate_run_health
from wwgpt.train import run_scientific_single
from wwgpt.utils import sha256_bytes


def _metric(loss: float) -> dict[str, float]:
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "bits_per_token": loss / math.log(2),
        "top1_accuracy": 0.25,
        "top5_accuracy": 0.5,
        "token_error": 0.75,
    }


def test_fixed_corpus_overtraining_runs_extended_horizon_and_records_collapse(
    tmp_path, monkeypatch
):
    cfg = ExperimentConfig(
        model=ModelConfig(
            n_layer=1,
            n_head=1,
            n_embd=1,
            block_size=4,
            vocab_size=8,
            mlp_mult=1,
        ),
        train=TrainConfig(
            batch_size=1,
            gradient_accumulation=1,
            max_steps=5,
            allow_overtraining=True,
            eval_interval=1,
            checkpoint_interval=5,
            spectral_interval=99,
            eval_batches=1,
            grad_clip=0.0,
            training_sampling="random_window",
            evaluation_sampling="fixed_probe",
            test_evaluation_mode="final_checkpoint",
            lr_schedule="warmup_cosine",
            warmup_ratio=0.2,
        ),
        wwpgd=WWPGDConfig(enabled=False, extension="none"),
    )
    nominal_tokens = 3 * cfg.train.batch_size * cfg.model.block_size
    manifest = {
        "storage_format": "raw_memmap_v1",
        "dataset_name": "unit",
        "dataset_config": "unit",
        "dataset_revision": "unit",
        "realized_tokens": nominal_tokens,
        "validation_tokens": 64,
        "test_tokens": 64,
        "validation_document_count": 1,
        "test_document_count": 1,
    }
    data = TokenData(
        train=np.full(64, 1, dtype=np.int64),
        val=np.full(64, 2, dtype=np.int64),
        test=np.full(64, 3, dtype=np.int64),
        vocab_size=8,
        corpus_hash="corpus",
        data_manifest=manifest,
        tokenizer_manifest={"tokenizer_hash": "tok"},
    )
    torch.manual_seed(0)
    init_state = {
        name: value.detach().clone()
        for name, value in GPT(cfg.model).state_dict().items()
    }
    calls = {"train": 0, "val": 0, "test": 0}

    def fake_eval(_model, _probe_x, probe_y, _device):
        split = int(np.asarray(probe_y).max())
        if split == 1:
            calls["train"] += 1
            loss = 1.0
        elif split == 2:
            calls["val"] += 1
            loss = 0.5 if calls["val"] == 1 else 0.9
        else:
            calls["test"] += 1
            loss = 0.8 if calls["test"] == 1 else 0.7
        return _metric(loss), loss

    monkeypatch.setattr("wwgpt.train._evaluate_probe_batches", fake_eval)
    monkeypatch.setattr("wwgpt.train.spectral_summary", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "wwgpt.train.nonmutating_weightwatcher_details",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    run_dir = run_scientific_single(
        tmp_path,
        "adamw",
        7,
        cfg,
        data,
        "pair",
        init_state,
        sha256_bytes(b"init"),
        0,
        1,
        device="cpu",
    )

    run_manifest = json.loads((run_dir / "manifest.json").read_text())
    completion = json.loads((run_dir / "run_complete.json").read_text())
    final = json.loads((run_dir / "final_checkpoint_metrics.json").read_text())
    selected = json.loads(
        (run_dir / "selected_checkpoint_metrics.json").read_text()
    )
    lr_rows = list(csv.DictReader((run_dir / "lrs.csv").open()))

    assert run_manifest["training_protocol"] == "fixed_corpus_overtraining"
    assert run_manifest["valid_for_scaling_law_fit"] is False
    assert run_manifest["nominal_optimizer_steps"] == 3
    assert run_manifest["resolved_optimizer_steps"] == 5
    assert run_manifest["nominal_realized_train_tokens"] == 12
    assert run_manifest["realized_train_tokens"] == 20
    assert run_manifest["overtraining_optimizer_steps"] == 2
    assert run_manifest["overtraining_tokens"] == 8
    assert run_manifest["resolved_lr_decay_steps"] == 5

    assert completion["overtraining_active"] is True
    assert completion["optimizer_step_limit_source"] == "overtraining_max_steps"
    assert completion["step"] == 5
    assert completion["final_checkpoint_test_loss"] == pytest.approx(0.8)
    assert completion["validation_selected_checkpoint_test_loss"] == pytest.approx(
        0.7
    )
    assert completion["test_loss_delta_final_minus_selected"] == pytest.approx(
        0.1
    )
    assert final["test_evaluated"] is True
    assert selected["selected_step"] == 1
    assert selected["test_evaluated"] is True
    assert calls["test"] == 2
    assert {int(row["optimizer_step"]) for row in lr_rows} == {1, 2, 3, 4, 5}

    health = generate_run_health(run_dir)
    assert health["ready_for_analysis"] is True, health["findings"]

    inventory = build_run_inventory(
        [
            {
                "run_dir": run_dir,
                "manifest": run_manifest,
                "seed": 7,
                "level": 0,
                "token_multiplier": 1,
                "pair_id": "pair",
                "optimizer_raw": "adamw",
                "optimizer_family": "adamw",
                "base_optimizer": "adamw",
                "extension": "none",
            }
        ]
    ).iloc[0]
    assert inventory.training_protocol == "fixed_corpus_overtraining"
    assert inventory.overtraining_active
    assert not inventory.valid_for_scaling_law_fit
    assert inventory.test_loss_delta_final_minus_selected == pytest.approx(0.1)


def test_overtraining_does_not_change_the_prepared_corpus_identity():
    nominal = ExperimentConfig(
        model=ModelConfig(
            n_layer=1,
            n_head=1,
            n_embd=64,
            block_size=256,
            vocab_size=8192,
            profile_name="scaling_level0_adaptive_alpha",
        ),
        train=TrainConfig(
            batch_size=16,
            gradient_accumulation=1,
            training_sampling="random_window",
            evaluation_sampling="random_per_eval",
        ),
    )
    overtraining = ExperimentConfig(
        model=nominal.model,
        train=TrainConfig(
            batch_size=16,
            gradient_accumulation=1,
            max_steps=1000,
            allow_overtraining=True,
            training_sampling="random_window",
            evaluation_sampling="fixed_probe",
            test_evaluation_mode="final_checkpoint",
        ),
    )
    assert _budget_identity(nominal, 0, 20) == _budget_identity(
        overtraining, 0, 20
    )
