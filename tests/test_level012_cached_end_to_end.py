from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from wwgpt.config import load_config
from wwgpt.data import TokenData
from wwgpt.model import GPT
from wwgpt.train import run_scientific_single
from wwgpt.utils import sha256_bytes
from wwgpt.ww import StockWWPGDCandidate, projected_matrix_modules


def _tiny_scientific_data(vocab_size: int, realized_tokens: int) -> TokenData:
    train = np.arange(512, dtype=np.int64) % vocab_size
    val = (np.arange(128, dtype=np.int64) + 7) % vocab_size
    test = (np.arange(128, dtype=np.int64) + 13) % vocab_size
    return TokenData(
        train=train,
        val=val,
        test=test,
        vocab_size=vocab_size,
        corpus_hash=sha256_bytes(train.tobytes() + val.tobytes() + test.tobytes()),
        data_manifest={
            "storage_format": "raw_memmap_v1",
            "dataset_name": "offline-level-smoke",
            "dataset_config": "fixture",
            "dataset_revision": "fixture-v1",
            "realized_tokens": realized_tokens,
            "validation_tokens": len(val),
            "test_tokens": len(test),
            "validation_document_count": 1,
            "test_document_count": 1,
        },
        tokenizer_manifest={
            "tokenizer_hash": sha256_bytes(b"offline-level-smoke-tokenizer"),
            "tokenizer_type": "offline",
            "vocab_size": vocab_size,
        },
    )


def _fake_candidate_builder(model, *, event_index=0, actual_step=0, cfg=None, layer_selector=None, **_kwargs):
    rows = []
    originals = {}
    candidates = {}
    relative = {}
    changed = {}
    diagnostics = []

    for index, (name, weight) in enumerate(projected_matrix_modules(model)):
        row = {
            "name": name,
            "longname": name,
            "alpha": 3.0,
            "D": 0.01,
            "xmin": 1.0,
            "detX_num": 16,
            "num_evals": 16,
            "spectral_estimator": "weightwatcher",
            "projected": True,
        }
        rows.append(row)
        selected = layer_selector(model, name, row) if layer_selector else True
        original = weight.detach().clone()
        # The first eligible layer converges immediately; the remaining layers
        # take an ordinary fast step.  This creates terminal and nonterminal rows
        # in the same flush, matching the failure mode seen on the MacBook.
        displacement_scale = 1e-6 if index == 0 else 1e-2
        candidate = original + displacement_scale
        originals[name] = original
        candidates[name] = candidate
        norm = max(float(torch.linalg.norm(original.float())), 1e-12)
        movement = float(torch.linalg.norm((candidate - original).float())) / norm
        relative[name] = movement
        changed[name] = bool(selected) and not torch.equal(candidate, original)
        if selected:
            diagnostics.append(
                {
                    "layer_name": name,
                    "diagnostics_mode": "compatibility",
                    "native_internal_diagnostics": False,
                    "valid_observable_diagnostic": True,
                    "unsupported_internal_fields": json.dumps(["private_fields"]),
                    "candidate_changed": changed[name],
                    "candidate_relative_frobenius_change": movement,
                }
            )

    return StockWWPGDCandidate(
        pre_projection_details=pd.DataFrame(rows),
        original_weights=originals,
        candidate_weights=candidates,
        original_to_candidate_relative_change=relative,
        stock_candidate_changed=changed,
        runtime=0.0,
        stock_config=cfg,
        internal_diagnostics=diagnostics,
        candidate_execution_device="live",
        live_model_device="cpu",
        candidate_offloaded=False,
    )


@pytest.mark.parametrize("level", [0, 1, 2])
def test_level_cached_endpoint_run_completes_end_to_end(level, monkeypatch, tmp_path: Path):
    base = load_config(Path(f"configs/level{level}_adaptive_alpha.yaml"), level)
    model_cfg = replace(base.model, block_size=8, vocab_size=64)
    train_cfg = replace(
        base.train,
        batch_size=1,
        gradient_accumulation=1,
        max_steps=6,
        eval_interval=2,
        checkpoint_interval=4,
        spectral_interval=99,
        eval_batches=1,
        training_sampling="random_window",
        evaluation_sampling="fixed_probe",
    )
    adaptive = replace(
        base.wwpgd.adaptive,
        start_step=1,
        min_observations=1,
        apply_interval=1,
        endpoint_stop_relative_distance=1e-4,
        max_endpoint_age_steps=20,
        cache_endpoint_on_cpu=False,
        refresh_at_final_step=True,
        log_every_fast_step=True,
    )
    cfg = replace(
        base,
        model=model_cfg,
        train=train_cfg,
        measurement=replace(
            base.measurement,
            alpha_interval=2,
            trap_diagnostic_interval=99,
        ),
        wwpgd=replace(base.wwpgd, enabled=True, extension="wwpgd", adaptive=adaptive),
        seeds=[1337],
        token_multipliers=[1],
    )

    monkeypatch.setattr("wwgpt.train.build_stock_wwpgd_candidate", _fake_candidate_builder)
    monkeypatch.setattr("wwgpt.train.spectral_summary", lambda *args, **kwargs: [])

    torch.manual_seed(100 + level)
    initial_model = GPT(cfg.model)
    init_state = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    init_hash = sha256_bytes(
        b"".join(init_state[name].cpu().numpy().tobytes() for name in sorted(init_state))
    )
    data = _tiny_scientific_data(
        cfg.model.vocab_size,
        cfg.train.max_steps * cfg.train.batch_size * cfg.model.block_size,
    )

    pair_root = tmp_path / f"level_{level}"
    pair_id = f"pair-level-{level}"
    baseline_cfg = replace(
        cfg,
        wwpgd=replace(cfg.wwpgd, enabled=False, extension="none"),
    )
    baseline_run = run_scientific_single(
        pair_root,
        "adamw",
        1337,
        baseline_cfg,
        data,
        pair_id,
        init_state,
        init_hash,
        level,
        1,
        device="cpu",
    )
    run = run_scientific_single(
        pair_root,
        "adamw_wwpgd",
        1337,
        cfg,
        data,
        pair_id,
        init_state,
        init_hash,
        level,
        1,
        device="cpu",
    )

    baseline_complete = json.loads((baseline_run / "run_complete.json").read_text())
    complete = json.loads((run / "run_complete.json").read_text())
    baseline_manifest = json.loads((baseline_run / "manifest.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    assert baseline_complete["step"] == 6
    assert baseline_complete["run_health_ready_for_analysis"] is True
    assert baseline_manifest["initialization_hash"] == manifest["initialization_hash"]
    assert baseline_manifest["data_hash"] == manifest["data_hash"]
    assert baseline_manifest["initial_minibatch_indices"] == manifest["initial_minibatch_indices"]
    assert complete["step"] == 6
    assert complete["optimizer_step_count"] == 6
    assert complete["completed_measurement_count"] == 3
    assert complete["endpoint_activation_count"] > 0
    assert complete["fast_layer_decision_count"] > 0
    assert complete["run_health_ready_for_analysis"] is True
    assert complete["run_health_error_count"] == 0
    assert math.isfinite(float(complete["mean_controller_gain_requested"]))
    assert math.isfinite(float(complete["mean_controller_gain_applied"]))

    with (run / "wwpgd_controller.csv").open(newline="") as handle:
        controller = list(csv.DictReader(handle))
    with (run / "wwpgd_endpoint_relaxation.csv").open(newline="") as handle:
        relaxation = list(csv.DictReader(handle))

    assert controller
    assert {row["action_type"] for row in controller} == {"slow_measurement"}
    assert relaxation
    assert {row["action_type"] for row in relaxation} == {"fast_endpoint_relaxation"}
    assert not (run / "wwpgd_projection.csv").exists()
    assert complete["projected_matrix_count"] == complete["fast_changed_layer_count"]
    assert any(
        row["converged"].lower() == "true"
        or row["invalidated"].lower() == "true"
        for row in relaxation
    )
    assert any(row["changed"].lower() == "true" for row in relaxation)
    assert all(row["controller_gain_requested"] != "" for row in relaxation)
    assert all(row["applied_relative_frobenius_change"] != "" for row in relaxation)
    assert (run / "selected_checkpoint_metrics.json").is_file()
    assert (run / "checkpoints" / "checkpoint_step_000006.pt").is_file()
    assert json.loads((run / "events.jsonl").read_text()) == {"event": "complete"}
