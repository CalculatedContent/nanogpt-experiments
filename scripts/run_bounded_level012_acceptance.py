#!/usr/bin/env python3
"""Generate deterministic bounded Level 0-2 paired runs for release acceptance.

This is an offline infrastructure test, not a scientific experiment. It exercises
real training, cached-endpoint routing, finalization, checkpoints, and the exact
schema-v3 result layout without downloading FineWeb or depending on stochastic
WeightWatcher fits. Package-level WWPGD and WeightWatcher readiness is tested
separately by ``wwgpt local-readiness``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from wwgpt.acceleration_analysis import plan_manifest
from wwgpt.config import load_config
from wwgpt.data import TokenData
from wwgpt.model import GPT
from wwgpt.train import run_scientific_single
from wwgpt.utils import sha256_bytes
from wwgpt.ww import StockWWPGDCandidate, projected_matrix_modules


def _tiny_data(vocab_size: int, realized_tokens: int) -> TokenData:
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
            "dataset_name": "offline-level-release-acceptance",
            "dataset_config": "fixture",
            "dataset_revision": "fixture-v1",
            "realized_tokens": realized_tokens,
            "validation_tokens": len(val),
            "test_tokens": len(test),
            "validation_document_count": 1,
            "test_document_count": 1,
        },
        tokenizer_manifest={
            "tokenizer_hash": sha256_bytes(b"offline-level-release-tokenizer"),
            "tokenizer_type": "offline",
            "vocab_size": vocab_size,
        },
    )


def _candidate_builder(
    model,
    *,
    event_index=0,
    actual_step=0,
    cfg=None,
    layer_selector=None,
    **_kwargs,
):
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
        # One terminal row plus ordinary changed rows covers both schemas in the
        # same flush, matching the Mac failures fixed by PRs #104-#106.
        displacement = 1e-6 if index == 0 else 1e-2
        candidate = original + displacement
        norm = max(float(torch.linalg.norm(original.float())), 1e-12)
        movement = float(torch.linalg.norm((candidate - original).float())) / norm
        originals[name] = original
        candidates[name] = candidate
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


def _run_level(results_root: Path, plan_path: Path, level: int) -> list[Path]:
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
    torch.manual_seed(100 + level)
    initial_model = GPT(cfg.model)
    init_state = {
        name: value.detach().clone()
        for name, value in initial_model.state_dict().items()
    }
    init_hash = sha256_bytes(
        b"".join(init_state[name].cpu().numpy().tobytes() for name in sorted(init_state))
    )
    data = _tiny_data(cfg.model.vocab_size, 48)
    pair_id = "pair_1337"
    pair_root = (
        results_root
        / "experiments"
        / f"level_{level:02d}"
        / "multiplier_1"
        / pair_id
    )
    pair_root.mkdir(parents=True, exist_ok=False)
    frozen_plan = plan_manifest(plan_path)
    (pair_root / "pair_manifest.json").write_text(
        json.dumps(
            {
                "pair_id": pair_id,
                "seed": 1337,
                "level": level,
                "token_multiplier": 1,
                "initialization_hash": init_hash,
                "base_optimizer": "adamw",
                "extensions": ["none", "wwpgd"],
                "arms": ["adamw", "adamw_wwpgd"],
                **frozen_plan,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    baseline_cfg = replace(
        cfg,
        wwpgd=replace(cfg.wwpgd, enabled=False, extension="none"),
    )
    with patch("wwgpt.train.build_stock_wwpgd_candidate", _candidate_builder), patch(
        "wwgpt.train.spectral_summary", lambda *args, **kwargs: []
    ):
        baseline = run_scientific_single(
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
            analysis_plan_manifest=frozen_plan,
        )
        wwpgd = run_scientific_single(
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
            analysis_plan_manifest=frozen_plan,
        )
    return [baseline, wwpgd]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()
    root = args.results_root.resolve()
    if root.exists() and not args.keep_existing:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = root / "release_acceptance_plan.yaml"
    plan.write_text("mode: exploratory\nprimary_outcomes: []\n")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    runs = []
    for level in (0, 1, 2):
        runs.extend(_run_level(root, plan, level))
    complete = sorted(root.rglob("run_complete.json"))
    if len(complete) != 6:
        raise RuntimeError(f"expected six complete runs, found {len(complete)}")
    print(
        json.dumps(
            {
                "results_root": str(root),
                "levels": [0, 1, 2],
                "completed_runs": len(complete),
                "run_dirs": [str(path) for path in runs],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
