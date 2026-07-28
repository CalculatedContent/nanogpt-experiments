from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wwgpt.data import TokenData
from wwgpt.ww import StockWWPGDCandidate, projected_matrix_modules
from wwgpt.acceleration_analysis import plan_manifest
from wwgpt.analysis import analyze_results, discover_canonical_runs
from wwgpt.config import load_config
from wwgpt.model import GPT
from wwgpt.reproducibility import write_reproducibility_report
from wwgpt.run_health import generate_experiment_health
from wwgpt.train import run_scientific_single
from wwgpt.utils import sha256_bytes, write_json


def _tiny_scientific_data(vocab_size: int, realized_tokens: int) -> TokenData:
    train = np.arange(512, dtype=np.int64) % vocab_size
    val = (np.arange(128, dtype=np.int64) + 7) % vocab_size
    test = (np.arange(128, dtype=np.int64) + 13) % vocab_size
    return TokenData(
        train=train, val=val, test=test, vocab_size=vocab_size,
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


def _fake_candidate_builder(
    model, *, event_index=0, actual_step=0, cfg=None, layer_selector=None, **_kwargs
):
    rows = []
    originals = {}
    candidates = {}
    relative = {}
    changed = {}
    diagnostics = []
    for index, (name, weight) in enumerate(projected_matrix_modules(model)):
        row = {
            "name": name, "longname": name, "alpha": 3.0, "D": 0.01,
            "xmin": 1.0, "detX_num": 16, "num_evals": 16,
            "spectral_estimator": "weightwatcher", "projected": True,
        }
        rows.append(row)
        selected = layer_selector(model, name, row) if layer_selector else True
        original = weight.detach().clone()
        candidate = original + (1e-6 if index == 0 else 1e-2)
        originals[name] = original
        candidates[name] = candidate
        norm = max(float(torch.linalg.norm(original.float())), 1e-12)
        movement = float(torch.linalg.norm((candidate - original).float())) / norm
        relative[name] = movement
        changed[name] = bool(selected) and not torch.equal(candidate, original)
        if selected:
            diagnostics.append({
                "layer_name": name, "diagnostics_mode": "compatibility",
                "native_internal_diagnostics": False,
                "valid_observable_diagnostic": True,
                "unsupported_internal_fields": json.dumps(["private_fields"]),
                "candidate_changed": changed[name],
                "candidate_relative_frobenius_change": movement,
            })
    return StockWWPGDCandidate(
        pre_projection_details=pd.DataFrame(rows),
        original_weights=originals, candidate_weights=candidates,
        original_to_candidate_relative_change=relative,
        stock_candidate_changed=changed, runtime=0.0, stock_config=cfg,
        internal_diagnostics=diagnostics, candidate_execution_device="live",
        live_model_device="cpu", candidate_offloaded=False,
    )


def _level_config(level: int):
    base = load_config(Path(f"configs/level{level}_adaptive_alpha.yaml"), level)
    # Exercise the Level 0/1/2 orchestration identities with one bounded model.
    # The real level-specific architectures are covered by the runtime/config tests;
    # retaining the Level 2 width here makes this post-processing acceptance test
    # unnecessarily slow without adding a new code path.
    model = replace(
        base.model, n_layer=1, n_head=1, n_embd=32, block_size=8, vocab_size=64
    )
    train = replace(
        base.train,
        batch_size=1,
        gradient_accumulation=1,
        max_steps=4,
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
    return replace(
        base,
        model=model,
        train=train,
        measurement=replace(
            base.measurement,
            alpha_interval=2,
            trap_diagnostic_interval=99,
        ),
        wwpgd=replace(
            base.wwpgd,
            enabled=True,
            extension="wwpgd",
            adaptive=adaptive,
        ),
        seeds=[1337],
        token_multipliers=[20],
    )


def test_level012_training_and_all_postprocessing_complete(monkeypatch):
    # Use an independent temporary root. Pytest prunes its numbered tmp roots at
    # session boundaries, which can race with this deliberately longer acceptance
    # test when multiple audit invocations overlap.
    workspace = Path(tempfile.mkdtemp(prefix="wwgpt-level012-pipeline-"))
    results = workspace / "results"
    plan = workspace / "plan.yaml"
    plan.write_text("mode: exploratory\nprimary_outcomes: [paired_validation_loss_auc]\n")
    frozen_plan = plan_manifest(plan)

    monkeypatch.setattr(
        "wwgpt.train.build_stock_wwpgd_candidate", _fake_candidate_builder
    )
    monkeypatch.setattr("wwgpt.train.spectral_summary", lambda *args, **kwargs: [])
    # The full plotting modules have their own notebook and unit coverage. Keep
    # this acceptance focused on the command pipeline, canonical discovery,
    # acceleration schema, integrity, and report finalization rather than
    # generating hundreds of redundant PNGs for six tiny models.
    monkeypatch.setattr(
        "wwgpt.cross_level_analysis.analyze_cross_level_effects",
        lambda _root, output_dir, **_kwargs: Path(output_dir),
    )
    monkeypatch.setattr(
        "wwgpt.seed_analysis.analyze_seed_results",
        lambda _root, output_dir, **_kwargs: Path(output_dir),
    )
    monkeypatch.setattr(
        "wwgpt.generalization_analysis.analyze_generalization_results",
        lambda _root, output_dir, **_kwargs: Path(output_dir),
    )
    monkeypatch.setattr(
        "wwgpt.weightwatcher_analysis.analyze_weightwatcher_results",
        lambda _root, output_dir, **_kwargs: Path(output_dir),
    )
    monkeypatch.setattr(
        "wwgpt.wwpgd_diagnostics_analysis.analyze_wwpgd_diagnostics",
        lambda _root, output_dir, **_kwargs: Path(output_dir),
    )

    for level in (0, 1, 2):
        cfg = _level_config(level)
        torch.manual_seed(100 + level)
        initial_model = GPT(cfg.model)
        initial_state = {
            name: value.detach().clone()
            for name, value in initial_model.state_dict().items()
        }
        initialization_hash = sha256_bytes(
            b"".join(
                initial_state[name].cpu().numpy().tobytes()
                for name in sorted(initial_state)
            )
        )
        data = _tiny_scientific_data(
            cfg.model.vocab_size,
            cfg.train.max_steps * cfg.train.batch_size * cfg.model.block_size,
        )
        pair_id = f"pair_1337_level{level}"
        pair = (
            results
            / "experiments"
            / f"level_{level:02d}"
            / "multiplier_20"
            / pair_id
        )
        pair.mkdir(parents=True)
        write_json(
            pair / "pair_manifest.json",
            {
                "pair_id": pair_id,
                "seed": 1337,
                "level": level,
                "token_multiplier": 20,
                "base_optimizer": "adamw",
                "initialization_hash": initialization_hash,
                "extensions": ["none", "wwpgd"],
                **frozen_plan,
            },
        )
        baseline = replace(
            cfg,
            wwpgd=replace(cfg.wwpgd, enabled=False, extension="none"),
        )
        for arm, arm_config in (("adamw", baseline), ("adamw_wwpgd", cfg)):
            run_scientific_single(
                pair,
                arm,
                1337,
                arm_config,
                data,
                pair_id,
                initial_state,
                initialization_hash,
                level,
                20,
                device="cpu",
                analysis_plan_manifest=frozen_plan,
            )

    health = generate_experiment_health(results)
    assert health["ready_for_analysis"] is True
    assert health["run_count"] == 6

    report = write_reproducibility_report(
        results,
        strict=True,
        analysis_plan=plan,
    )
    repeated_report = write_reproducibility_report(
        results,
        strict=True,
        analysis_plan=plan,
    )
    assert report == repeated_report
    assert repeated_report.is_file()

    runs = discover_canonical_runs(results, include_legacy=True)
    assert len(runs) == 6
    assert {(row["level"], row["extension"]) for row in runs} == {
        (level, extension)
        for level in (0, 1, 2)
        for extension in ("none", "wwpgd")
    }

    analysis = results / "analysis"
    inventory = pd.read_csv(analysis / "runs_manifest.csv")
    acceleration = pd.read_csv(analysis / "acceleration_by_seed.csv")
    alpha = pd.read_csv(analysis / "alpha_summary_by_step.csv")
    integrity = json.loads((analysis / "integrity_summary.json").read_text())
    reproducibility = json.loads(
        (analysis / "reproducibility_report.json").read_text()
    )
    assert len(inventory) == 6
    assert len(acceleration) == 3
    assert set(acceleration.level) == {0, 1, 2}
    assert len(alpha) == 12
    assert set(alpha.level) == {0, 1, 2}
    assert integrity["valid_for_publication"] is True
    assert reproducibility["run_count"] == 6

    # A single Level 1 layout must also be independently analyzable. The old
    # resolver was hard-coded to level_00/multiplier_20 and returned no pairs.
    level_one = results / "experiments" / "level_01" / "multiplier_20"
    level_one_analysis = analyze_results(level_one, plan)
    assert len(pd.read_csv(level_one_analysis / "runs_manifest.csv")) == 2
