from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wwgpt.analysis import (
    align_curves,
    discover_experiment_runs,
    discover_pair_directories,
    load_run_artifacts,
    normalize_metrics,
    paired_curve_differences,
    select_valid_run_directory,
    terminal_results,
    add_generalization_measures,
    vocab_size_from_artifacts,
)


def _write_run(parent: Path, opt: str, seed: int, complete: bool = True, offset: float = 0.0) -> Path:
    run = parent / opt / f"run_20260101-00000{seed}_{opt}"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"optimizer": opt, "seed": seed, "pair_id": parent.name, "valid_for_science": True, "parameter_report": {"total_parameters": 10, "vocab_size": 10}, "estimated_flops": 1000, "requested_tokens": 64, "realized_tokens": 64, "initialization_hash": "abc", "tokenizer_hash": "tok", "dataset_name": "fixture"}))
    pd.DataFrame({"step": [1, 2, 3], "tokens_processed": [16, 32, 48], "elapsed_time": [1.0, 2.0, 3.0], "train_loss": [3.0, 2.8, 2.5 + offset], "val_loss": [3.1, 2.9, 2.6 + offset], "tokens_per_second": [16, 16, 16], "projection_overhead": [0.0, 0.1 if opt == "adamw_wwpgd" else 0.0, 0.0]}).to_csv(run / "metrics.csv", index=False)
    pd.DataFrame({"layer_name": ["blocks.0"], "step": [2], "alpha_before": [2.3], "alpha_after": [2.1], "relative_frobenius_change": [0.01], "projection_runtime": [0.1], "warning": [""]}).to_csv(run / "wwpgd_projection.csv", index=False)
    (run / "events.jsonl").write_text('{"event":"complete"}\n' if complete else '{"event":"started"}\n')
    if complete:
        (run / "run_complete.json").write_text(json.dumps({"step": 3, "final_val_loss": 2.6 + offset}))
    return run


def _fixture(root: Path) -> Path:
    pair = root / "pair_11_fixture"
    pair.mkdir()
    (pair / "pair_manifest.json").write_text(json.dumps({"pair_id": pair.name, "seed": 11}))
    _write_run(pair, "adamw", 11, True, 0.0)
    _write_run(pair, "adamw_wwpgd", 11, True, -0.1)
    missing = root / "pair_22_missing"
    missing.mkdir()
    _write_run(missing, "adamw", 22, True, 0.2)
    return root


def test_discovery_loading_and_missing_arm(tmp_path: Path):
    root = _fixture(tmp_path)
    assert [p.name for p in discover_pair_directories(root)] == ["pair_11_fixture", "pair_22_missing"]
    runs = discover_experiment_runs(root)
    assert len(runs) == 4
    assert any(r["optimizer"] == "adamw_wwpgd" and r["run_dir"] is None for r in runs if r["pair_id"] == "pair_22_missing")
    assert {r["seed"] for r in runs if r["seed"] is not None} == {11, 22}


def test_select_valid_completed_run(tmp_path: Path):
    pair = tmp_path / "pair_1_x"
    old = _write_run(pair, "adamw", 1, False)
    new = _write_run(pair, "adamw", 2, True)
    selected, note = select_valid_run_directory(pair / "adamw")
    assert selected == new
    assert "completed" in note
    assert load_run_artifacts(old)["metrics.csv"].shape[0] == 3


def test_metrics_projection_terminal_and_alignment(tmp_path: Path):
    root = _fixture(tmp_path)
    runs = discover_experiment_runs(root)
    art = next(r["artifacts"] for r in runs if r["optimizer"] == "adamw_wwpgd" and r["run_dir"])
    metrics = normalize_metrics(art["metrics.csv"])
    assert {"tokens_seen", "validation_loss", "elapsed_seconds", "projection_seconds"}.issubset(metrics.columns)
    assert "alpha_before" in art["wwpgd_projection.csv"].columns
    term = terminal_results(runs)
    assert "wwpgd_minus_adamw_final_validation_loss" in term.columns
    assert term.loc[term["seed"] == 11, "wwpgd_minus_adamw_final_validation_loss"].iloc[0] < 0
    adamw = normalize_metrics(next(r["artifacts"]["metrics.csv"] for r in runs if r["pair_id"] == "pair_11_fixture" and r["optimizer"] == "adamw"))
    ww = normalize_metrics(next(r["artifacts"]["metrics.csv"] for r in runs if r["pair_id"] == "pair_11_fixture" and r["optimizer"] == "adamw_wwpgd"))
    grid, vals = align_curves([adamw, ww], "tokens_seen", "validation_loss")
    assert len(grid) and vals.shape[0] == 2
    dgrid, diffs = paired_curve_differences([(adamw, ww)], "tokens_seen", "validation_loss")
    assert len(dgrid) and diffs.shape[0] == 1


def test_missing_optional_columns_are_tolerated():
    df = normalize_metrics(pd.DataFrame({"step": [1], "val_loss": [2.0]}))
    assert "validation_loss" in df.columns
    assert "tokens_seen" not in df.columns


def test_generalization_measures_add_perplexity_capacity_and_gaps(tmp_path: Path):
    root = _fixture(tmp_path)
    runs = discover_experiment_runs(root)
    art = next(r["artifacts"] for r in runs if r["pair_id"] == "pair_11_fixture" and r["optimizer"] == "adamw")
    measured = add_generalization_measures(art["metrics.csv"].drop(columns=[]), vocab_size=10)
    assert "val_perplexity" in measured.columns
    assert "val_token_prediction_capacity" in measured.columns
    assert "capacity_generalization_gap" in measured.columns
    assert measured["val_token_prediction_capacity"].notna().all()
    assert vocab_size_from_artifacts(art) == 10
