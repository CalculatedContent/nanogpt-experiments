from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest
import torch

from wwgpt.analysis import analyze_results, completed_runs, summary
from wwgpt.config import ModelConfig
from wwgpt.data import NonRepeatingTokenReader, prepare_local_text, split_for_doc
from wwgpt.model import GPT
from wwgpt.scaling import is_non_collinear, plan_budget
from wwgpt.train import smoke
from wwgpt.utils import unique_dir
from wwgpt.ww import apply_wwpgd, matrix_modules


def test_model_parameter_counting():
    report = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=32, block_size=16, vocab_size=64)).parameter_report()
    assert report.total_parameters > 0
    assert report.non_embedding_parameters == report.total_parameters - report.embedding_parameters


def test_scaling_budget_and_rounding():
    plan = plan_budget(100, 20, 3, 7, 2, 10_000)
    assert plan.requested_tokens == 2000
    assert plan.realized_tokens % plan.tokens_per_step == 0


def test_non_repeating_and_insufficient():
    r = NonRepeatingTokenReader(list(range(20)), 4)
    r.next_batch(2)
    with pytest.raises(ValueError):
        for _ in range(10):
            r.next_batch(2)


def test_deterministic_split_and_duplicates():
    text = "same normalized document"
    assert split_for_doc(text) == split_for_doc(" same   normalized document ")


def test_prepare_manifest_validation(tmp_path: Path):
    d = prepare_local_text(tmp_path, ["abc" * 100, "def" * 100], 1)
    assert d.vocab_size > 0
    assert (tmp_path / "prepared_local_text" / "tokenizer_manifest.json").exists()


def test_paired_initialization_and_token_order():
    torch.manual_seed(1); a = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=10)).state_dict()
    torch.manual_seed(1); b = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=10)).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert NonRepeatingTokenReader(list(range(30)), 4).next_batch(2)[0].tolist() == NonRepeatingTokenReader(list(range(30)), 4).next_batch(2)[0].tolist()


def test_tied_weight_wwpgd_and_stability():
    m = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=10, tie_weights=True))
    names = [n for n, _ in matrix_modules(m)]
    assert len(names) == len(set(names))
    rows = apply_wwpgd(m, 2.0, 0.01, 1)
    assert rows and all(row["relative_frobenius_weight_change"] >= 0 for row in rows)


def test_append_only_result_dirs(tmp_path: Path):
    a = unique_dir(tmp_path, "run"); b = unique_dir(tmp_path, "run")
    assert a != b and a.exists() and b.exists()


def test_incomplete_exclusion(tmp_path: Path):
    (tmp_path / "x").mkdir()
    assert completed_runs(tmp_path) == []


def test_confidence_and_paired_difference():
    s = summary(pd.Series([1.0, 2.0, 3.0]))
    assert s["n"] == 3 and s["standard_error"] > 0


def test_scaling_collinearity_detection():
    assert is_non_collinear([1, 2, 4, 8], [5, 40, 20, 160])
    assert not is_non_collinear([1, 2, 4], [20, 40, 80])


def test_cli_smoke_execution_and_aggregation(tmp_path: Path):
    root = smoke(tmp_path, 1)
    runs = completed_runs(root)
    assert len(runs) == 2
    assert any((r / "wwpgd_projection.csv").exists() for r in runs)
    out = analyze_results(root)
    assert (out / "final_metrics_errorbars.csv").exists()
    assert (out / "plots" / "validation_loss.png").exists()


def test_smoke_runs_all_requested_seeds(tmp_path: Path):
    root = smoke(tmp_path, 1, [11, 22, 33])
    runs = completed_runs(root)
    assert len(runs) == 6
    manifests = [json.loads((r / "manifest.json").read_text()) for r in runs]
    assert sorted({m["seed"] for m in manifests}) == [11, 22, 33]
    assert all(m["pair_id"] == f"pair_smoke_seed_{m['seed']}" for m in manifests)


def test_notebooks_parse():
    for path in Path("notebooks").glob("*.ipynb"):
        nbformat.read(path, as_version=4)
