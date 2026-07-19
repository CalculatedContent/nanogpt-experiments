from __future__ import annotations

import json
import math
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

from wwgpt.analysis import analyze_results, completed_runs, summary
from wwgpt.config import ModelConfig
from wwgpt.data import NonRepeatingTokenReader, prepare_local_text, prepare_scientific_data, split_for_doc
from wwgpt.model import GPT
from wwgpt.scaling import is_non_collinear, plan_budget
from wwgpt.train import smoke
from wwgpt.utils import unique_dir
from wwgpt.ww import apply_wwpgd, fallback_spectral_summary, is_projected_layer, spectral_summary, weightwatcher_details, matrix_modules


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
    runs = completed_runs(root, scientific_only=False)
    assert len(runs) == 2
    assert completed_runs(root) == []
    assert any((r / "wwpgd_projection.csv").exists() for r in runs)
    out = analyze_results(root)
    assert (out / "runs_manifest.csv").exists()
    assert not (out / "final_metrics_errorbars.csv").exists()


def test_smoke_runs_all_requested_seeds(tmp_path: Path, capsys):
    root = smoke(tmp_path, 1, [11, 22, 33])
    stderr = capsys.readouterr().err
    assert "[wwgpt run-multiseed] starting smoke run" in stderr
    assert "smoke progress optimizer=" in stderr
    assert "completed smoke run" in stderr
    runs = completed_runs(root, scientific_only=False)
    assert len(runs) == 6
    manifests = [json.loads((r / "manifest.json").read_text()) for r in runs]
    assert sorted({m["seed"] for m in manifests}) == [11, 22, 33]
    assert all(m["pair_id"] == f"pair_invalid_seed_{m['seed']}" for m in manifests)


def test_prepare_scientific_data_logs_progress(tmp_path: Path, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
model:
  n_layer: 1
  n_head: 1
  n_embd: 8
  block_size: 8
  vocab_size: 64
train:
  batch_size: 2
  gradient_accumulation: 1
""")
    docs = []
    i = 0
    while sum(1 for d in docs if split_for_doc(d) == "train") < 350 or sum(1 for d in docs if split_for_doc(d) == "val") < 10:
        docs.append(f"scientific logging fixture document {i} " + ("abc xyz " * 30))
        i += 1
    prepared = prepare_scientific_data(tmp_path, 0, 1, cfg, docs, min_validation_tokens=1)
    stderr = capsys.readouterr().err
    assert prepared.root is not None
    assert "[wwgpt prepare-data] starting" in stderr
    assert "training BPE tokenizer" in stderr
    assert "wrote train_tokens.npy" in stderr


def test_notebooks_parse():
    for path in Path("notebooks").glob("*.ipynb"):
        nbformat.read(path, as_version=4)


def test_projected_layer_selector_excludes_embeddings_and_lm_head():
    m = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=10, tie_weights=True))
    eligible = [n for n, _ in matrix_modules(m) if is_projected_layer(n)]
    assert eligible
    assert all(not n.startswith(("wte", "wpe")) and n != "lm_head" for n in eligible)


def test_fallback_spectral_marked_non_scientific():
    m = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=10))
    rows = fallback_spectral_summary(m)
    assert rows and all(r["spectral_estimator"] == "fallback_non_scientific" and r["valid_for_science"] is False for r in rows)


def test_validation_probe_fixed_and_distinct():
    from wwgpt.data import fixed_probe
    tx, ty, th = fixed_probe(list(range(100)), 4, 2, 2)
    vx, vy, vh = fixed_probe(list(range(1000,1100)), 4, 2, 2)
    vx2, vy2, vh2 = fixed_probe(list(range(1000,1100)), 4, 2, 2)
    assert vh == vh2 and (vx == vx2).all()
    assert th != vh


def test_legacy_estimator_formula_invalid_regression():
    exp = 2.5
    ranks = np.arange(1, 100)
    eig = ranks ** (-exp)
    slope = np.polyfit(np.log(ranks), np.log(eig), 1)[0]
    old_alpha = 1 - slope
    assert abs(old_alpha - exp) > 0.25


def test_scientific_evaluation_uses_all_probe_batches():
    import inspect
    import wwgpt.train

    source = inspect.getsource(wwgpt.train.run_scientific_single)
    assert "val_x[0]" not in source
    assert "train_x[0]" not in source
    assert source.count("_evaluate_probe_batches(") == 2


def test_streaming_metrics_match_full_logit_reference():
    from wwgpt.train import _evaluate_probe_batches, _metrics
    cfg = ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=11)
    torch.manual_seed(123)
    m = GPT(cfg)
    x = np.array([[[0,1,2,3],[4,5,6,7]], [[1,2,3,4],[5,6,7,8]]], dtype=np.int64)
    y = (x + 1) % cfg.vocab_size
    with torch.no_grad():
        streamed, sloss = _evaluate_probe_batches(m, x, y, torch.device("cpu"))
        logits = []
        targets = []
        losses = []
        for bx, by in zip(x, y, strict=True):
            lg, loss = m(torch.tensor(bx), torch.tensor(by))
            losses.append(float(loss) * by.size)
            logits.append(lg)
            targets.append(torch.tensor(by))
        ref_loss = sum(losses) / y.size
        ref = _metrics(ref_loss, torch.cat(logits, dim=0), torch.cat(targets, dim=0))
    assert sloss == pytest.approx(ref_loss)
    for k in ["loss", "perplexity", "top1_accuracy", "top5_accuracy", "token_error"]:
        assert streamed[k] == pytest.approx(ref[k])


def test_evaluation_does_not_retain_full_logit_lists():
    import inspect
    import wwgpt.train
    source = inspect.getsource(wwgpt.train._evaluate_probe_batches)
    assert "logits_batches" not in source
    assert "target_batches" not in source
    assert "torch.cat" not in source


def test_diagnostics_do_not_change_later_training_batches():
    from wwgpt.data import RandomWindowTokenReader, stable_seed
    from wwgpt.ww import fallback_spectral_summary
    seed = stable_seed(3, "pair", "train_reader_v1")
    tokens = list(range(500))
    ref = RandomWindowTokenReader(tokens, 8, seed)
    diag = RandomWindowTokenReader(tokens, 8, seed)
    first = ref.next_batch(4); diag.next_batch(4)
    torch.manual_seed(5); m = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=20))
    fallback_spectral_summary(m)
    assert all((a == b).all() for a, b in zip(ref.next_batch(4), diag.next_batch(4), strict=True))
