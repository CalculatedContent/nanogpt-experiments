from __future__ import annotations

import sys
import types

import pandas as pd

from wwgpt.config import ModelConfig
from wwgpt.model import GPT
from wwgpt.ww import add_weightwatcher_diagnostic_fields, spectral_summary, weightwatcher_run_aggregates


def tiny_model():
    return GPT(ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=16))


def test_mocked_weightwatcher_diagnostics_request_randomized_traps(monkeypatch):
    captured = {}

    class FakeWatcher:
        def __init__(self, model):
            captured["model"] = model

        def analyze(self, **kwargs):
            captured["kwargs"] = kwargs
            return pd.DataFrame(
                {
                    "layer_id": [7],
                    "name": ["key"],
                    "longname": ["blocks.0.attn.key"],
                    "M": [8],
                    "N": [8],
                    "alpha": [2.25],
                    "spectral_norm": [1.5],
                    "stable_rank": [3.0],
                    "rand_num_spikes": [2],
                }
            )

    fake = types.ModuleType("weightwatcher")
    fake.WeightWatcher = FakeWatcher
    monkeypatch.setitem(sys.modules, "weightwatcher", fake)

    rows = spectral_summary(tiny_model(), step=4, tokens_seen=128, optimizer="adamw", seed=11, pair_id="p")

    assert captured["kwargs"] == {"detX": True, "randomize": True, "plot": False}
    row = rows[0]
    assert row["layer_id"] == 7
    assert row["matrix_shape"] == "[8, 8]"
    assert row["alpha"] == 2.25
    assert row["spectral_norm"] == 1.5
    assert row["stable_rank"] == 3.0
    assert row["rand_num_spikes"] == 2
    assert row["trap_flag"] is True
    assert "rand_num_spikes > 0" in row["trap_rule"]


def test_mocked_weightwatcher_unsupported_fields_are_null_with_explanation():
    df = add_weightwatcher_diagnostic_fields(pd.DataFrame({"layer_id": [1], "name": ["linear"], "alpha": [2.0]}))
    row = df.iloc[0]
    assert pd.isna(row["spectral_norm"])
    assert pd.isna(row["stable_rank"])
    assert pd.isna(row["rand_num_spikes"])
    assert pd.isna(row["trap_flag"])
    assert "not returned by installed WeightWatcher" in row["unsupported_field_explanation"]


def test_weightwatcher_run_aggregates_from_long_form_rows():
    rows = [
        {"step": 1, "tokens_seen": 8, "optimizer": "adamw", "seed": 1, "pair_id": "p", "alpha": 2.0, "spectral_norm": 4.0, "stable_rank": 2.0, "trap_flag": False},
        {"step": 1, "tokens_seen": 8, "optimizer": "adamw", "seed": 1, "pair_id": "p", "alpha": 3.0, "spectral_norm": 6.0, "stable_rank": 4.0, "trap_flag": True},
    ]
    [agg] = weightwatcher_run_aggregates(rows)
    assert agg["eligible_layer_count"] == 2
    assert agg["mean_alpha"] == 2.5
    assert agg["mean_spectral_norm"] == 5.0
    assert agg["mean_stable_rank"] == 3.0
    assert agg["trap_layer_count"] == 1
    assert agg["trap_layer_fraction"] == 0.5


def test_tiny_real_weightwatcher_integration_has_diagnostic_columns():
    rows = spectral_summary(tiny_model(), step=1, tokens_seen=32, optimizer="adamw", seed=3, pair_id="real")
    assert rows
    required = {"layer_id", "longname", "matrix_shape", "alpha", "spectral_norm", "trap_flag", "trap_rule", "weightwatcher_version"}
    assert required.issubset(rows[0])
    assert rows[0]["weightwatcher_version"]
