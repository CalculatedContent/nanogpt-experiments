"""Focused regression tests for cached endpoint relaxation artifacts and math."""
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from wwgpt.adaptive_wwpgd import AdaptiveWWPGDConfig, CachedLayerEndpoint
from wwgpt.config import WWPGDConfig
from wwgpt.train import WWPGDExtension, write_csv_union_schema


def _endpoint(weight, *, distance=1.0, step=0):
    target = weight.detach().clone() + distance
    initial = float(torch.linalg.norm(target - weight.detach()) / torch.linalg.norm(weight.detach()))
    return CachedLayerEndpoint("layer", target, weight.detach().clone(), step, 0, 3.0, 3.0,
        2.0, 1.0, 1.0, "above_target", 1.0, 1.0, 1.0, initial, initial,
        initial, 0.01, 1.0, 10.0, 10.0)


def _extension(**kwargs):
    adaptive = AdaptiveWWPGDConfig(apply_mode="cached_endpoint_relaxation",
        measurement_source="explicit_interval", measurement_interval=100, max_per_step_gain=0.1,
        dose_schedule="fixed_per_step_gain",
        max_cumulative_relative_frobenius_change_per_refresh=1.0,
        max_endpoint_age_steps=100, endpoint_stop_relative_distance=0.0, **kwargs)
    return WWPGDExtension(WWPGDConfig(adaptive=adaptive), interval=100)


def test_zero_change_projection_artifact_is_readable(tmp_path):
    path = tmp_path / "wwpgd_projection.csv"
    write_csv_union_schema(path, [], empty_fields=["optimizer_step", "layer_name", "changed"])
    assert list(pd.read_csv(path).columns) == ["optimizer_step", "layer_name", "changed"]


@pytest.mark.parametrize("artifact", ["measurement", "relaxation", "controller"])
def test_cached_union_schema_csv_is_readable(tmp_path, artifact):
    path = tmp_path / f"{artifact}.csv"
    write_csv_union_schema(path, [], empty_fields=["optimizer_step", "layer_name", "action_type"])
    assert pd.read_csv(path).empty


def test_gain_point_one_applies_ten_percent_and_residual_contracts(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.data.fill_(1.0)
    ext = _extension()
    ext.endpoint_cache["layer"] = _endpoint(model.weight)
    monkeypatch.setattr("wwgpt.ww.projected_matrix_modules", lambda _model: [("layer", model.weight)])
    before = ext.endpoint_cache["layer"].endpoint_tensor - model.weight.detach().clone()
    result = ext.after_optimizer_step_fast(model=model, optimizer_step=1, total_optimizer_steps=10)
    after = ext.endpoint_cache["layer"].endpoint_tensor - model.weight.detach()
    assert result.changed_layer_count == 1
    assert torch.linalg.norm(after) == pytest.approx(0.9 * float(torch.linalg.norm(before)))
    assert result.changed_rows[0]["controller_gain_applied"] == pytest.approx(0.1)


def test_fast_step_before_endpoint_and_measurement_step_are_logged(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    ext = _extension(skip_fast_apply_on_measurement_step=True)
    monkeypatch.setattr("wwgpt.ww.projected_matrix_modules", lambda _model: [])
    ext.after_optimizer_step_fast(model=model, optimizer_step=1, total_optimizer_steps=100,
                                  measurement_interval=10)
    ext.after_optimizer_step_fast(model=model, optimizer_step=10, total_optimizer_steps=100,
                                  measurement_interval=10)
    assert ext.fast_step_rows == [
        {"optimizer_step": 1, "measurement_step": False, "active_endpoint_count": 0,
         "changed_layer_count": 0, "any_change": False, "skip_reason": "no_active_endpoint"},
        {"optimizer_step": 10, "measurement_step": True, "active_endpoint_count": 0,
         "changed_layer_count": 0, "any_change": False, "skip_reason": "measurement_step"},
    ]


def test_applied_update_decreases_as_residual_decreases(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False); model.weight.data.fill_(1.0)
    ext = _extension()
    ext.endpoint_cache["layer"] = _endpoint(model.weight)
    monkeypatch.setattr("wwgpt.ww.projected_matrix_modules", lambda _model: [("layer", model.weight)])
    first = ext.after_optimizer_step_fast(model=model, optimizer_step=1, total_optimizer_steps=10).changed_rows[0]
    second = ext.after_optimizer_step_fast(model=model, optimizer_step=2, total_optimizer_steps=10).changed_rows[0]
    assert second["applied_relative_frobenius_change"] < first["applied_relative_frobenius_change"]


def test_trust_region_clipping_is_enforced(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False); model.weight.data.fill_(1.0)
    ext = _extension(max_relative_frobenius_change_per_step=0.001)
    ext.endpoint_cache["layer"] = _endpoint(model.weight)
    monkeypatch.setattr("wwgpt.ww.projected_matrix_modules", lambda _model: [("layer", model.weight)])
    row = ext.after_optimizer_step_fast(model=model, optimizer_step=1, total_optimizer_steps=10).changed_rows[0]
    assert row["applied_relative_frobenius_change"] <= 0.001000001
    assert row["trust_region_scale"] < 1.0


@pytest.mark.parametrize("level,layers,heads,width", [(0,1,1,64), (1,2,2,128), (2,4,3,192)])
def test_level_config_dimensions(level, layers, heads, width):
    cfg = yaml.safe_load(Path(f"configs/level{level}_adaptive_alpha.yaml").read_text())
    assert (cfg["model"]["n_layer"], cfg["model"]["n_head"], cfg["model"]["n_embd"]) == (layers, heads, width)
