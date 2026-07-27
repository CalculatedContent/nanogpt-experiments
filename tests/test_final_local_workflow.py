from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from wwgpt.config import load_config
from wwgpt.model import GPT
from wwgpt.seed_analysis import _summary
from wwgpt.ww import weightwatcher_details


def test_fixed_configs_are_uniform_and_bounded() -> None:
    for level in (0, 1, 2):
        cfg = load_config(Path(f"configs/level{level}_fixed_wwpgd.yaml"), level)
        assert cfg.wwpgd.target_alpha == pytest.approx(2.0)
        assert cfg.wwpgd.adaptive.mode == "uniform"
        assert cfg.wwpgd.adaptive.max_hardness == pytest.approx(0.25)
        assert cfg.wwpgd.adaptive.max_relative_frobenius_change_per_step == pytest.approx(0.001)
        assert cfg.wwpgd.adaptive.max_cumulative_relative_frobenius_change_per_refresh == pytest.approx(0.025)


def test_seed_summary_uses_standard_uncertainty_names() -> None:
    row = _summary(pd.Series([1.0, 2.0, 3.0]).to_numpy(), seed=1)
    assert "two_sd_lower" in row
    assert "two_sd_upper" in row
    assert not any("bollinger" in key.lower() for key in row)


def test_weightwatcher_cpu_offload_metadata(monkeypatch) -> None:
    import wwgpt.ww as module

    model = GPT(load_config(Path("configs/level0_adaptive_alpha.yaml"), 0).model)
    monkeypatch.setattr(module, "_model_device", lambda candidate: torch.device("mps") if candidate is model else torch.device("cpu"))

    class DummyWatcher:
        def __init__(self, model):
            assert next(model.parameters()).device.type == "cpu"
        def analyze(self, **_kwargs):
            return pd.DataFrame([{"name": "blocks.0.attn.key", "alpha": 2.5}])

    monkeypatch.setattr("weightwatcher.WeightWatcher", DummyWatcher)
    details = weightwatcher_details(model)
    assert details.analysis_offloaded.eq(True).all()
    assert details.analysis_execution_device.eq("cpu").all()
    assert details.live_model_device.eq("mps").all()


def test_analysis_notebooks_use_canonical_modules() -> None:
    expected = {
        "02_compare_single_level.ipynb": "analyze_seed_results",
        "03_weightwatcher_analysis.ipynb": "analyze_weightwatcher_results",
        "04_scaling_laws.ipynb": "analyze_cross_level_effects",
        "05_overfitting_and_generalization.ipynb": "analyze_generalization_results",
        "07_wwpgd_diagnostics.ipynb": "analyze_wwpgd_diagnostics",
    }
    for name, symbol in expected.items():
        notebook = json.loads((Path("notebooks") / name).read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert symbol in source
        assert any(
            "parameters" in cell.get("metadata", {}).get("tags", [])
            for cell in notebook["cells"]
        )
