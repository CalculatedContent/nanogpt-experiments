import pandas as pd
import pytest
import torch
from torch import nn

from wwgpt.config import ExperimentConfig, WWPGDConfig, validate_experiment_config
from wwgpt.train import CANONICAL_TRIAL_ARMS, MeasurementOnlyExtension
from wwgpt.ww import (
    StockWWPGDCandidate,
    apply_external_wwpgd,
    resolved_external_wwpgd_config,
)


def _candidate(model: nn.Linear) -> StockWWPGDCandidate:
    original = model.weight.detach().clone()
    displacement = torch.arange(1, original.numel() + 1, dtype=original.dtype).reshape_as(original)
    cfg = resolved_external_wwpgd_config()
    return StockWWPGDCandidate(
        pd.DataFrame({"alpha": [2.5]}),
        {"layer": original},
        {"layer": original + displacement},
        {"layer": float(torch.linalg.norm(displacement) / torch.linalg.norm(original))},
        {"layer": True},
        0.0,
        cfg,
    )


def test_norm_matched_sham_is_deterministic_orthogonal_and_norm_matched(monkeypatch):
    monkeypatch.setattr("wwgpt.ww.projected_matrix_modules", lambda model: [("layer", model.weight)])
    first = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        first.weight.fill_(1.0)
    second = nn.Linear(4, 4, bias=False)
    second.load_state_dict(first.state_dict())
    original = first.weight.detach().clone()
    candidate1 = _candidate(first)
    candidate2 = _candidate(second)

    rows1 = apply_external_wwpgd(first, stock_candidate=candidate1,
        layer_hardness={"layer": 0.4}, global_event_hardness=1.0,
        layer_max_relative_change={"layer": None}, sham_seed=1234)
    rows2 = apply_external_wwpgd(second, stock_candidate=candidate2,
        layer_hardness={"layer": 0.4}, global_event_hardness=1.0,
        layer_max_relative_change={"layer": None}, sham_seed=1234)

    assert torch.equal(first.weight, second.weight)
    real = candidate1.candidate_weights["layer"] - original
    sham = first.weight.detach() - original
    assert torch.linalg.norm(sham) == pytest.approx(0.4 * torch.linalg.norm(real), rel=1e-6)
    assert abs(rows1[0]["real_candidate_displacement_cosine"]) < 1e-6
    assert rows1[0]["displacement_kind"] == "norm_matched_sham"
    assert rows2[0]["sham_seed"] == 1234


def test_measurement_only_never_generates_candidate_or_changes_weights(monkeypatch):
    model = nn.Linear(3, 3, bias=False)
    before = model.weight.detach().clone()
    calls = []
    monkeypatch.setattr("wwgpt.train.nonmutating_weightwatcher_details",
                        lambda *args, **kwargs: calls.append(1) or pd.DataFrame({"alpha": [2.0]}))
    monkeypatch.setattr("wwgpt.train.build_stock_wwpgd_candidate",
                        lambda *args, **kwargs: pytest.fail("measurement-only generated a candidate"))
    extension = MeasurementOnlyExtension(interval=2)

    assert extension.after_optimizer_step(model=model, optimizer_step=1)[0] is None
    details, projection_rows, controller_rows = extension.after_optimizer_step(model=model, optimizer_step=2)
    assert len(details) == 1
    assert projection_rows == controller_rows == []
    assert calls == [1]
    assert torch.equal(model.weight, before)


def test_controls_are_optional_and_canonical_six_remain_unchanged():
    assert CANONICAL_TRIAL_ARMS == (
        "adamw", "adamw_wwpgd", "muon", "muon_wwpgd",
        "stable_adamw", "stable_adamw_wwpgd",
    )
    for extension in ("measurement_only", "norm_matched_sham"):
        cfg = ExperimentConfig(extensions=["none", extension],
            wwpgd=WWPGDConfig(extension=extension, enabled=extension == "norm_matched_sham"))
        validate_experiment_config(cfg)


def test_delayed_onset_requires_a_predeclared_positive_step():
    with pytest.raises(ValueError, match="delayed_onset_step"):
        validate_experiment_config(ExperimentConfig(
            extensions=["delayed_onset"],
            wwpgd=WWPGDConfig(extension="delayed_onset", enabled=True),
        ))
    validate_experiment_config(ExperimentConfig(
        extensions=["delayed_onset"],
        wwpgd=WWPGDConfig(extension="delayed_onset", enabled=True, delayed_onset_step=100),
    ))


def test_delayed_onset_in_arm_list_cannot_silently_default_to_step_one():
    with pytest.raises(ValueError, match="predeclared step"):
        validate_experiment_config(ExperimentConfig(
            extensions=["none", "measurement_only", "norm_matched_sham", "delayed_onset", "wwpgd"],
            wwpgd=WWPGDConfig(extension="wwpgd", enabled=True),
        ))
