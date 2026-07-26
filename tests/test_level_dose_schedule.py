from pathlib import Path

import pytest

from wwgpt.adaptive_wwpgd import (
    effective_base_gain,
    hardness_for_alpha,
    validate_adaptive_level_schedule,
)
from wwgpt.config import load_config
from wwgpt.model import GPT
from wwgpt.scaling import plan_budget, selected_parameter_count


@pytest.mark.parametrize(
    ("level", "interval", "first_active", "expected_gain"),
    [
        (0, 25, 75, 0.02),
        (1, 250, 500, 0.002049),
        (2, 1000, 3000, 0.000511),
    ],
)
def test_real_level_configs_have_bounded_comparable_dose(
    level: int,
    interval: int,
    first_active: int,
    expected_gain: float,
) -> None:
    config_path = Path(f"configs/level{level}_adaptive_alpha.yaml")
    cfg = load_config(config_path, level)
    report = GPT(cfg.model).parameter_report()
    count = selected_parameter_count(report, cfg.parameter_count_convention)
    budget = plan_budget(
        count,
        20,
        cfg.train.batch_size,
        cfg.model.block_size,
        cfg.train.gradient_accumulation,
        10**18,
    )

    schedule = validate_adaptive_level_schedule(
        cfg.wwpgd.adaptive,
        budget.steps,
        cfg.measurement.alpha_interval,
    )

    assert cfg.wwpgd.target_alpha == 2.0
    assert cfg.measurement.alpha_interval == interval
    assert schedule["first_possible_active_endpoint_step"] == first_active
    assert schedule["expected_endpoint_opportunities"] > 0
    assert schedule["effective_base_gain"] == pytest.approx(expected_gain, rel=5e-3)
    assert schedule["worst_case_endpoint_fraction_per_refresh"] <= pytest.approx(
        cfg.wwpgd.adaptive.max_endpoint_fraction_per_refresh,
        abs=1e-12,
    )
    assert schedule["cumulative_refresh_cap"] == pytest.approx(0.025)
    assert schedule["per_step_frobenius_cap"] == pytest.approx(0.001)


def test_effective_gain_decreases_with_longer_refresh_cadence() -> None:
    configs = [load_config(Path(f"configs/level{level}_adaptive_alpha.yaml"), level) for level in range(3)]
    gains = [
        effective_base_gain(cfg.wwpgd.adaptive, cfg.measurement.alpha_interval)
        for cfg in configs
    ]
    assert gains[0] > gains[1] > gains[2] > 0


def test_uniform_mode_respects_fixed_global_hardness() -> None:
    hardness, normalized, target = hardness_for_alpha(
        100.0,
        {"mode": "uniform", "max_hardness": 0.25, "target_alpha": 2.0},
    )
    assert hardness == pytest.approx(0.25)
    assert normalized != normalized  # NaN: alpha distance is irrelevant in uniform mode.
    assert target == 2.0


def test_alpha_distance_hardness_decreases_toward_target() -> None:
    cfg = load_config(Path("configs/level0_adaptive_alpha.yaml"), 0).wwpgd.adaptive
    layer_cfg = {
        **cfg.__dict__,
        "target_alpha": 2.0,
        "above_target": cfg.above_target.__dict__,
        "below_target": cfg.below_target.__dict__,
    }
    far = hardness_for_alpha(4.0, layer_cfg)[0]
    near = hardness_for_alpha(2.6, layer_cfg)[0]
    inside = hardness_for_alpha(2.3, layer_cfg)[0]
    assert far > near > inside == 0.0
