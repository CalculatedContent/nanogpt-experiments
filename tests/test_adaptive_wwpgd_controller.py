import math
import pytest
from wwgpt.adaptive_wwpgd import AdaptiveWWPGDConfig, AdaptiveWWPGDController, hardness_for_alpha, resolve_layer_config, validate_adaptive_config


def cfg(**kw):
    return {**AdaptiveWWPGDConfig(mode="alpha_linear", deadband_above_target=0.4, full_strength_alpha=4.0).__dict__, "target_alpha": 2.0, **kw}


def test_uniform_mode_reproduces_unit_hardness():
    h, _, _ = hardness_for_alpha(2.1, {**cfg(), "mode":"uniform"})
    assert h == 1.0


def test_linear_controller_fixture_values():
    vals=[(2.03,0),(2.26,0),(2.54,.0875),(2.68,.175),(2.81,.25625),(3.03,.39375),(3.24,.525),(3.61,.75625),(4.0,1)]
    for a, exp in vals:
        h,_,_=hardness_for_alpha(a, cfg(response_curve="linear"))
        assert h == pytest.approx(exp)


def test_smoothstep_monotonic_bounded():
    hs=[hardness_for_alpha(a, cfg(response_curve="smoothstep"))[0] for a in [2.0,2.4,2.8,3.2,3.6,4.0,4.4]]
    assert hs == sorted(hs)
    assert all(0 <= h <= 1 for h in hs)


def test_piecewise_interpolation_and_validation():
    c=cfg(mode="alpha_piecewise", piecewise_points=[[2,0],[3,0.5],[4,1]])
    assert hardness_for_alpha(2.5, c)[0] == pytest.approx(.25)
    assert hardness_for_alpha(1.0, c)[0] == 0
    assert hardness_for_alpha(5.0, c)[0] == 1
    with pytest.raises(ValueError): validate_adaptive_config(AdaptiveWWPGDConfig(mode="alpha_piecewise", piecewise_points=[[2,0]]), 2)
    with pytest.raises(ValueError): validate_adaptive_config(AdaptiveWWPGDConfig(mode="alpha_piecewise", piecewise_points=[[2,0],[2,1]]), 2)


def test_deadband_and_full_strength():
    assert hardness_for_alpha(2.4, cfg())[0] == 0
    assert hardness_for_alpha(4.0, cfg())[0] == 1


def test_nan_alpha_zero_hardness():
    assert hardness_for_alpha(float("nan"), cfg())[0] == 0


def test_override_precedence_matrix_glob_exact():
    ac=AdaptiveWWPGDConfig(matrix_type_overrides={"W_V":{"full_strength_alpha":3.8}}, layer_overrides={"blocks.*.attn.value":{"deadband_above_target":.6}, "blocks.0.attn.value":{"max_hardness":.25}})
    r=resolve_layer_config(ac, "blocks.0.attn.value", 2.0)
    assert r["full_strength_alpha"] == 3.8
    assert r["deadband_above_target"] == .6
    assert r["max_hardness"] == .25


def test_ema_first_observation_behavior():
    c=AdaptiveWWPGDController(AdaptiveWWPGDConfig(alpha_ema_beta=.8), 2.0)
    assert c.observe("L", 3.0) == (1, 3.0)
    n, ema=c.observe("L", 4.0)
    assert n == 2 and ema == pytest.approx(3.2)
