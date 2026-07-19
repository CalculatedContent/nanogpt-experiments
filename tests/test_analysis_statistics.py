from __future__ import annotations

import math

import pandas as pd
import pytest
from scipy import stats

from wwgpt.analysis import paired_effect_estimates, paired_extension_effects, student_t_summary


def test_student_t_summary_matches_analytic_mean_sd_and_ci():
    values = [1.0, 2.0, 3.0, 4.0]
    out = student_t_summary(values)
    mean = 2.5
    sd = math.sqrt(5.0 / 3.0)
    se = sd / 2.0
    half = stats.t.ppf(0.975, 3) * se
    assert out["n"] == 4
    assert out["mean"] == pytest.approx(mean)
    assert out["std"] == pytest.approx(sd)
    assert out["ci_low"] == pytest.approx(mean - half)
    assert out["ci_high"] == pytest.approx(mean + half)


def test_paired_effect_estimates_preserve_seeds_and_separate_bases():
    rows = []
    for seed, adamw_base, adamw_ww, muon_base, muon_ww, stable_base, stable_ww in [
        (1, 10.0, 9.0, 5.0, 7.0, 20.0, 19.5),
        (2, 12.0, 10.0, 6.0, 9.0, 18.0, 17.0),
    ]:
        rows.extend([
            {"scientific_schema_version": 3, "level": 0, "token_multiplier": 20, "base_optimizer": "adamw", "extension": "none", "seed": seed, "loss": adamw_base},
            {"scientific_schema_version": 3, "level": 0, "token_multiplier": 20, "base_optimizer": "adamw", "extension": "wwpgd", "seed": seed, "loss": adamw_ww},
            {"scientific_schema_version": 3, "level": 0, "token_multiplier": 20, "base_optimizer": "muon", "extension": "none", "seed": seed, "loss": muon_base},
            {"scientific_schema_version": 3, "level": 0, "token_multiplier": 20, "base_optimizer": "muon", "extension": "wwpgd", "seed": seed, "loss": muon_ww},
            {"scientific_schema_version": 3, "level": 0, "token_multiplier": 20, "base_optimizer": "stableadamw", "extension": "none", "seed": seed, "loss": stable_base},
            {"scientific_schema_version": 3, "level": 0, "token_multiplier": 20, "base_optimizer": "stableadamw", "extension": "wwpgd", "seed": seed, "loss": stable_ww},
        ])
    paired = paired_extension_effects(pd.DataFrame(rows), "loss")
    assert len(paired) == 6
    assert paired.groupby("base_optimizer")["seed"].nunique().to_dict() == {"adamw": 2, "muon": 2, "stableadamw": 2}
    assert paired.groupby("base_optimizer")["wwpgd_minus_none_loss"].apply(list).to_dict() == {
        "adamw": [-1.0, -2.0],
        "muon": [2.0, 3.0],
        "stableadamw": [-0.5, -1.0],
    }
    estimates = paired_effect_estimates(paired, "loss").set_index("base_optimizer")
    assert estimates.loc["adamw", "paired_effect_mean"] == pytest.approx(-1.5)
    assert estimates.loc["muon", "paired_effect_mean"] == pytest.approx(2.5)
    assert estimates.loc["stableadamw", "paired_effect_mean"] == pytest.approx(-0.75)
