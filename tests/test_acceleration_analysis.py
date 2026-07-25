import json

import pandas as pd

from wwgpt.acceleration_analysis import analyze_acceleration_pairs, paired_auc, sustained_tokens_to_threshold


def curve(tokens, losses):
    return pd.DataFrame({"tokens_seen": tokens, "validation_loss": losses,
                         "step": range(len(tokens)), "elapsed_seconds": tokens})


def test_known_two_x_speedup_and_artifacts(tmp_path):
    plan = tmp_path / "plan.yaml"
    plan.write_text("mode: confirmatory\nthresholds: [2.0]\nprimary_outcomes: [tokens_saved]\nfixed_token_budgets: [75]\n")
    pairs = [{"seed": 7, "base_optimizer": "adamw",
              "baseline": curve([0, 100, 200], [3, 2, 1]),
              "wwpgd": curve([0, 50, 100], [3, 2, 1])}]
    analyze_acceleration_pairs(pairs, tmp_path / "out", plan)
    row = pd.read_csv(tmp_path / "out/acceleration_by_seed.csv").iloc[0]
    assert row.speedup_ratio == 2 and row.tokens_saved == 50
    assert pd.isna(row["baseline_loss_at_75_tokens"]) is False
    expected = {"acceleration_by_seed.csv", "acceleration_summary.csv", "validation_auc_by_seed.csv",
                "threshold_crossing_audit.csv", "paired_learning_curves.png", "tokens_saved_by_seed.png",
                "analysis_plan_manifest.json"}
    assert expected <= {p.name for p in (tmp_path / "out").iterdir()}
    assert json.loads((tmp_path / "out/analysis_plan_manifest.json").read_text())["analysis_mode"] == "confirmatory"


def test_identical_curves_and_unequal_grids_auc():
    a = curve([0, 50, 100], [3, 2, 1])
    b = curve([0, 25, 75, 100], [3, 2.5, 1.5, 1])
    result = paired_auc(a, b)
    assert result["common_token_start"] == 0 and result["common_token_end"] == 100
    identical = paired_auc(a, a)
    assert identical["paired_auc_difference"] == 0


def test_noisy_dip_is_not_sustained_and_not_reached_is_none():
    noisy = curve([0, 10, 20, 30], [3, 1.9, 2.2, 1.8])
    assert sustained_tokens_to_threshold(noisy, 2.0) is None
    assert sustained_tokens_to_threshold(curve([0, 10], [3, 2.5]), 2.0) is None


def test_auc_never_extrapolates_beyond_common_support():
    result = paired_auc(curve([0, 100], [3, 1]), curve([50, 200], [2, 0]))
    assert result["common_token_start"] == 50
    assert result["common_token_end"] == 100
