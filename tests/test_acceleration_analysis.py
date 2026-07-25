import json

import pandas as pd
import pytest

from wwgpt.acceleration_analysis import (
    _backward_validation_join,
    analyze_acceleration_pairs,
    analyze_paired_alpha_validation,
    paired_auc,
    sustained_tokens_to_threshold,
)


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


def test_alpha_alignment_uses_same_or_immediately_preceding_validation_event():
    alpha = pd.DataFrame({"optimizer_step": [10, 15, 20], "median_absolute_alpha_error": [.4, .3, .2]})
    metrics = pd.DataFrame({"step": [10, 20], "validation_loss": [3.0, 2.0]})
    joined = _backward_validation_join(alpha, metrics)
    assert joined.validation_step.tolist() == [10, 10, 20]
    assert joined.validation_loss.tolist() == [3.0, 3.0, 2.0]
    assert (joined.validation_step <= joined.optimizer_step).all()


def test_temporal_outputs_are_paired_seed_trajectories(tmp_path):
    out = tmp_path / "analysis"; out.mkdir()
    alpha = []
    runs = []
    for arm, ext, errors, losses in [
        ("adamw", "none", [.5, .4, .3], [3.0, 2.8, 2.5]),
        ("adamw_wwpgd", "wwpgd", [.5, .2, .1], [3.0, 2.6, 2.2]),
    ]:
        run = tmp_path / arm; run.mkdir()
        pd.DataFrame({"step": [10, 20, 30], "tokens_seen": [100, 200, 300],
                      "validation_loss": losses}).to_csv(run / "metrics.csv", index=False)
        for step, error in zip([10, 20, 30], errors):
            alpha.append({"arm_name": arm, "seed": 7, "optimizer_step": step,
                          "tokens_seen": step * 10, "median_absolute_alpha_error": error,
                          "fraction_inside_configured_target_deadband": 1 - error})
        runs.append({"run_dir": run, "seed": 7, "base_optimizer": "adamw",
                     "extension": ext, "optimizer_raw": arm})
    pd.DataFrame(alpha).to_csv(out / "alpha_summary_by_step.csv", index=False)
    pd.DataFrame([{"seed": 7, "base_optimizer": "adamw", "tokens_saved": 50}]).to_csv(
        out / "acceleration_by_seed.csv", index=False)
    pd.DataFrame({"optimizer_step": [20], "cache_activated": [True]}).to_csv(
        tmp_path / "adamw_wwpgd" / "wwpgd_endpoint_measurements.csv", index=False)
    analyze_paired_alpha_validation(runs, out)
    aligned = pd.read_csv(out / "alpha_validation_alignment_by_seed.csv")
    assert len(aligned) == 3
    assert aligned.delta_alpha_distance.tolist() == pytest.approx([0.0, -.2, -.2])
    assert aligned.delta_validation_loss.tolist() == pytest.approx([0.0, -.2, -.3])
    assert (aligned.validation_step <= aligned.optimizer_step).all()
    event = pd.read_csv(out / "endpoint_event_study.csv")
    assert event.event_time.tolist() == [-1, 0, 1, 2, 3]
    assert event.observed.tolist() == [True, True, True, False, False]
    assert {"acceleration_alpha_association.csv", "paired_alpha_and_validation_plot.png"} <= {
        p.name for p in out.iterdir()}
