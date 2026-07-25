from pathlib import Path

import pandas as pd

from wwgpt.alpha_analysis import analyze_alpha_trajectories, prepare_alpha_measurements
from wwgpt.ww import alpha_measurement_exclusion_reason


def manifest(arm="adamw"):
    return {"arm_name": arm, "seed": 7, "extension_hyperparameters": {
        "target_alpha": 2.0, "min_tail": 5, "adaptive": {"max_D": .2,
        "above_target": {"deadband": .4}, "below_target": {"deadband": .2}}}}


def rows(arm="adamw"):
    return pd.DataFrame([
        {"arm_name": arm, "seed": 7, "optimizer_step": 1, "tokens_seen": 100,
         "layer_name": "blocks.0.attn.key", "matrix_type": "W_K", "block": 0,
         "alpha": 2.1, "xmin": 1, "D": .1, "detX_num": 6,
         "spectral_estimator": "weightwatcher", "projected": True},
        {"arm_name": arm, "seed": 7, "optimizer_step": 1, "tokens_seen": 100,
         "layer_name": "blocks.0.attn.query", "matrix_type": "W_Q", "block": 0,
         "alpha": 4, "xmin": 1, "D": .3, "detX_num": 6,
         "spectral_estimator": "weightwatcher", "projected": True},
    ])


def test_shared_quality_gate_is_explicit():
    row = rows().iloc[0].to_dict()
    assert alpha_measurement_exclusion_reason(row, max_D=.2, min_tail=5) == ""
    row["spectral_estimator"] = "fallback_non_scientific"
    assert alpha_measurement_exclusion_reason(row, max_D=.2, min_tail=5) == "spectral_estimator_not_weightwatcher"


def test_target_and_deadband_come_from_manifest(tmp_path: Path):
    rows().to_csv(tmp_path / "alpha_measurements.csv", index=False)
    frame = prepare_alpha_measurements(tmp_path, manifest())
    assert frame.target_alpha.unique().tolist() == [2.0]
    assert frame.alpha_valid.tolist() == [True, False]
    assert frame.inside_band.tolist() == [True, False]


def test_writes_trajectory_artifacts(tmp_path: Path):
    runs=[]
    for arm in ("adamw", "adamw_wwpgd"):
        run = tmp_path / arm; run.mkdir(); rows(arm).to_csv(run / "alpha_measurements.csv", index=False)
        runs.append({"run_dir": run, "manifest": manifest(arm)})
    out=tmp_path / "analysis"; analyze_alpha_trajectories(runs, out)
    expected = {"alpha_summary_by_step.csv", "alpha_summary_by_matrix_type.csv",
                "paired_alpha_distance_differences.csv", "alpha_distance_trajectories.png",
                "fraction_near_target.png", "alpha_summary_by_transformer_block.csv"}
    assert expected <= {p.name for p in out.iterdir()}
    summary=pd.read_csv(out / "alpha_summary_by_step.csv")
    assert summary.valid_layer_count.tolist() == [1, 1]
    assert summary.excluded_layer_count.tolist() == [1, 1]
