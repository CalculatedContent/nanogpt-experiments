import json
from pathlib import Path

import pandas as pd

from wwgpt.integrity import audit_arm, audit_trial, audit_experiment, CANONICAL_ARMS


def _write_run(trial: Path, arm: str, *, data_hash="data", tokenizer_hash="tok", weight_decay=0.1, complete=True, projection=True):
    run = trial / arm / "run_000001"
    run.mkdir(parents=True)
    base = arm.removesuffix("_wwpgd")
    is_ww = arm.endswith("_wwpgd")
    fp = {"name": base, "groups": [{"weight_decay": weight_decay, "lr": 0.001}]}
    man = {
        "valid_for_science": True,
        "arm_name": arm,
        "optimizer": arm,
        "base_optimizer": base,
        "extension": "wwpgd" if is_ww else "none",
        "seed": 7,
        "data_hash": data_hash,
        "tokenizer_hash": tokenizer_hash,
        "model_configuration_hash": "model",
        "model_config_hash": "model-old",
        "realized_tokens": 1024,
        "requested_tokens": 1024,
        "target_train_tokens": 1024,
        "initialization_hash": "init",
        "training_schedule_hash": f"sched-{base}",
        "resolved_stochastic_seeds": {"train_reader_seed": hash(base) % 1000, "dropout_seed": 1},
        "optimizer_fingerprint": fp,
        "weight_decay": weight_decay,
        "wwpgd_implementation": "ww_pgd" if is_ww else "none",
        "wwpgd_commit": "abc" if is_ww else "",
    }
    (run / "manifest.json").write_text(json.dumps(man))
    if complete:
        (run / "run_complete.json").write_text(json.dumps({"step": 2, "wwpgd_call_count": 2 if is_ww else 0, "projected_matrix_count": 4 if is_ww else 0}))
    pd.DataFrame([{"step": 2, "validation_loss": 1.0, "selected_checkpoint_step": 2, "test_loss": 1.2}]).to_csv(run / "metrics.csv", index=False)
    ck = run / "checkpoints"; ck.mkdir()
    (ck / "latest.json").write_text("{}")
    if is_ww and projection:
        pd.DataFrame([{"projection_event": 1, "layer_name": "blocks.0.attn.c_proj"}]).to_csv(run / "wwpgd_projection.csv", index=False)
    return run


def _write_trial(tmp_path: Path, arms=CANONICAL_ARMS, **kwargs):
    trial = tmp_path / "trial_7"
    trial.mkdir()
    (trial / "trial_manifest.json").write_text(json.dumps({"trial_id": "trial_7"}))
    for arm in arms:
        _write_run(trial, arm, **kwargs)
    return trial


def test_valid_baseline_does_not_require_projection_artifacts(tmp_path):
    run = _write_run(tmp_path / "trial", "adamw")
    result = audit_arm(run, "adamw")
    assert result["passed"]
    assert not (run / "wwpgd_projection.csv").exists()


def test_valid_wwpgd_arm_requires_metadata_counts_and_projection(tmp_path):
    run = _write_run(tmp_path / "trial", "adamw_wwpgd")
    result = audit_arm(run, "adamw_wwpgd")
    assert result["passed"]


def test_missing_projection_records_fails_wwpgd_arm(tmp_path):
    run = _write_run(tmp_path / "trial", "adamw_wwpgd", projection=False)
    result = audit_arm(run, "adamw_wwpgd")
    assert not result["passed"]
    assert any("projection" in r for r in result["reasons"])


def test_mismatched_weight_decay_fails_pair_fingerprint(tmp_path):
    trial = _write_trial(tmp_path)
    adamw_ww = next((trial / "adamw_wwpgd").glob("run_*/manifest.json"))
    man = json.loads(adamw_ww.read_text())
    man["optimizer_fingerprint"]["groups"][0]["weight_decay"] = 0.2
    man["weight_decay"] = 0.2
    adamw_ww.write_text(json.dumps(man))
    result = audit_trial(trial)
    assert not result["publication_eligible"]
    assert any("base_optimizer_fingerprint_mismatch" in r for r in result["reasons"])


def test_mismatched_data_fails_all_arm_identity(tmp_path):
    trial = _write_trial(tmp_path)
    muon = next((trial / "muon").glob("run_*/manifest.json"))
    man = json.loads(muon.read_text()); man["data_hash"] = "other"; muon.write_text(json.dumps(man))
    result = audit_trial(trial)
    assert not result["publication_eligible"]
    assert "all_arms:data_hash_mismatch" in result["reasons"]


def test_incomplete_trial_is_not_publication_eligible(tmp_path):
    trial = _write_trial(tmp_path, arms=("adamw", "adamw_wwpgd"))
    result = audit_trial(trial)
    assert not result["publication_eligible"]
    assert result["passed_arm_count"] == 2
    summary_path = audit_experiment(tmp_path)
    summary = json.loads(summary_path.read_text())
    assert summary["publication_eligible_trials"] == 0
