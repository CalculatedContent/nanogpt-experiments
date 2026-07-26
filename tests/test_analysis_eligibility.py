import json

import pytest
import yaml

from wwgpt.acceleration_analysis import plan_manifest, verify_analysis_eligibility


def _plan(path, *, seeds=10):
    path.write_text(yaml.safe_dump({"mode": "confirmatory", "confirmatory_paired_seeds": seeds,
        "thresholds": [2.5], "primary_outcomes": ["tokens_saved"]}))
    return path


def _runs(tmp_path, digest, count):
    rows = []
    for seed in range(count):
        for extension in ("none", "wwpgd"):
            run = tmp_path / f"{seed}_{extension}"; run.mkdir(parents=True)
            (run / "run_complete.json").write_text("{}")
            rows.append({"seed": seed, "base_optimizer": "adamw", "extension": extension,
                         "run_dir": run, "manifest": {"analysis_plan_sha256": digest}})
    return rows


def test_plan_manifest_propagates_frozen_confirmatory_fields(tmp_path):
    manifest = plan_manifest(_plan(tmp_path / "plan.yaml"))
    assert len(manifest["analysis_plan_sha256"]) == 64
    assert manifest["confirmatory_paired_seeds"] == 10
    assert manifest["analysis_thresholds"] == [2.5]
    assert manifest["analysis_primary_outcomes"] == ["tokens_saved"]


def test_confirmatory_hash_mismatch_writes_eligibility_then_fails(tmp_path):
    plan = _plan(tmp_path / "plan.yaml")
    runs = _runs(tmp_path / "runs", "0" * 64, 10)
    with pytest.raises(RuntimeError, match="does not match"):
        verify_analysis_eligibility(runs, tmp_path / "analysis", plan)
    artifact = json.loads((tmp_path / "analysis/analysis_eligibility.json").read_text())
    assert artifact["eligible"] is False
    assert artifact["plan_hash_status"] == "mismatch"


def test_insufficient_confirmatory_pairs_write_eligibility_then_fail(tmp_path):
    plan = _plan(tmp_path / "plan.yaml")
    digest = plan_manifest(plan)["analysis_plan_sha256"]
    with pytest.raises(RuntimeError, match="requires 10"):
        verify_analysis_eligibility(_runs(tmp_path / "runs", digest, 5), tmp_path / "analysis", plan)
    artifact = json.loads((tmp_path / "analysis/analysis_eligibility.json").read_text())
    assert artifact["observed_paired_seeds_by_optimizer"] == {"adamw": 5}
    assert artifact["eligible"] is False
