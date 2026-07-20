import json

import pytest

from wwgpt.config import ExperimentConfig, ModelConfig, TrainConfig, WWPGDConfig
from wwgpt.scaling import plan_budget, resolve_optimizer_steps
from wwgpt.train import _select_resume_run, _trial_manifest


class TinyData:
    corpus_hash = "corpus"
    data_manifest = {"dataset_name": "fixture", "dataset_config": "fixture", "dataset_revision": "fixture", "realized_tokens": 999}
    tokenizer_manifest = {"tokenizer_hash": "tok"}


def test_yaml_cap_below_budget_is_honored_and_cap_above_does_not_increase():
    assert resolve_optimizer_steps(10, 3) == 3
    assert resolve_optimizer_steps(10, 30) == 10


@pytest.mark.parametrize("grad_accum", [1, 4])
def test_gradient_accumulation_changes_tokens_per_step_not_step_semantics(grad_accum):
    plan = plan_budget(100, 20, batch_size=2, block_size=5, grad_accum=grad_accum, available_tokens=10**9)
    assert resolve_optimizer_steps(plan.steps, 2) == 2
    assert plan.tokens_per_step == 10 * grad_accum


def test_canonical_trial_manifest_records_resolved_steps_for_all_six_arms():
    cfg = ExperimentConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=4, vocab_size=16),
        train=TrainConfig(batch_size=1, gradient_accumulation=2, max_steps=3, lr_schedule="constant"),
        wwpgd=WWPGDConfig(extension="none"),
    )
    manifest = _trial_manifest("trial", 0, 20, 123, cfg, TinyData(), "init")
    tb = manifest["shared"]["token_budget"]
    assert tb["configured_max_steps"] == 3
    assert tb["resolved_optimizer_steps"] == 3
    assert tb["tokens_per_optimizer_step"] == 8
    assert tb["resolved_train_tokens"] == 24
    assert tb["optimizer_step_limit_source"] == "configured_max_steps"
    assert len(manifest["arms"]) == 6
    assert {json.dumps(arm["token_budget"], sort_keys=True) for arm in manifest["arms"]} == {json.dumps(tb, sort_keys=True)}


def test_resume_selection_rejects_conflicting_resolved_step_horizon(tmp_path):
    arm = tmp_path / "adamw"
    run = arm / "run_1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "latest.json").write_text("{}")
    (run / "initialization_hash.txt").write_text("init")
    identity = {
        "pair_id": "pair",
        "arm_name": "adamw",
        "seed": 1,
        "configuration_hash": "cfg",
        "data_hash": "data",
        "tokenizer_hash": "tok",
        "initialization_hash": "init",
        "optimizer_fingerprint": {"x": "y"},
        "immediate_projection_spectral": False,
        "resolved_optimizer_steps": 2,
    }
    manifest = {**identity, "resolved_optimizer_steps": 3}
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "data_manifest.json").write_text(json.dumps({"corpus_hash": "data"}))
    (run / "tokenizer_manifest.json").write_text(json.dumps({"tokenizer_hash": "tok"}))
    with pytest.raises(RuntimeError, match="resolved_optimizer_steps"):
        _select_resume_run(arm, identity)
