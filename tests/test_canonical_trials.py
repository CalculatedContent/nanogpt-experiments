import json
from pathlib import Path

from wwgpt.config import ExperimentConfig, ModelConfig, TrainConfig, WWPGDConfig
from wwgpt.train import run_canonical_trials, CANONICAL_TRIAL_ARMS, CANONICAL_TRIAL_PAIRS
from wwgpt.analysis import discover_trial_manifests, discover_canonical_runs


class TinyData:
    train = [0, 1, 2, 3, 4, 5, 6, 7]
    val = [0, 1, 2, 3]
    corpus_hash = "corpus"
    data_manifest = {"dataset_name": "fixture", "dataset_config": "fixture", "dataset_revision": "fixture", "realized_tokens": 8}
    tokenizer_manifest = {"tokenizer_hash": "tok"}


def test_canonical_trial_mocked_orchestration(monkeypatch, tmp_path):
    cfg = ExperimentConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=16),
        train=TrainConfig(batch_size=1, max_steps=1, lr_schedule="constant", learning_rate=1e-3, weight_decay=0.2),
        wwpgd=WWPGDConfig(extension="none"),
    )
    calls = []
    monkeypatch.setattr("wwgpt.train.load_config", lambda config_path, level: cfg)
    monkeypatch.setattr("wwgpt.data.load_prepared_scientific_data", lambda data_root, level, token_multiplier: TinyData())

    def fake_run(parent, optimizer_name, seed, arm_cfg, data, pair_id, *args, **kwargs):
        calls.append((parent, optimizer_name, seed, arm_cfg.wwpgd.extension, pair_id))
        run = parent / optimizer_name / "run_mock"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({"arm_name": optimizer_name, "seed": seed, "scientific_schema_version": 3}))
        (run / "metrics.csv").write_text("step,val_loss\n1,1.0\n")
        (run / "run_complete.json").write_text("{}")
        return run

    monkeypatch.setattr("wwgpt.train.run_scientific_single", fake_run)
    exp_root = run_canonical_trials(0, tmp_path / "data", tmp_path / "results", 2, seeds=[123])
    trial_dirs = list(exp_root.glob("trial_123*"))
    assert len(trial_dirs) == 1
    trial = trial_dirs[0]
    assert sorted(c.name for c in trial.iterdir() if c.is_dir() and c.name != "initial_state") == sorted(CANONICAL_TRIAL_ARMS)
    assert [c[1] for c in calls] == list(CANONICAL_TRIAL_ARMS)

    manifest = json.loads((trial / "trial_manifest.json").read_text())
    assert [a["arm_name"] for a in manifest["arms"]] == list(CANONICAL_TRIAL_ARMS)
    assert manifest["pairs"] == [{"baseline": b, "wwpgd": w} for b, w in CANONICAL_TRIAL_PAIRS.items()]
    assert len(manifest["pairs"]) == 3
    assert {"model_config", "data_manifest", "token_budget", "seed", "initialization_hash"}.issubset(manifest["shared"])
    assert len({str(trial / a) for a in CANONICAL_TRIAL_ARMS}) == 6

    trials = discover_trial_manifests(exp_root)
    assert len(trials) == 1 and trials[0]["valid"]
    runs = discover_canonical_runs(exp_root)
    assert [r["optimizer_raw"] for r in runs] == list(CANONICAL_TRIAL_ARMS)
    assert all(r["pair_dir"] == trial for r in runs)
