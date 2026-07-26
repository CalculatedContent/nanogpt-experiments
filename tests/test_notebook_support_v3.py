from pathlib import Path

import pandas as pd
import pytest

from wwgpt.notebook_support import (
    effect_direction,
    normalize_selected_checkpoint_metrics,
    paired_effect,
    required_probe_tokens,
    split_capacity,
)


def test_selected_checkpoint_aliases_match_runtime_schema() -> None:
    raw = pd.DataFrame(
        [
            {
                "selected_step": 75,
                "train_loss": 2.0,
                "validation_loss": 2.2,
                "test_loss": 2.3,
                "train_perplexity": 7.0,
                "validation_perplexity": 9.0,
                "test_perplexity": 10.0,
                "train_accuracy": 0.40,
                "validation_accuracy": 0.35,
                "test_accuracy": 0.34,
                "train_validation_gap": 0.2,
                "train_test_gap": 0.3,
            }
        ]
    )
    normalized = normalize_selected_checkpoint_metrics(raw)
    assert normalized.loc[0, "selected_checkpoint_step"] == 75
    assert normalized.loc[0, "validation_next_token_accuracy"] == pytest.approx(0.35)
    assert normalized.loc[0, "test_next_token_accuracy"] == pytest.approx(0.34)
    assert normalized.loc[0, "train_test_loss_gap"] == pytest.approx(0.3)
    assert normalized.loc[0, "train_test_perplexity_gap"] == pytest.approx(3.0)


def test_paired_effect_sign_conventions_are_explicit() -> None:
    assert paired_effect("test_loss", 2.0, 1.8) == pytest.approx(-0.2)
    assert effect_direction("test_loss") == "negative_is_better"
    assert paired_effect("test_next_token_accuracy", 0.30, 0.35) == pytest.approx(0.05)
    assert effect_direction("test_next_token_accuracy") == "positive_is_better"


def test_probe_capacity_uses_eval_batches_batch_size_and_context(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        '{"optimizer_hyperparameters":{"eval_batches":20,"batch_size":16},'
        '"model_config":{"block_size":256}}'
    )
    (run / "data_manifest.json").write_text(
        '{"validation_tokens":100000,"test_tokens":90000}'
    )
    assert required_probe_tokens(
        {
            "optimizer_hyperparameters": {"eval_batches": 20, "batch_size": 16},
            "model_config": {"block_size": 256},
        }
    ) == 81921
    capacity = split_capacity(run)
    assert capacity["validation_capacity_ok"] is True
    assert capacity["test_capacity_ok"] is True
