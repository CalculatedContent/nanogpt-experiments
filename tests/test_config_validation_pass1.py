from pathlib import Path

import pytest
import yaml

from wwgpt.config import load_config


def _write_config(tmp_path, payload):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_valid_default_configuration_loads_and_validates_model():
    cfg = load_config(Path("configs/default.yaml"), level=0)
    assert cfg.model.n_embd == 64
    assert cfg.model.n_head == 1
    assert cfg.train.batch_size > 0
    assert cfg.seeds


def test_unknown_root_and_nested_yaml_keys_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown configuration key.*typo_root"):
        load_config(_write_config(tmp_path, {"typo_root": True}), level=0)

    with pytest.raises(ValueError, match="unknown configuration key.*train.typo_batch"):
        load_config(_write_config(tmp_path, {"train": {"typo_batch": 16}}), level=0)


def test_invalid_wwpgd_values_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="wwpgd.blend_eta"):
        load_config(_write_config(tmp_path, {"wwpgd": {"blend_eta": 1.5}}), level=0)

    with pytest.raises(ValueError, match="train.wwpgd_interval"):
        load_config(_write_config(tmp_path, {"train": {"wwpgd_interval": 0}}), level=0)


def test_invalid_schedule_values_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="train.batch_size"):
        load_config(_write_config(tmp_path, {"train": {"batch_size": 0}}), level=0)

    with pytest.raises(ValueError, match="model.block_size"):
        load_config(_write_config(tmp_path, {"model": {"block_size": 0}}), level=0)

    with pytest.raises(ValueError, match="train.learning_rate"):
        load_config(_write_config(tmp_path, {"train": {"learning_rate": 0.0}}), level=0)

    with pytest.raises(ValueError, match="train.weight_decay"):
        load_config(_write_config(tmp_path, {"train": {"weight_decay": -0.1}}), level=0)

    with pytest.raises(ValueError, match="warmup_steps"):
        load_config(_write_config(tmp_path, {"train": {"warmup_steps": -1}}), level=0)

    with pytest.raises(ValueError, match="lr_decay_steps.*greater than.*warmup_steps"):
        load_config(
            _write_config(
                tmp_path,
                {"train": {"lr_schedule": "warmup_cosine", "warmup_steps": 10, "lr_decay_steps": 10}},
            ),
            level=0,
        )


def test_invalid_model_dimensions_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="n_embd must be divisible by n_head"):
        load_config(_write_config(tmp_path, {"model": {"n_head": 3, "n_embd": 64}}), level=0)

    with pytest.raises(ValueError, match="schema-v3 requires attention head dimension 64"):
        load_config(_write_config(tmp_path, {"model": {"n_head": 2, "n_embd": 64}}), level=0)


def test_invalid_seed_and_optimizer_names_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="seeds must contain at least one seed"):
        load_config(_write_config(tmp_path, {"seeds": []}), level=0)

    with pytest.raises(ValueError, match="unknown base_optimizer"):
        load_config(_write_config(tmp_path, {"base_optimizer": "not_an_optimizer"}), level=0)

    with pytest.raises(ValueError, match="unknown extension"):
        load_config(_write_config(tmp_path, {"extensions": ["none", "invalid_extension"]}), level=0)
