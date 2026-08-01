from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "level_0_baseline" / "src"))
sys.path.insert(0, str(REPO / "level_0_wwpgd" / "src"))

from level0_wwpgd.config import load_config  # noqa: E402
from level0_wwpgd.model import GPT, GPTConfig  # noqa: E402
from level0_wwpgd.wwpgd_extension import WWPGDExtension  # noqa: E402


def _tiny_model() -> GPT:
    return GPT(
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=16,
        )
    )


def _config() -> dict:
    return yaml.safe_load(
        (REPO / "level_0_wwpgd/configs/level0.yaml").read_text()
    )["wwpgd"]


def _linalg_error() -> BaseException:
    error_type = getattr(torch._C, "_LinAlgError", RuntimeError)
    return error_type(
        "linalg.svd failed to converge because the matrix is ill-conditioned"
    )


def _adapter_with(candidate_call):
    return (
        lambda module, resolved: object(),
        lambda: object(),
        lambda requested: object(),
        lambda enabled, requested: {},
        candidate_call,
    )


def test_retryable_failure_skips_event_without_mutating_live_model(monkeypatch):
    model = _tiny_model()
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    extension = WWPGDExtension(model, _config())

    def fail(*args, **kwargs):
        del args, kwargs
        raise _linalg_error()

    monkeypatch.setattr(extension, "_adapter", lambda: _adapter_with(fail))
    rows = extension.after_optimizer_step(
        optimizer_step=1,
        total_optimizer_steps=10,
        tokens_seen=128,
    )

    assert len(rows) == 1
    assert rows[0]["projection_status"] == "skipped_retryable_error"
    assert rows[0]["consecutive_failures"] == 1
    assert rows[0]["changed"] is False
    assert extension.call_count == 1
    assert extension.successful_call_count == 0
    assert extension.failed_call_count == 1
    assert extension.consecutive_failure_count == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_success_resets_consecutive_failure_counter(monkeypatch):
    model = _tiny_model()
    extension = WWPGDExtension(model, _config())
    attempts = 0

    def fail_then_succeed(holder, config, **kwargs):
        nonlocal attempts
        del config, kwargs
        attempts += 1
        if attempts == 1:
            raise _linalg_error()

        records = []
        with torch.no_grad():
            for safe_name in holder.safe_to_live:
                layer = getattr(holder, safe_name)
                layer.weight.add_(1e-4)
                records.append(
                    {
                        "name": safe_name,
                        "alpha": 2.5,
                        "D": 0.1,
                        "xmin": 0.01,
                        "num_evals": layer.weight.shape[0],
                    }
                )
        return {"ww_logs": [pd.DataFrame(records)]}

    monkeypatch.setattr(
        extension,
        "_adapter",
        lambda: _adapter_with(fail_then_succeed),
    )

    failed_rows = extension.after_optimizer_step(
        optimizer_step=1,
        total_optimizer_steps=10,
        tokens_seen=128,
    )
    success_rows = extension.after_optimizer_step(
        optimizer_step=2,
        total_optimizer_steps=10,
        tokens_seen=256,
    )

    assert failed_rows[0]["projection_status"] == "skipped_retryable_error"
    assert len(success_rows) == 6
    assert {row["projection_status"] for row in success_rows} == {"projected"}
    assert extension.call_count == 2
    assert extension.successful_call_count == 1
    assert extension.failed_call_count == 1
    assert extension.consecutive_failure_count == 0
    assert extension.max_observed_consecutive_failures == 1


def test_fifth_consecutive_retryable_failure_stops_run(monkeypatch):
    model = _tiny_model()
    config = _config()
    config["max_consecutive_failures"] = 5
    extension = WWPGDExtension(model, config)

    def fail(*args, **kwargs):
        del args, kwargs
        raise _linalg_error()

    monkeypatch.setattr(extension, "_adapter", lambda: _adapter_with(fail))

    for step in range(1, 5):
        rows = extension.after_optimizer_step(
            optimizer_step=step,
            total_optimizer_steps=10,
            tokens_seen=step * 128,
        )
        assert rows[0]["consecutive_failures"] == step

    with pytest.raises(RuntimeError, match="5 consecutive numerical projection"):
        extension.after_optimizer_step(
            optimizer_step=5,
            total_optimizer_steps=10,
            tokens_seen=640,
        )

    assert extension.call_count == 5
    assert extension.failed_call_count == 5
    assert extension.successful_call_count == 0
    assert extension.consecutive_failure_count == 5
    assert extension.max_observed_consecutive_failures == 5


def test_config_rejects_nonpositive_failure_limit(tmp_path):
    cfg = yaml.safe_load(
        (REPO / "level_0_wwpgd/configs/level0.yaml").read_text()
    )
    cfg["wwpgd"]["max_consecutive_failures"] = 0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="max_consecutive_failures"):
        load_config(path)
