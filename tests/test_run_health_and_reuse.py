from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from wwgpt.config import ModelConfig, TrainConfig
from wwgpt.data import required_evaluation_tokens
from wwgpt.run_health import generate_experiment_health, generate_run_health


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def _valid_run(tmp_path: Path, extension: str = "none") -> Path:
    run = tmp_path / "pair_1337" / ("adamw_wwpgd" if extension == "wwpgd" else "adamw") / "run_1"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _write_json(
        run / "manifest.json",
        {
            "valid_for_science": True, "scientific_schema_version": 3,
            "arm_name": "adamw_wwpgd" if extension == "wwpgd" else "adamw",
            "base_optimizer": "adamw", "extension": extension, "seed": 1337,
            "level": 0, "token_multiplier": 20,
            "optimizer_hyperparameters": {"eval_batches": 1, "batch_size": 1},
            "model_config": {"block_size": 4},
            "wwpgd_installed_version": "0.1", "wwpgd_commit": "runtime",
        },
    )
    _write_json(
        run / "run_complete.json",
        {"step": 1, "wwpgd_call_count": 0, "stock_wwpgd_invocation_count": 0},
    )
    _write_json(
        run / "data_manifest.json",
        {"validation_tokens": 100, "test_tokens": 100},
    )
    selected = {
        "train_loss": 2.0, "train_perplexity": 7.0, "train_accuracy": 0.2,
        "validation_loss": 2.1, "validation_perplexity": 8.0, "validation_accuracy": 0.19,
        "test_loss": 2.2, "test_perplexity": 9.0, "test_accuracy": 0.18,
        "checkpoint_path": str(checkpoint), "checkpoint_hash": digest,
        "selection_metric": "validation_loss",
    }
    _write_json(run / "selected_checkpoint_metrics.json", selected)
    pd.DataFrame(
        [{
            "evaluation_index": 0, "step": 1, "train_loss": 2.0,
            "validation_loss": 2.1, "gradient_norm_before_clip": 1.0,
            "gradient_norm_after_clip": 1.0, "model_parameter_l2_norm": 3.0,
            "eligible_matrix_l2_norm": 2.0, "model_max_abs_parameter": 0.5,
            "model_nonfinite_parameter_count": 0,
            "gradient_nonfinite_element_count_before_clip": 0,
            "total_elapsed_seconds": 1.0,
        }]
    ).to_csv(run / "metrics.csv", index=False)
    pd.DataFrame(
        [{
            "optimizer_step": 1, "layer_name": "blocks.0.attn.key",
            "alpha": 2.5, "valid_for_science": True, "projected": True,
            "included_in_projected_alpha_summary": True,
        }]
    ).to_csv(run / "alpha_measurements.csv", index=False)
    pd.DataFrame([{"step": 1, "trap_layer_fraction": 0.0}]).to_csv(
        run / "weightwatcher_aggregates.csv", index=False
    )
    return run


def test_required_evaluation_tokens_matches_fixed_probe() -> None:
    class Cfg:
        train = TrainConfig(batch_size=2, eval_batches=3)
        model = ModelConfig(block_size=4)
    assert required_evaluation_tokens(Cfg()) == 25


def test_run_health_writes_machine_readable_artifacts(tmp_path: Path) -> None:
    run = _valid_run(tmp_path)
    report = generate_run_health(run)
    assert report["ready_for_analysis"] is True
    assert (run / "run_health.json").is_file()
    assert (run / "run_health.csv").is_file()


def test_run_health_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    run = _valid_run(tmp_path)
    frame = pd.read_csv(run / "metrics.csv")
    frame.loc[0, "validation_loss"] = float("nan")
    frame.to_csv(run / "metrics.csv", index=False)
    report = generate_run_health(run)
    assert report["ready_for_analysis"] is False
    assert any(row["check"] == "finite_metrics" for row in report["findings"])


def test_experiment_health_aggregates_runs(tmp_path: Path) -> None:
    _valid_run(tmp_path)
    report = generate_experiment_health(tmp_path)
    assert report["run_count"] == 1
    assert report["ready_for_analysis"] is True
    assert (tmp_path / "analysis" / "run_health_summary.csv").is_file()


def test_local_runner_prepares_before_optimizer_loop() -> None:
    text = Path("scripts/run_local_mac_experiments.sh").read_text()
    preparation = text.index('# Prepare/reuse each immutable data identity exactly once.')
    mode_loop = text.index('for MODE in')
    assert preparation < mode_loop
    assert 'wwgpt check-health' in text
    assert '--wwpgd-max-endpoint-fraction-per-refresh' in text
    assert '--wwpgd-candidate-device' in text


def test_experiment_health_ignores_superseded_incomplete_attempt(tmp_path: Path) -> None:
    complete = _valid_run(tmp_path)
    abandoned = complete.parent / "run_99999999_abandoned"
    abandoned.mkdir()
    (abandoned / "manifest.json").write_text((complete / "manifest.json").read_text())

    report = generate_experiment_health(tmp_path)
    assert report["ready_for_analysis"] is True
    assert report["run_count"] == 1
    assert report["total_attempt_count"] == 2
    assert report["excluded_superseded_attempt_count"] == 1


def test_run_health_reports_malformed_optional_alpha_schema_without_crashing(
    tmp_path: Path,
) -> None:
    run = _valid_run(tmp_path)
    pd.DataFrame(
        [{"optimizer_step": 1, "layer_name": "blocks.0.attn.key", "not_alpha": 2.5}]
    ).to_csv(run / "alpha_measurements.csv", index=False)

    report = generate_run_health(run)

    assert report["ready_for_analysis"] is False
    assert any(
        row["check"] == "artifact_schema"
        and "alpha_measurements.csv" in row["message"]
        for row in report["findings"]
    )


def test_run_health_reports_unreadable_optional_csv_without_crashing(
    tmp_path: Path,
) -> None:
    run = _valid_run(tmp_path)
    (run / "alpha_measurements.csv").write_text('alpha,"unterminated\n')

    report = generate_run_health(run)

    assert report["ready_for_analysis"] is False
    assert any(
        row["check"] == "artifact_parse"
        and "alpha_measurements.csv" in row["message"]
        for row in report["findings"]
    )


def test_run_health_reports_malformed_wwpgd_optional_schemas_without_crashing(
    tmp_path: Path,
) -> None:
    run = _valid_run(tmp_path, extension="wwpgd")
    _write_json(
        run / "run_complete.json",
        {"step": 1, "wwpgd_call_count": 1, "stock_wwpgd_invocation_count": 1},
    )
    pd.DataFrame(
        [{"optimizer_step": 1, "layer_name": "blocks.0.attn.key", "other": 1}]
    ).to_csv(run / "wwpgd_internal_diagnostics.csv", index=False)
    pd.DataFrame(
        [{"optimizer_step": 1, "layer_name": "blocks.0.attn.key", "other": 1}]
    ).to_csv(run / "wwpgd_endpoint_measurements.csv", index=False)
    pd.DataFrame(
        [{"optimizer_step": 1, "layer_name": "blocks.0.attn.key", "other": 1}]
    ).to_csv(run / "wwpgd_endpoint_relaxation.csv", index=False)

    report = generate_run_health(run)

    assert report["ready_for_analysis"] is False
    schema_messages = [
        row["message"]
        for row in report["findings"]
        if row["check"] == "artifact_schema"
    ]
    assert any("wwpgd_internal_diagnostics.csv" in message for message in schema_messages)
    assert any("wwpgd_endpoint_measurements.csv" in message for message in schema_messages)
    assert any("wwpgd_endpoint_relaxation.csv" in message for message in schema_messages)
