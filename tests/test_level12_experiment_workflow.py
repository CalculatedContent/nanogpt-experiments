from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from wwgpt.config import load_config
from wwgpt.cross_level_analysis import analyze_cross_level_effects
from wwgpt.data import (
    evaluation_probe_capacity,
    required_evaluation_tokens,
    validate_evaluation_capacity,
)


@pytest.mark.parametrize(
    ("level", "interval", "first_active"),
    [(1, 250, 500), (2, 1000, 3000)],
)
def test_level_dry_run_reports_runtime_alpha_cadence(
    tmp_path: Path, level: int, interval: int, first_active: int
) -> None:
    command = [
        sys.executable,
        "-m",
        "wwgpt.cli",
        "run-multiseed",
        "--level",
        str(level),
        "--config",
        f"configs/level{level}_adaptive_alpha.yaml",
        "--data-root",
        str(tmp_path / "data"),
        "--results-root",
        str(tmp_path / "results"),
        "--token-multiplier",
        "20",
        "--seeds",
        "1337",
        "--optimizer",
        "adamw",
        "--extensions",
        "none,wwpgd",
        "--device",
        "cpu",
        "--dry-run",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout.split("\n", 1)[1])
    assert payload["endpoint_measurement_interval"] == interval
    assert payload["endpoint_measurement_source"] == "measurement.alpha_interval"
    assert (
        payload["wwpgd_adaptive_schedule"]["first_possible_active_endpoint_step"]
        == first_active
    )


def test_evaluation_capacity_is_enforced_before_training() -> None:
    cfg = load_config(Path("configs/level2_adaptive_alpha.yaml"), 2)
    required = required_evaluation_tokens(cfg)
    assert required == 81_921
    assert evaluation_probe_capacity(required, cfg) == cfg.train.eval_batches
    validate_evaluation_capacity(
        {"validation_tokens": required, "test_tokens": required}, cfg
    )
    with pytest.raises(RuntimeError, match="insufficient test tokens"):
        validate_evaluation_capacity(
            {"validation_tokens": required, "test_tokens": required - 1}, cfg
        )


def _write_run(
    root: Path,
    *,
    level: int,
    seed: int,
    extension: str,
    test_loss: float,
    test_perplexity: float,
    test_accuracy: float,
    pair_suffix: str = "",
    run_name: str = "run_001",
) -> Path:
    arm = "adamw" if extension == "none" else "adamw_wwpgd"
    run = (
        root
        / "experiments"
        / f"level_{level:02d}"
        / "multiplier_20"
        / f"pair_{seed}{pair_suffix}"
        / arm
        / run_name
    )
    run.mkdir(parents=True)
    manifest = {
        "scientific_schema_version": 3,
        "valid_for_science": True,
        "pair_id": f"pair_{seed}{pair_suffix}",
        "seed": seed,
        "level": level,
        "token_multiplier": 20,
        "base_optimizer": "adamw",
        "extension": extension,
        "arm_name": arm,
        "selected_parameter_count": [1000, 2000, 4000][level],
    }
    selected = {
        "selected_checkpoint_step": 10,
        "test_loss": test_loss,
        "test_perplexity": test_perplexity,
        "test_accuracy": test_accuracy,
        "validation_loss": test_loss - 0.1,
        "validation_perplexity": test_perplexity - 0.2,
        "validation_accuracy": test_accuracy + 0.01,
        "train_validation_gap": 0.2,
        "train_test_gap": 0.3,
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "run_complete.json").write_text("{}")
    (run / "selected_checkpoint_metrics.json").write_text(json.dumps(selected))
    return run


def test_cross_level_analysis_selects_newest_complete_pair_without_mixing_arms(
    tmp_path: Path,
) -> None:
    # Older complete pair: effect -1.0.
    old_base = _write_run(
        tmp_path,
        level=0,
        seed=1,
        extension="none",
        test_loss=10.0,
        test_perplexity=20.0,
        test_accuracy=0.2,
        pair_suffix="_old",
    )
    old_ww = _write_run(
        tmp_path,
        level=0,
        seed=1,
        extension="wwpgd",
        test_loss=9.0,
        test_perplexity=19.0,
        test_accuracy=0.3,
        pair_suffix="_old",
    )
    # Newer complete pair: effect -0.25. Mixing newest arms independently could
    # incorrectly produce a different effect, so both arms must share pair_id.
    new_base = _write_run(
        tmp_path,
        level=0,
        seed=1,
        extension="none",
        test_loss=5.0,
        test_perplexity=10.0,
        test_accuracy=0.4,
        pair_suffix="_new",
    )
    new_ww = _write_run(
        tmp_path,
        level=0,
        seed=1,
        extension="wwpgd",
        test_loss=4.75,
        test_perplexity=9.75,
        test_accuracy=0.45,
        pair_suffix="_new",
    )
    for index, run in enumerate((old_base, old_ww, new_base, new_ww), start=1):
        timestamp = 1_000 + index
        for artifact in (
            run / "manifest.json",
            run / "run_complete.json",
            run / "selected_checkpoint_metrics.json",
        ):
            artifact.touch()
            artifact.chmod(0o644)
            import os
            os.utime(artifact, (timestamp, timestamp))

    paired = analyze_cross_level_effects(tmp_path, tmp_path / "analysis")[
        "cross_level_paired_effects_by_seed.csv"
    ]
    loss = paired[paired.metric.eq("test_loss")]
    assert len(loss) == 1
    assert loss.iloc[0].pair_id == "pair_1_new"
    assert loss.iloc[0].paired_effect == pytest.approx(-0.25)


def test_cross_level_analysis_pairs_seeds_and_reports_model_trend(
    tmp_path: Path,
) -> None:
    for level in range(3):
        for seed in (1, 2):
            baseline_loss = 3.0 - 0.1 * level + 0.01 * seed
            improvement = 0.01 * (level + 1)
            _write_run(
                tmp_path,
                level=level,
                seed=seed,
                extension="none",
                test_loss=baseline_loss,
                test_perplexity=20.0 - level,
                test_accuracy=0.20 + 0.02 * level,
            )
            _write_run(
                tmp_path,
                level=level,
                seed=seed,
                extension="wwpgd",
                test_loss=baseline_loss - improvement,
                test_perplexity=20.0 - level - improvement,
                test_accuracy=0.20 + 0.02 * level + improvement,
            )
    output = tmp_path / "analysis"
    tables = analyze_cross_level_effects(tmp_path, output)
    paired = tables["cross_level_paired_effects_by_seed.csv"]
    summary = tables["cross_level_paired_effect_summary.csv"]
    trends = tables["cross_level_model_size_trends.csv"]
    readiness = tables["cross_level_scaling_readiness.csv"].iloc[0]

    test_loss = paired[paired.metric.eq("test_loss")]
    assert len(test_loss) == 6
    assert (test_loss.paired_effect < 0).all()
    assert set(summary[summary.metric.eq("test_loss")].level) == {0, 1, 2}
    trend = trends[trends.metric.eq("test_loss")].iloc[0]
    assert trend.trend_direction == "more_favorable_with_level"
    assert bool(readiness.ready_for_descriptive_level_trend)
    assert not bool(readiness.ready_for_scaling_law_fit)
    for name in tables:
        assert (output / name).is_file()


def test_scaling_notebook_uses_cross_level_analysis_without_level_filter() -> None:
    notebook = json.loads(Path("notebooks/04_scaling_laws.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "analyze_cross_level_effects" in source
    assert "LEVEL is intentionally not used as a filter" in source
    assert "runs[runs.level" not in source
