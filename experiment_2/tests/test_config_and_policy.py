from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiment_2" / "src"))

from experiment_2.config import load_config, validate_config  # noqa: E402
from experiment_2.model import GPT, GPTConfig, projected_modules  # noqa: E402
from experiment_2.policy import LayerwiseController, decide_projection  # noqa: E402


def test_level0_and_level1_protocols_are_separate_and_valid() -> None:
    expected = {
        "level0": (4, 4, 128, 256, 2_000, 24),
        "level1": (8, 8, 512, 512, 10_000, 48),
    }
    for scale, values in expected.items():
        cfg = load_config(REPO / "experiment_2" / "configs" / f"{scale}.yaml")
        layers, heads, width, context, steps, matrices = values
        assert cfg["experiment"]["name"] == "experiment_2"
        assert cfg["experiment"]["scale"] == scale
        assert cfg["model"]["n_layer"] == layers
        assert cfg["model"]["n_head"] == heads
        assert cfg["model"]["n_embd"] == width
        assert cfg["model"]["block_size"] == context
        assert cfg["training"]["max_steps"] == steps
        assert cfg["controller"]["target_alpha"] == 2.0
        assert cfg["wwpgd"]["target_alpha"] == 2.0
        model = GPT(GPTConfig(**cfg["model"]))
        assert len(projected_modules(model)) == matrices


def test_config_rejects_invalid_controller_relationships() -> None:
    cfg = load_config(REPO / "experiment_2" / "configs" / "level0.yaml")
    cfg["controller"]["exit_tolerance"] = cfg["controller"]["enter_tolerance"]
    with pytest.raises(ValueError, match="exit_tolerance"):
        validate_config(cfg)


def test_projection_decision_accepts_helpful_bounded_candidate() -> None:
    original = torch.ones(4, 4)
    adamw_delta = torch.full((4, 4), 0.01)
    projection_delta = torch.full((4, 4), 0.02)
    decision = decide_projection(
        alpha_before=3.0,
        alpha_after_candidate=2.2,
        target_alpha=2.0,
        adamw_delta=adamw_delta,
        projection_delta=projection_delta,
        original_weight=original,
        min_alignment_cosine=-0.1,
        min_alpha_improvement=0.01,
        max_projection_to_adamw_ratio=0.5,
        max_relative_frobenius_change=0.1,
    )
    assert decision.status == "projected"
    assert math.isclose(decision.alignment_cosine, 1.0, rel_tol=1e-6)
    assert math.isclose(decision.requested_ratio, 2.0, rel_tol=1e-6)
    assert math.isclose(decision.scale, 0.25, rel_tol=1e-6)
    assert decision.alpha_improvement > 0


def test_projection_decision_rejects_candidate_that_undoes_adamw() -> None:
    original = torch.ones(4, 4)
    adamw_delta = torch.full((4, 4), 0.01)
    projection_delta = torch.full((4, 4), -0.01)
    decision = decide_projection(
        alpha_before=3.0,
        alpha_after_candidate=2.1,
        target_alpha=2.0,
        adamw_delta=adamw_delta,
        projection_delta=projection_delta,
        original_weight=original,
        min_alignment_cosine=-0.1,
        min_alpha_improvement=0.01,
        max_projection_to_adamw_ratio=0.5,
        max_relative_frobenius_change=0.1,
    )
    assert decision.status == "rejected_adamw_opposition"
    assert decision.alignment_cosine < -0.99


def test_layer_controller_selects_largest_alpha_errors_and_respects_pause() -> None:
    model = GPT(GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=1, n_embd=16))
    cfg = load_config(REPO / "experiment_2" / "configs" / "level0.yaml")["controller"]
    cfg = dict(cfg)
    cfg.update(start_step=1, max_active_layers=2)
    controller = LayerwiseController(model, cfg)

    measurements = []
    for index, (name, matrix_type, block, _) in enumerate(projected_modules(model)):
        measurements.append(
            {
                "layer_name": name,
                "matrix_type": matrix_type,
                "block": block,
                "alpha": 4.0 - 0.2 * index,
                "D": 0.05,
                "xmin": 1.0,
                "num_evals": 16,
            }
        )
    controller.update_measurements(measurements, step=1)
    cohort = controller.choose_cohort(step=1)
    assert len(cohort) == 2
    assert cohort[0] == measurements[0]["layer_name"]
    assert cohort[1] == measurements[1]["layer_name"]

    controller.global_pause_until = 100
    assert controller.choose_cohort(step=2) == []
