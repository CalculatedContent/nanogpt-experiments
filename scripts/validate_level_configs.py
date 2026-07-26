#!/usr/bin/env python3
"""Validate and print the real Level 0-2 adaptive WWPGD schedules."""
from __future__ import annotations

import json
from pathlib import Path

from wwgpt.adaptive_wwpgd import validate_adaptive_level_schedule
from wwgpt.config import load_config
from wwgpt.model import GPT
from wwgpt.scaling import plan_budget, selected_parameter_count


def level_schedule(level: int, token_multiplier: int = 20) -> dict[str, object]:
    path = Path(f"configs/level{level}_adaptive_alpha.yaml")
    cfg = load_config(path, level)
    report = GPT(cfg.model).parameter_report()
    count = selected_parameter_count(report, cfg.parameter_count_convention)
    budget = plan_budget(
        count,
        token_multiplier,
        cfg.train.batch_size,
        cfg.model.block_size,
        cfg.train.gradient_accumulation,
        10**18,
    )
    schedule = validate_adaptive_level_schedule(
        cfg.wwpgd.adaptive,
        budget.steps,
        cfg.measurement.alpha_interval,
    )
    return {
        "level": level,
        "config": str(path),
        "target_alpha": cfg.wwpgd.target_alpha,
        "layer_lr": cfg.train.layer_lr,
        "optimizer_steps": budget.steps,
        "measurement_interval": cfg.measurement.alpha_interval,
        **schedule,
    }


def main() -> None:
    rows = [level_schedule(level) for level in (0, 1, 2)]
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
