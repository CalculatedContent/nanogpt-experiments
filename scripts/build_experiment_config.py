#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one explicit Level 0-2 optimizer/WWPGD experiment config."
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wwpgd-mode", choices=["adaptive", "uniform"], default="adaptive")
    parser.add_argument("--uniform-hardness", type=float, default=0.25)
    parser.add_argument("--layer-lr", choices=["flat", "llrd", "manual"])
    parser.add_argument(
        "--lr-scale-rule", choices=["fixed", "linear_batch", "sqrt_batch"]
    )
    parser.add_argument("--lr-reference-tokens-per-step", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--muon-learning-rate", type=float)
    parser.add_argument("--stable-learning-rate", type=float)
    parser.add_argument("--target-alpha", type=float)
    parser.add_argument("--blend-eta", type=float)
    parser.add_argument("--cayley-eta", type=float)
    parser.add_argument("--max-per-step-gain", type=float)
    parser.add_argument("--max-endpoint-fraction-per-refresh", type=float)
    parser.add_argument("--max-cumulative-relative-change-per-refresh", type=float)
    parser.add_argument("--per-step-relative-change-cap", type=float)
    parser.add_argument(
        "--matrix-lr-multiplier", action="append", default=[], metavar="ROLE=VALUE"
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.base_config.read_text())
    train = config.setdefault("train", {})
    wwpgd = config.setdefault("wwpgd", {})
    adaptive = wwpgd.setdefault("adaptive", {})

    if args.wwpgd_mode == "uniform":
        adaptive["mode"] = "uniform"
        adaptive["max_hardness"] = args.uniform_hardness
    else:
        adaptive["mode"] = "alpha_distance"

    for name, value in (
        ("layer_lr", args.layer_lr),
        ("lr_scale_rule", args.lr_scale_rule),
        ("lr_reference_tokens_per_step", args.lr_reference_tokens_per_step),
        ("learning_rate", args.learning_rate),
        ("muon_learning_rate", args.muon_learning_rate),
        ("stable_learning_rate", args.stable_learning_rate),
    ):
        if value is not None:
            train[name] = value

    for name, value in (
        ("target_alpha", args.target_alpha),
        ("blend_eta", args.blend_eta),
        ("cayley_eta", args.cayley_eta),
    ):
        if value is not None:
            wwpgd[name] = value

    for name, value in (
        ("max_per_step_gain", args.max_per_step_gain),
        ("max_endpoint_fraction_per_refresh", args.max_endpoint_fraction_per_refresh),
        (
            "max_cumulative_relative_frobenius_change_per_refresh",
            args.max_cumulative_relative_change_per_refresh,
        ),
        (
            "max_relative_frobenius_change_per_step",
            args.per_step_relative_change_cap,
        ),
    ):
        if value is not None:
            adaptive[name] = value

    multipliers = dict(train.get("matrix_lr_multipliers") or {})
    for item in args.matrix_lr_multiplier:
        if "=" not in item:
            parser.error("--matrix-lr-multiplier requires ROLE=VALUE")
        role, raw = item.split("=", 1)
        multipliers[role] = float(raw)
    if multipliers:
        train["matrix_lr_multipliers"] = multipliers

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
