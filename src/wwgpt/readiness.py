"""Executable local readiness checks for Level 0-2 nanoGPT/WWPGD experiments."""
from __future__ import annotations

import json
import math
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

from wwgpt.adaptive_wwpgd import validate_adaptive_level_schedule
from wwgpt.config import load_config
from wwgpt.device import device_summary, precision_policy, resolve_device, synchronize_device
from wwgpt.model import GPT
from wwgpt.optimizers import (
    apply_lr_schedule,
    build_optimizer_bundle,
    optimizer_fingerprint,
    optimizer_step,
    resolve_warmup_steps,
)
from wwgpt.pip_wwpgd_adapter import resolve_pip_wwpgd_provenance
from wwgpt.scaling import plan_budget, selected_parameter_count
from wwgpt.ww import (
    apply_external_wwpgd,
    build_stock_wwpgd_candidate,
    external_wwpgd_config_from_experiment,
    is_projected_layer,
)


def _parse_ints(values: Iterable[int] | str) -> list[int]:
    if isinstance(values, str):
        return [int(value) for value in values.split(",") if value]
    return [int(value) for value in values]


def _parse_strings(values: Iterable[str] | str) -> list[str]:
    if isinstance(values, str):
        return [value.strip() for value in values.split(",") if value.strip()]
    return [str(value).strip() for value in values if str(value).strip()]


def _protected_state(model: GPT) -> dict[str, torch.Tensor]:
    eligible = {
        f"{name}.weight"
        for name, _module in model.named_modules()
        if is_projected_layer(name)
    }
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name not in eligible
    }


def _protected_unchanged(model: GPT, before: dict[str, torch.Tensor]) -> bool:
    current = model.state_dict()
    return all(
        name in current and torch.equal(value, current[name].detach().cpu())
        for name, value in before.items()
    )


def _all_finite(model: GPT) -> bool:
    return all(torch.isfinite(parameter).all() for parameter in model.parameters())


def _one_optimizer_and_wwpgd_check(
    *,
    level: int,
    optimizer_name: str,
    config_path: Path,
    device: torch.device,
    precision: str | None,
) -> dict[str, Any]:
    cfg = load_config(config_path, level)
    torch.manual_seed(10_000 + level)
    model = GPT(cfg.model).to(device)
    bundle, llrd_gamma = build_optimizer_bundle(model, cfg.train, optimizer_name)
    sequence_length = min(cfg.model.block_size, 32)
    batch_size = min(cfg.train.batch_size, 2)
    x = torch.randint(0, cfg.model.vocab_size, (batch_size, sequence_length), device=device)
    y = torch.randint(0, cfg.model.vocab_size, (batch_size, sequence_length), device=device)
    model.train()
    _, loss = model(x, y)
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("nonfinite readiness loss before optimizer step")
    loss.backward()
    if cfg.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
    warmup = resolve_warmup_steps(10, cfg.train.warmup_ratio, cfg.train.warmup_steps, cfg.train.lr_decay_steps)
    apply_lr_schedule(bundle, 0, 10, warmup, cfg.train)
    for optimizer in bundle.optimizers:
        optimizer_step(optimizer, device)
    bundle.zero_grad()
    synchronize_device(device)
    if not _all_finite(model):
        raise RuntimeError("nonfinite model after base optimizer step")

    protected_before = _protected_state(model)
    candidate_started = time.perf_counter()
    candidate = build_stock_wwpgd_candidate(
        model,
        event_index=0,
        actual_step=1,
        cfg=external_wwpgd_config_from_experiment(cfg.wwpgd),
    )
    candidate_seconds = time.perf_counter() - candidate_started
    if not _protected_unchanged(model, protected_before):
        raise RuntimeError("stock candidate generation changed protected tensors")
    if candidate.pre_projection_details is None or candidate.pre_projection_details.empty:
        raise RuntimeError("pip-installed WWPGD returned no usable WeightWatcher table")

    eligible_names = sorted(candidate.original_weights)
    hardness = {name: 0.05 for name in eligible_names}
    protected_before_apply = _protected_state(model)
    rows = apply_external_wwpgd(
        model,
        event_index=0,
        scheduled_token_fraction=0.0,
        actual_step=1,
        actual_tokens_seen=batch_size * sequence_length,
        cfg=external_wwpgd_config_from_experiment(cfg.wwpgd),
        layer_hardness=hardness,
        global_event_hardness=1.0,
        stock_candidate=candidate,
    )
    synchronize_device(device)
    if not _protected_unchanged(model, protected_before_apply):
        raise RuntimeError("WWPGD readiness application changed protected tensors")
    if not _all_finite(model):
        raise RuntimeError("nonfinite model after WWPGD readiness application")
    changed = sum(bool(row.get("changed")) for row in rows)
    movement = [
        float(row.get("relative_frobenius_change_applied", 0.0) or 0.0)
        for row in rows
        if math.isfinite(float(row.get("relative_frobenius_change_applied", 0.0) or 0.0))
    ]
    return {
        "level": level,
        "optimizer": optimizer_name,
        "device": str(device),
        "loss": float(loss.detach().cpu()),
        "base_optimizer_finite": True,
        "eligible_matrix_count": len(eligible_names),
        "stock_candidate_changed_count": sum(candidate.stock_candidate_changed.values()),
        "applied_changed_count": changed,
        "maximum_readiness_relative_change": max(movement, default=0.0),
        "wwpgd_diagnostic_rows": len(candidate.internal_diagnostics),
        "wwpgd_native_internal_diagnostics": bool(
            any(str(row.get("diagnostics_mode")) == "native" for row in candidate.internal_diagnostics)
        ),
        "wwpgd_candidate_seconds": candidate_seconds,
        "resolved_llrd_gamma": llrd_gamma,
        "optimizer_fingerprint": json.dumps(optimizer_fingerprint(bundle), sort_keys=True, default=str),
        "status": "PASS",
        "error": "",
    }


def _layer_lr_check(level: int, config_path: Path, device: torch.device) -> list[dict[str, Any]]:
    cfg = load_config(config_path, level)
    rows: list[dict[str, Any]] = []
    for mode in ("flat", "llrd", "manual"):
        model = GPT(cfg.model).to(device)
        train = replace(cfg.train, layer_lr=mode)
        bundle, gamma = build_optimizer_bundle(model, train, "adamw")
        groups = [group for optimizer in bundle.optimizers for group in optimizer.param_groups]
        peaks = [float(group["peak_lr"]) for group in groups]
        if not peaks or not all(math.isfinite(value) and value > 0 for value in peaks):
            raise RuntimeError(f"invalid parameter-group learning rates for {mode}")
        if mode == "flat" and max(peaks) != min(peaks):
            raise RuntimeError("flat learning-rate mode produced unequal peak learning rates")
        if mode == "llrd" and level > 0 and not max(peaks) > min(peaks):
            raise RuntimeError("LLRD did not produce depth-dependent learning rates")
        rows.append(
            {
                "level": level,
                "layer_lr_mode": mode,
                "parameter_group_count": len(groups),
                "minimum_peak_lr": min(peaks),
                "maximum_peak_lr": max(peaks),
                "resolved_llrd_gamma": gamma,
                "status": "PASS",
            }
        )
    return rows


def run_readiness_check(
    output: Path,
    *,
    device: str = "auto",
    levels: Iterable[int] | str = (0, 1, 2),
    optimizers: Iterable[str] | str = ("adamw", "muon", "stableadamw"),
    precision: str | None = None,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    levels = _parse_ints(levels)
    optimizers = _parse_strings(optimizers)
    failures: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    lr_rows: list[dict[str, Any]] = []
    for level in levels:
        config_path = Path(f"configs/level{level}_adaptive_alpha.yaml")
        cfg = load_config(config_path, level)
        report = GPT(cfg.model).parameter_report()
        count = selected_parameter_count(report, cfg.parameter_count_convention)
        budget = plan_budget(
            count,
            20,
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
        schedule_rows.append(
            {
                "level": level,
                "config": str(config_path),
                "parameter_count": count,
                "optimizer_steps": budget.steps,
                "target_alpha": cfg.wwpgd.target_alpha,
                "measurement_interval": cfg.measurement.alpha_interval,
                **schedule,
            }
        )
        try:
            lr_rows.extend(_layer_lr_check(level, config_path, resolved_device))
        except Exception as exc:
            failures.append({"level": level, "component": "layer_learning_rates", "error": f"{type(exc).__name__}: {exc}"})
        for optimizer_name in optimizers:
            try:
                optimizer_rows.append(
                    _one_optimizer_and_wwpgd_check(
                        level=level,
                        optimizer_name=optimizer_name,
                        config_path=config_path,
                        device=resolved_device,
                        precision=precision,
                    )
                )
            except Exception as exc:
                record = {
                    "level": level,
                    "optimizer": optimizer_name,
                    "device": str(resolved_device),
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                optimizer_rows.append(record)
                failures.append(record)
    package = resolve_pip_wwpgd_provenance()
    report = {
        "status": "PASS" if not failures else "ERROR",
        "ready": not failures,
        "device": device_summary(device),
        "precision": {key: value for key, value in precision_policy(resolved_device, precision).items() if key != "torch_dtype"},
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "package_provenance": package,
        "levels": levels,
        "optimizers": optimizers,
        "failures": failures,
        "adaptive_schedules": schedule_rows,
        "optimizer_checks": optimizer_rows,
        "layer_learning_rate_checks": lr_rows,
        "scientific_result": False,
        "note": "Readiness uses synthetic tokens and is not evidence of WWPGD efficacy.",
    }
    (output / "readiness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    pd.DataFrame(optimizer_rows).to_csv(output / "readiness_optimizer_checks.csv", index=False)
    pd.DataFrame(schedule_rows).to_csv(output / "readiness_level_schedules.csv", index=False)
    pd.DataFrame(lr_rows).to_csv(output / "readiness_layer_lr_checks.csv", index=False)
    return report
