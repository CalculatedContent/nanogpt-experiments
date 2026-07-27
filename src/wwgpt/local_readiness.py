from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from wwgpt.adaptive_wwpgd import validate_adaptive_level_schedule
from wwgpt.config import load_config
from wwgpt.device import device_summary, resolve_device, synchronize_device
from wwgpt.model import GPT
from wwgpt.optim import (
    apply_lr_schedule,
    build_optimizer_bundle,
    optimizer_fingerprint,
    resolve_learning_rates,
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


def _memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size
    except (AttributeError, OSError, ValueError):
        return None


def _level_config(level: int) -> Path:
    return Path(f"configs/level{level}_adaptive_alpha.yaml")


def estimate_level(level: int, token_multiplier: int = 20) -> dict[str, Any]:
    cfg = load_config(_level_config(level), level)
    model = GPT(cfg.model)
    report = model.parameter_report()
    count = selected_parameter_count(report, cfg.parameter_count_convention)
    budget = plan_budget(
        count, token_multiplier, cfg.train.batch_size, cfg.model.block_size,
        cfg.train.gradient_accumulation, 10**18
    )
    schedule = validate_adaptive_level_schedule(
        cfg.wwpgd.adaptive, budget.steps, cfg.measurement.alpha_interval
    )
    evaluation_events = math.ceil(budget.steps / cfg.train.eval_interval)
    return {
        "level": level,
        "config": str(_level_config(level)),
        "model": asdict(cfg.model),
        "parameter_report": asdict(report),
        "selected_parameter_count": count,
        "optimizer_steps_per_arm": budget.steps,
        "tokens_per_arm": budget.realized_tokens,
        "tokens_per_optimizer_step": (
            cfg.train.batch_size * cfg.train.gradient_accumulation * cfg.model.block_size
        ),
        "evaluation_events_per_arm": evaluation_events,
        "evaluation_tokens_per_arm": (
            evaluation_events * cfg.train.eval_batches * cfg.train.batch_size
            * cfg.model.block_size
        ),
        "alpha_measurement_interval": cfg.measurement.alpha_interval,
        "trap_measurement_interval": cfg.measurement.trap_diagnostic_interval,
        "adaptive_schedule": schedule,
        "learning_rate_resolution": resolve_learning_rates(cfg.train, cfg.model.block_size),
        "candidate_device": cfg.wwpgd.candidate_device,
    }


def _protected_state(model: GPT) -> dict[str, torch.Tensor]:
    eligible = {
        f"{name}.weight"
        for name, module in model.named_modules()
        if is_projected_layer(name) and getattr(module, "weight", None) is not None
    }
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name not in eligible
    }


def _assert_protected(model: GPT, before: dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    changed = [
        name for name, value in before.items()
        if name not in state or not torch.equal(value, state[name].detach().cpu())
    ]
    if changed:
        raise RuntimeError(f"WWPGD changed protected tensors: {changed}")


def run_real_package_smoke(level: int, optimizer_name: str, device: str) -> dict[str, Any]:
    cfg = load_config(_level_config(level), level)
    model_cfg = replace(cfg.model, block_size=16, vocab_size=256)
    train_cfg = replace(
        cfg.train, batch_size=1, gradient_accumulation=1, max_steps=2,
        eval_batches=1, lr_scale_rule="fixed"
    )
    resolved_device = resolve_device(device)
    torch.manual_seed(1000 + level)
    model = GPT(model_cfg).to(resolved_device)
    bundle, _ = build_optimizer_bundle(model, train_cfg, optimizer_name)
    warmup = resolve_warmup_steps(2, train_cfg.warmup_ratio, train_cfg.warmup_steps)
    x = torch.randint(0, model_cfg.vocab_size, (1, model_cfg.block_size), device=resolved_device)
    y = torch.randint(0, model_cfg.vocab_size, (1, model_cfg.block_size), device=resolved_device)
    _, loss = model(x, y)
    assert loss is not None
    loss.backward()
    apply_lr_schedule(bundle, 0, 2, warmup, train_cfg)
    for optimizer in bundle.optimizers:
        optimizer.step()
    bundle.zero_grad()
    synchronize_device(resolved_device)
    protected = _protected_state(model)
    ww_cfg = replace(cfg.wwpgd, candidate_device="auto")
    candidate = build_stock_wwpgd_candidate(
        model, event_index=0, actual_step=1,
        cfg=external_wwpgd_config_from_experiment(ww_cfg),
    )
    hardness = {name: 0.02 for name in candidate.original_weights}
    rows = apply_external_wwpgd(
        model, event_index=0, actual_step=1, actual_tokens_seen=model_cfg.block_size,
        cfg=external_wwpgd_config_from_experiment(ww_cfg),
        layer_hardness=hardness, global_event_hardness=1.0,
        stock_candidate=candidate,
    )
    synchronize_device(resolved_device)
    _assert_protected(model, protected)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("nonfinite model parameter after WWPGD smoke")
    return {
        "level": level,
        "optimizer": optimizer_name,
        "device": str(resolved_device),
        "loss": float(loss.detach().cpu()),
        "wwpgd_rows": len(rows),
        "stock_candidate_changed_layers": int(sum(candidate.stock_candidate_changed.values())),
        "applied_changed_layers": int(sum(bool(row.get("changed")) for row in rows)),
        "candidate_execution_device": candidate.candidate_execution_device,
        "candidate_offloaded": candidate.candidate_offloaded,
        "diagnostics_mode": sorted({
            str(row.get("diagnostics_mode", "unknown"))
            for row in candidate.internal_diagnostics
        }),
        "optimizer_fingerprint": optimizer_fingerprint(bundle),
        "finite": True,
    }


def run_local_readiness(
    output: Path,
    device: str = "auto",
    levels: list[int] | None = None,
    optimizers: list[str] | None = None,
    run_package_smoke: bool = True,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    levels = levels or [0, 1, 2]
    optimizers = optimizers or ["adamw", "stableadamw", "muon"]
    findings: list[dict[str, str]] = []
    started = time.perf_counter()
    try:
        resolved = device_summary(device)
    except Exception as exc:
        resolved = {"requested_device": device, "error": str(exc)}
        findings.append({"severity": "ERROR", "check": "device", "message": str(exc)})
    estimates = [estimate_level(level) for level in levels]
    disk = shutil.disk_usage(output)
    memory = _memory_bytes()
    if disk.free < 10 * 1024**3:
        findings.append({
            "severity": "WARNING", "check": "disk",
            "message": f"less than 10 GiB free at {output}",
        })
    smokes: list[dict[str, Any]] = []
    if run_package_smoke and not any(row["severity"] == "ERROR" for row in findings):
        for level in levels:
            for optimizer in optimizers:
                try:
                    smokes.append(run_real_package_smoke(level, optimizer, device))
                except Exception as exc:
                    findings.append({
                        "severity": "ERROR",
                        "check": f"level_{level}_{optimizer}_real_package_smoke",
                        "message": f"{type(exc).__name__}: {exc}",
                    })
    report = {
        "ready": not any(row["severity"] == "ERROR" for row in findings),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": resolved,
        "package_provenance": resolve_pip_wwpgd_provenance(),
        "physical_memory_bytes": memory,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "levels": estimates,
        "real_package_smokes": smokes,
        "findings": findings,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "local_readiness.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    return report
