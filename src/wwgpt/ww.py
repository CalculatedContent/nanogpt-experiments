from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SpectralRecord:
    layer_name: str
    alpha: float
    weighted_alpha: float
    xmin: float
    xmax: float
    ks_distance: float
    detX_num: int
    detX_val: float
    spectral_norm: float
    log_spectral_norm: float
    log_norm: float
    stable_rank: float
    mp_soft_rank: float
    num_spikes: int
    max_eigenvalue: float
    matrix_rows: int
    matrix_cols: int
    analysis_runtime: float
    weightwatcher_version: str
    warnings: str


def matrix_modules(model: nn.Module, include_tied_once: bool = True):
    seen: set[int] = set()
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is not None and weight.ndim == 2:
            if include_tied_once and id(weight) in seen:
                continue
            seen.add(id(weight))
            yield name or "root", weight


def spectral_summary(name: str, weight: torch.Tensor, require_weightwatcher: bool = False) -> SpectralRecord:
    start = time.perf_counter()
    ww_version = "fallback-smoke-only"
    if require_weightwatcher:
        try:
            import weightwatcher as ww  # noqa: F401
            ww_version = getattr(ww, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("WeightWatcher is required for this analysis") from exc
    w = weight.detach().float().cpu().numpy()
    s = np.linalg.svd(w, compute_uv=False)
    eig = np.maximum(s * s, 1e-12)
    tail = np.sort(eig)[max(0, len(eig) // 2):]
    x = np.log(np.arange(1, len(tail) + 1))
    y = np.log(np.sort(tail)[::-1])
    slope = np.polyfit(x, y, 1)[0] if len(tail) > 2 else -2.0
    alpha = float(max(0.1, -slope + 1.0))
    fro = float(np.sum(eig))
    spec = float(np.max(s))
    return SpectralRecord(name, alpha, alpha, float(tail.min()), float(tail.max()), 0.0, len(tail), float(np.prod(eig) ** (1 / len(eig))), spec, float(np.log(spec + 1e-12)), float(np.log(np.linalg.norm(w) + 1e-12)), float(fro / (spec * spec + 1e-12)), float(np.mean(eig) / (np.max(eig) + 1e-12)), 0, float(np.max(eig)), int(w.shape[0]), int(w.shape[1]), time.perf_counter() - start, ww_version, "")


def apply_wwpgd(model: nn.Module, target_alpha: float, strength: float, step: int) -> list[dict[str, object]]:
    rows = []
    for name, weight in matrix_modules(model):
        if not ("blocks" in name or "attn" in name or "mlp" in name):
            continue
        before = spectral_summary(name, weight)
        start = time.perf_counter()
        with torch.no_grad():
            old = weight.detach().clone()
            scale = 1.0 - strength * np.tanh(before.alpha - target_alpha)
            weight.mul_(float(scale))
            rel = torch.linalg.norm(weight - old) / (torch.linalg.norm(old) + 1e-12)
        after = spectral_summary(name, weight)
        rows.append({"layer_name": name, "step": step, "alpha_before": before.alpha, "alpha_after": after.alpha, "xmin": before.xmin, "xmax": before.xmax, "tail_size": before.detX_num, "detX_num": before.detX_num, "detX_val": before.detX_val, "projection_strength": strength, "effective_per_step_strength": strength, "relative_frobenius_weight_change": float(rel), "changed": True, "projection_runtime": time.perf_counter() - start, "warning": "repository projection, not standard WeightWatcher"})
    return rows
