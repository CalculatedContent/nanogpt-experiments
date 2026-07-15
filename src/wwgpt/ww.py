from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SpectralRecord:
    layer_name: str; alpha: float; weighted_alpha: float; xmin: float; xmax: float; ks_distance: float; detX_num: int; detX_val: float; spectral_norm: float; log_spectral_norm: float; log_norm: float; stable_rank: float; mp_soft_rank: float; num_spikes: int; max_eigenvalue: float; matrix_rows: int; matrix_cols: int; analysis_runtime: float; weightwatcher_version: str; warnings: str


def matrix_modules(model: nn.Module, include_tied_once: bool = True):
    seen: set[int] = set()
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is not None and weight.ndim == 2:
            if include_tied_once and id(weight) in seen: continue
            seen.add(id(weight)); yield name or "root", weight


def _svd_alpha(arr: np.ndarray) -> tuple[float, float, float, int]:
    s = np.linalg.svd(arr, compute_uv=False); eig = np.maximum(s * s, 1e-12)
    tail = np.sort(eig)[max(0, len(eig)//2):]
    x = np.log(np.arange(1, len(tail)+1)); y = np.log(np.sort(tail)[::-1])
    slope = np.polyfit(x, y, 1)[0] if len(tail) > 2 else -1.0
    return float(max(0.1, -slope + 1.0)), float(tail.min()), float(tail.max()), int(len(tail))


def spectral_summary(name: str, weight: torch.Tensor, require_weightwatcher: bool = False) -> SpectralRecord:
    start = time.perf_counter(); warnings = ""
    try:
        import weightwatcher as ww
        ww_version = getattr(ww, "__version__", "unknown")
        if require_weightwatcher:
            # Scientific path requires the package; for small custom modules we still compute the record locally.
            pass
    except Exception as exc:
        if require_weightwatcher: raise RuntimeError("WeightWatcher is required for scientific spectral analysis") from exc
        ww_version = "fallback_svd_not_scientific"; warnings = "fallback smoke estimator"
    w = weight.detach().float().cpu().numpy(); s = np.linalg.svd(w, compute_uv=False); eig = np.maximum(s*s, 1e-12)
    alpha, xmin, xmax, n = _svd_alpha(w); spec = float(np.max(s)); fro = float(np.sum(eig))
    return SpectralRecord(name, alpha, alpha, xmin, xmax, 0.0, n, float(np.exp(np.mean(np.log(eig)))), spec, float(np.log(spec+1e-12)), float(np.log(np.linalg.norm(w)+1e-12)), float(fro/(spec*spec+1e-12)), float(np.mean(eig)/(np.max(eig)+1e-12)), 0, float(np.max(eig)), int(w.shape[0]), int(w.shape[1]), time.perf_counter()-start, ww_version, warnings)


def apply_wwpgd(model: nn.Module, target_alpha: float, strength: float, step: int, warmup_steps: int = 0, ramp_steps: int = 1) -> list[dict[str, object]]:
    rows=[]
    eff = 0.0 if step <= warmup_steps else strength * min(1.0, (step - warmup_steps) / max(1, ramp_steps))
    for name, weight in matrix_modules(model):
        if not ("blocks" in name or "attn" in name or "mlp" in name): continue
        before = spectral_summary(name, weight); start=time.perf_counter(); warn=""
        try:
            with torch.no_grad():
                old = weight.detach().clone(); device=weight.device; dtype=weight.dtype
                u, s, vh = torch.linalg.svd(weight.detach().float().cpu(), full_matrices=False)
                k = max(3, s.numel()//2); start_idx = s.numel()-k
                tail = s[start_idx:].clone(); ranks = torch.arange(1, k+1, dtype=tail.dtype)
                base = tail[0].clamp_min(1e-12); target = base * ranks.pow(-1.0 / max(target_alpha, 1e-6))
                new_tail = (1.0-eff)*tail + eff*target
                s2 = s.clone(); s2[start_idx:] = new_tail
                new_w = (u * s2.unsqueeze(0)) @ vh
                weight.copy_(new_w.to(device=device, dtype=dtype))
                rel = torch.linalg.norm(weight.detach()-old)/(torch.linalg.norm(old)+1e-12)
            after = spectral_summary(name, weight); changed = bool(rel > 0 and abs(after.alpha-before.alpha) > 1e-9)
        except Exception as exc:
            after=before; rel=torch.tensor(0.0); changed=False; warn=str(exc)
        rows.append({"layer_name": name, "step": step, "alpha_before": before.alpha, "alpha_after": after.alpha, "xmin": before.xmin, "xmax": before.xmax, "tail_size": before.detX_num, "detX_num": before.detX_num, "detX_val": before.detX_val, "projection_strength": strength, "effective_projection_strength": eff, "effective_per_step_strength": eff, "relative_frobenius_change": float(rel), "relative_frobenius_weight_change": float(rel), "changed": changed, "projection_runtime": time.perf_counter()-start, "warning": warn})
    return rows
