from __future__ import annotations

import math
import time
from dataclasses import dataclass
from importlib import metadata
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

WWPGD_COMMIT = "bf970cb6b73e977f8374114c442ae5b0589eccaa"
SCIENTIFIC_SCHEMA_VERSION = 2
PROJECTED_LAYER_SUFFIXES = ("attn.c_attn", "attn.c_proj", "mlp.0", "mlp.2")


def matrix_modules(model: nn.Module, include_tied_once: bool = True):
    seen: set[int] = set()
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is not None and weight.ndim == 2:
            if include_tied_once and id(weight) in seen:
                continue
            seen.add(id(weight))
            yield name or "root", weight


def is_projected_layer(name: str) -> bool:
    return name.startswith("blocks.") and name.endswith(PROJECTED_LAYER_SUFFIXES)


def projected_matrix_modules(model: nn.Module):
    for name, weight in matrix_modules(model):
        if is_projected_layer(name):
            yield name, weight


def _ww_version() -> str:
    try:
        return metadata.version("weightwatcher")
    except Exception:
        import weightwatcher as ww
        return getattr(ww, "__version__", "unknown")


def weightwatcher_details(model: nn.Module) -> pd.DataFrame:
    import weightwatcher
    start = time.perf_counter()
    watcher = weightwatcher.WeightWatcher(model=model)
    details = watcher.analyze(detX=True, randomize=False, plot=False)
    if details is None:
        raise RuntimeError("WeightWatcher.analyze returned None")
    df = details.copy()
    df["analysis_runtime"] = time.perf_counter() - start
    df["weightwatcher_version"] = _ww_version()
    df["spectral_estimator"] = "weightwatcher"
    df["spectral_estimator_version"] = df["weightwatcher_version"]
    df["weightwatcher_configuration"] = '{"detX": true, "randomize": false, "plot": false}'
    df["valid_for_science"] = True
    return df


def spectral_summary(model: nn.Module, *, step: int, tokens_seen: int, optimizer: str, seed: int, pair_id: str) -> list[dict[str, object]]:
    df = weightwatcher_details(model)
    df["step"] = step; df["tokens_seen"] = tokens_seen; df["optimizer"] = optimizer; df["seed"] = seed; df["pair_id"] = pair_id
    return df.to_dict("records")


def fallback_spectral_summary(model: nn.Module, *, step: int = 0, tokens_seen: int = 0, optimizer: str = "smoke", seed: int = 0, pair_id: str = "smoke") -> list[dict[str, object]]:
    rows=[]
    for lid,(name,w) in enumerate(matrix_modules(model)):
        s=torch.linalg.svdvals(w.detach().float().cpu()); eig=(s*s).numpy()
        rows.append({"layer_id":lid,"name":name,"longname":name,"num_evals":len(eig),"spectral_norm":float(s.max()) if len(s) else 0.0,"stable_rank":float(eig.sum()/(eig.max()+1e-12)) if len(eig) else 0.0,"step":step,"tokens_seen":tokens_seen,"optimizer":optimizer,"seed":seed,"pair_id":pair_id,"analysis_runtime":0.0,"weightwatcher_version":"","spectral_estimator":"fallback_non_scientific","spectral_estimator_version":"","valid_for_science":False,"warning":"smoke-test fallback; not WeightWatcher alpha"})
    return rows


@dataclass
class WWTailConfig:
    min_tail: int = 5
    q: float = 1.0
    blend_eta: float = 0.5
    cayley_eta: float = 0.25
    use_detx: bool = True
    warmup_events: int = 0
    ramp_events: int = 5


def projection_hardness(event_index: int, cfg: WWTailConfig) -> float:
    if event_index < cfg.warmup_events: return 0.0
    if event_index >= cfg.warmup_events + cfg.ramp_events: return 1.0
    return max(0.0, min(1.0, (event_index - cfg.warmup_events + 1) / max(cfg.ramp_events, 1)))


def _cayley(lam_current: torch.Tensor, lam_target: torch.Tensor, eta: float) -> torch.Tensor:
    if eta <= 0.0: return lam_current
    g=torch.log(lam_current+1e-8)-torch.log(lam_target+1e-8)
    return lam_current*torch.clamp((1-eta*g)/(1+eta*g),0.1,10.0)


def _resolve_module(model: nn.Module, lname: str) -> nn.Module | None:
    cur: nn.Module = model
    for part in lname.split("."):
        if not hasattr(cur, part): return None
        cur=getattr(cur, part)
    return cur if hasattr(cur,"weight") else None


def apply_wwpgd_reference(model: nn.Module, *, details: pd.DataFrame | None = None, event_index: int = 0, scheduled_token_fraction: float = 0.0, actual_step: int = 0, actual_tokens_seen: int = 0, strength: float = 1.0, cfg: WWTailConfig | None = None) -> list[dict[str, object]]:
    cfg = cfg or WWTailConfig()
    hardness = projection_hardness(event_index, cfg) * strength
    if details is None:
        details = weightwatcher_details(model)
    rows=[]; key="longname" if "longname" in details.columns else "name"
    for _,row in details.iterrows():
        lname=str(row.get(key, row.get("name", "")))
        if not is_projected_layer(lname):
            continue
        mod=_resolve_module(model,lname); start=time.perf_counter(); reason=""; changed=False; rel=0.0; tail_size=0; tl_before=float("nan"); tl_after=float("nan")
        xmin=float(row.get("xmin", float("nan"))) if pd.notna(row.get("xmin", float("nan"))) else float("nan")
        detx_num=int(row.get("detX_num")) if "detX_num" in row and pd.notna(row.get("detX_num")) else None
        if mod is None or not math.isfinite(xmin) or xmin <= 0 or hardness <= 0:
            reason = "no_module_or_invalid_xmin_or_zero_strength"
        else:
            with torch.no_grad():
                W=mod.weight.data; old=W.detach().clone(); W2=W.reshape(W.size(0),-1).float(); U,S,Vh=torch.linalg.svd(W2, full_matrices=False); lam=(S.clamp_min(1e-8)**2); n=lam.numel()
                mask=lam >= xmin; tail_size=int(mask.sum().item())
                if tail_size < cfg.min_tail:
                    reason=f"insufficient_tail_size:{tail_size}"
                else:
                    lam_tail=lam[mask]; tl_before=float(torch.log(lam_tail+1e-8).sum().item())
                    r=torch.arange(1,tail_size+1,device=lam.device,dtype=torch.float32); mu=r.pow(-cfg.q)
                    A=torch.exp((torch.log(lam_tail+1e-8).sum()-torch.log(mu).sum())/tail_size)
                    target=A*mu; new_tail=_cayley(lam_tail,target,hardness*cfg.cayley_eta)
                    new_tail=new_tail*torch.exp((torch.log(lam_tail+1e-8).sum()-torch.log(new_tail+1e-8).sum())/tail_size)
                    Snew=S.clone(); Snew[mask]=torch.sqrt(new_tail.clamp_min(1e-8)); shaped=(U*Snew.unsqueeze(0))@Vh
                    blend=hardness*cfg.blend_eta
                    Wnew=(1-blend)*W2 + blend*shaped
                    S_after=(1-blend)*S + blend*Snew
                    W.copy_(Wnew.reshape_as(W).to(device=W.device,dtype=W.dtype))
                    rel=float((torch.linalg.norm(W-old)/(torch.linalg.norm(old)+1e-12)).item())
                    changed=rel>0
                    tl_after=float(torch.log(S_after.square()[mask]+1e-8).sum().item())
        rows.append({"projection_event":event_index,"scheduled_token_fraction":scheduled_token_fraction,"actual_step":actual_step,"actual_tokens_seen":actual_tokens_seen,"layer_name":lname,"hardness":hardness,"projection_runtime":time.perf_counter()-start,"changed":changed,"skip_reason":reason,"relative_frobenius_change":rel,"relative_frobenius_weight_change":rel,"xmin":xmin,"detX_num":detx_num,"tail_size":tail_size,"TraceLog_before":tl_before,"TraceLog_after":tl_after,"wwpgd_implementation":"reference","wwpgd_commit":WWPGD_COMMIT})
    return rows


def apply_wwpgd(model: nn.Module, target_alpha: float, strength: float, step: int, warmup_steps: int = 0, ramp_steps: int = 1):
    return apply_wwpgd_reference(model, event_index=step, actual_step=step, strength=strength, cfg=WWTailConfig(warmup_events=warmup_steps, ramp_events=ramp_steps))