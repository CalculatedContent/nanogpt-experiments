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


def apply_wwpgd_reference(
    model: nn.Module,
    *,
    details: pd.DataFrame | None = None,
    event_index: int = 0,
    scheduled_token_fraction: float = 0.0,
    actual_step: int = 0,
    actual_tokens_seen: int = 0,
    strength: float = 1.0,
    cfg: WWTailConfig | None = None,
) -> list[dict[str, object]]:
    cfg = cfg or WWTailConfig()
    schedule_hardness = projection_hardness(event_index, cfg)
    effective_hardness = schedule_hardness * strength
    hardness = effective_hardness
    effective_cayley_eta = effective_hardness * cfg.cayley_eta
    effective_blend_eta = effective_hardness * cfg.blend_eta
    if details is None:
        details = weightwatcher_details(model)

    rows = []
    key = "longname" if "longname" in details.columns else "name"
    for _, row in details.iterrows():
        lname = str(row.get(key, row.get("name", "")))
        if not is_projected_layer(lname):
            continue

        mod = _resolve_module(model, lname)
        start = time.perf_counter()
        reason = ""
        changed = False
        rel = 0.0
        tail_size = 0
        tl_before = float("nan")
        tl_after = float("nan")
        xmin = float(row.get("xmin", float("nan"))) if pd.notna(row.get("xmin", float("nan"))) else float("nan")
        detx_num = int(row.get("detX_num")) if "detX_num" in row and pd.notna(row.get("detX_num")) else None

        if mod is None or not math.isfinite(xmin) or xmin <= 0 or hardness <= 0:
            reason = "no_module_or_invalid_xmin_or_zero_strength"
        else:
            with torch.no_grad():
                W = mod.weight.data
                old = W.detach().clone()
                W2 = W.reshape(W.size(0), -1).float()
                U, S, Vh = torch.linalg.svd(W2, full_matrices=False)
                lam = S.clamp_min(1e-8).square()
                n = int(lam.numel())
                powerlaw_tail_size = int((lam >= xmin).sum().item())
                detx_tail_size = int(detx_num) if detx_num is not None else 0
                lam_thr = float(xmin)
                if cfg.use_detx and detx_num is not None and detx_num > 0:
                    k_pl = max(cfg.min_tail, min(powerlaw_tail_size, n))
                    k_detx = max(cfg.min_tail, min(detx_tail_size, n))
                    k_star = max(1, min(n, int(0.5 * (k_pl + k_detx))))
                    lam_thr = max(float(xmin), float(lam[k_star - 1].detach().cpu()))
                mask = lam >= lam_thr
                tail_size = int(mask.sum().item())

                if tail_size < cfg.min_tail:
                    reason = f"insufficient_tail_size:{tail_size}"
                else:
                    lam_tail = lam[mask]
                    tl_before = float(torch.log(lam_tail + 1e-8).sum().item())
                    r = torch.arange(1, tail_size + 1, device=lam.device, dtype=torch.float32)
                    mu = r.pow(-cfg.q)
                    A = torch.exp(
                        (torch.log(lam_tail + 1e-8).sum() - torch.log(mu).sum())
                        / tail_size
                    )
                    target = A * mu
                    new_tail = _cayley(lam_tail, target, effective_cayley_eta)
                    new_tail = new_tail * torch.exp(
                        (
                            torch.log(lam_tail + 1e-8).sum()
                            - torch.log(new_tail + 1e-8).sum()
                        )
                        / tail_size
                    )
                    Snew = S.clone()
                    Snew[mask] = torch.sqrt(new_tail.clamp_min(1e-8))
                    shaped = (U * Snew.unsqueeze(0)) @ Vh
                    blend = effective_blend_eta

                    Wnew = (1.0 - blend) * W2 + blend * shaped
                    S_after = (1.0 - blend) * S + blend * Snew

                    W.copy_(
                        Wnew.reshape_as(W).to(
                            device=W.device,
                            dtype=W.dtype,
                        )
                    )

                    rel = float(
                        (
                            torch.linalg.norm(W - old)
                            / (torch.linalg.norm(old) + 1e-12)
                        ).item()
                    )
                    changed = rel > 0
                    tl_after = float(
                        torch.log(
                            S_after.square()[mask] + 1e-8
                        ).sum().item()
                    )

        rows.append(
            {
                "projection_event": event_index,
                "scheduled_token_fraction": scheduled_token_fraction,
                "actual_step": actual_step,
                "actual_tokens_seen": actual_tokens_seen,
                "layer_name": lname,
                "hardness": hardness,
                "schedule_hardness": schedule_hardness,
                "scan_strength": strength,
                "effective_hardness": effective_hardness,
                "effective_cayley_eta": effective_cayley_eta,
                "effective_blend_eta": effective_blend_eta,
                "projection_runtime": time.perf_counter() - start,
                "changed": changed,
                "skip_reason": reason,
                "relative_frobenius_change": rel,
                "relative_frobenius_weight_change": rel,
                "xmin": xmin,
                "detX_num": detx_num,
                "tail_size": tail_size,
                "powerlaw_tail_size": locals().get("powerlaw_tail_size", 0),
                "detx_tail_size": locals().get("detx_tail_size", 0),
                "selected_tail_size": tail_size,
                "selected_tail_threshold": locals().get("lam_thr", xmin),
                "TraceLog_before": tl_before,
                "TraceLog_after": tl_after,
                "wwpgd_implementation": "reference",
                "wwpgd_commit": WWPGD_COMMIT,
            }
        )
    return rows


def apply_wwpgd(model: nn.Module, target_alpha: float, strength: float, step: int, warmup_steps: int = 0, ramp_steps: int = 1):
    return apply_wwpgd_reference(model, event_index=step, actual_step=step, strength=strength, cfg=WWTailConfig(warmup_events=warmup_steps, ramp_events=ramp_steps))
MEASURED_PROJECTION_SPECTRAL_FIELDS = [
    'projection_event','scheduled_token_fraction','actual_step','actual_tokens_seen','layer_name','match_key','match_status',
    'target_alpha','alpha_before','alpha_after','weighted_alpha_before','weighted_alpha_after','xmin_before','xmin_after',
    'detX_num_before','detX_num_after','D_before','D_after','num_evals_before','num_evals_after','status_before','status_after',
    'warning_before','warning_after','pre_weightwatcher_runtime','post_weightwatcher_runtime','weightwatcher_version','weightwatcher_configuration',
    'TraceLog_before','TraceLog_after','alpha_delta','abs_alpha_error_before','abs_alpha_error_after','abs_alpha_error_change','TraceLog_change',
    'immediate_spectral_source','measurement_valid_for_science'
]

def _finite(v):
    try: return math.isfinite(float(v))
    except Exception: return False

def _nan_if_missing(row, key):
    try:
        v=row.get(key, float('nan'))
        return float(v) if pd.notna(v) else float('nan')
    except Exception:
        return float('nan')

def _status(row):
    for k in ('status','fit_status'):
        if k in row and pd.notna(row.get(k)): return str(row.get(k))
    return 'ok' if _finite(row.get('alpha', float('nan'))) else 'missing_alpha'

def _warning(row):
    for k in ('warning','warnings'):
        if k in row and pd.notna(row.get(k)): return str(row.get(k))
    return ''

def _index_details(df: pd.DataFrame, key: str):
    out={}
    if key in df.columns:
        for _, r in df.iterrows():
            v=r.get(key)
            if pd.notna(v): out[str(v)]=r
    return out

def measured_projection_spectral_rows(pre: pd.DataFrame, post: pd.DataFrame, projection_rows: list[dict[str,object]], target_alpha: float) -> list[dict[str,object]]:
    """Pair one real pre-event WW details table with one real post-event table.

    Alpha deltas and error changes are derived only from measured finite alpha fields.
    Missing fields remain NaN and invalidate the row for scientific WeightWatcher use.
    """
    pre_long=_index_details(pre,'longname'); post_long=_index_details(post,'longname')
    pre_name=_index_details(pre,'name'); post_name=_index_details(post,'name')
    rows=[]
    pre_rt=float(pre['analysis_runtime'].iloc[0]) if 'analysis_runtime' in pre.columns and len(pre) else float('nan')
    post_rt=float(post['analysis_runtime'].iloc[0]) if 'analysis_runtime' in post.columns and len(post) else float('nan')
    wwver=str(pre['weightwatcher_version'].iloc[0]) if 'weightwatcher_version' in pre.columns and len(pre) else 'unknown'
    wwcfg=str(pre['weightwatcher_configuration'].iloc[0]) if 'weightwatcher_configuration' in pre.columns and len(pre) else ''
    for pr in projection_rows:
        lname=str(pr.get('layer_name',''))
        before=pre_long.get(lname); after=post_long.get(lname); match_key='longname'
        if before is None or after is None:
            before=pre_name.get(lname); after=post_name.get(lname); match_key='name'
        matched=before is not None and after is not None
        b=before if before is not None else pd.Series(dtype=object); a=after if after is not None else pd.Series(dtype=object)
        ab=_nan_if_missing(b,'alpha'); aa=_nan_if_missing(a,'alpha')
        wab=_nan_if_missing(b,'weighted_alpha'); waa=_nan_if_missing(a,'weighted_alpha')
        tb=_nan_if_missing(b,'TraceLog'); ta=_nan_if_missing(a,'TraceLog')
        sb=_status(b); sa=_status(a)
        valid=bool(matched and _finite(ab) and _finite(aa) and sb.lower() in {'ok','success','valid'} and sa.lower() in {'ok','success','valid'})
        def sub(x,y): return float(x)-float(y) if _finite(x) and _finite(y) else float('nan')
        row={
            'projection_event':pr.get('projection_event'),'scheduled_token_fraction':pr.get('scheduled_token_fraction'),'actual_step':pr.get('actual_step'),'actual_tokens_seen':pr.get('actual_tokens_seen'),'layer_name':lname,
            'match_key':match_key if matched else 'unmatched','match_status':'matched' if matched else 'unmatched','target_alpha':target_alpha,
            'alpha_before':ab,'alpha_after':aa,'weighted_alpha_before':wab,'weighted_alpha_after':waa,'xmin_before':_nan_if_missing(b,'xmin'),'xmin_after':_nan_if_missing(a,'xmin'),
            'detX_num_before':_nan_if_missing(b,'detX_num'),'detX_num_after':_nan_if_missing(a,'detX_num'),'D_before':_nan_if_missing(b,'D'),'D_after':_nan_if_missing(a,'D'),
            'num_evals_before':_nan_if_missing(b,'num_evals'),'num_evals_after':_nan_if_missing(a,'num_evals'),'status_before':sb,'status_after':sa,'warning_before':_warning(b),'warning_after':_warning(a),
            'pre_weightwatcher_runtime':pre_rt,'post_weightwatcher_runtime':post_rt,'weightwatcher_version':wwver,'weightwatcher_configuration':wwcfg,
            'TraceLog_before':tb,'TraceLog_after':ta,
            'alpha_delta':sub(aa,ab),'abs_alpha_error_before':abs(ab-target_alpha) if _finite(ab) else float('nan'),'abs_alpha_error_after':abs(aa-target_alpha) if _finite(aa) else float('nan'),
            'abs_alpha_error_change':sub(abs(aa-target_alpha) if _finite(aa) else float('nan'), abs(ab-target_alpha) if _finite(ab) else float('nan')),
            'TraceLog_change':sub(ta,tb),'immediate_spectral_source':'weightwatcher_measured','measurement_valid_for_science':valid,
        }
        rows.append(row)
    return rows
