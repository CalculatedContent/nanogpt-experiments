from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from importlib import metadata
import numpy as np
import pandas as pd
import torch
from torch import nn

WWPGD_COMMIT = "bf970cb6b73e977f8374114c442ae5b0589eccaa"
SCIENTIFIC_SCHEMA_VERSION = 3
PROJECTED_LAYER_SUFFIXES = ("attn.key", "attn.query", "attn.value", "attn.proj", "mlp.0", "mlp.2")


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


def weightwatcher_details(model: nn.Module, *, randomize: bool = False) -> pd.DataFrame:
    import weightwatcher
    start = time.perf_counter()
    watcher = weightwatcher.WeightWatcher(model=model)
    details = watcher.analyze(detX=True, randomize=randomize, plot=False)
    if details is None:
        raise RuntimeError("WeightWatcher.analyze returned None")
    df = details.copy()
    df["analysis_runtime"] = time.perf_counter() - start
    df["weightwatcher_version"] = _ww_version()
    df["spectral_estimator"] = "weightwatcher"
    df["spectral_estimator_version"] = df["weightwatcher_version"]
    df["weightwatcher_configuration"] = json.dumps({"detX": True, "randomize": bool(randomize), "plot": False}, sort_keys=True)
    df["valid_for_science"] = True
    return df


WW_DIAGNOSTIC_FIELDS = (
    "layer_id",
    "name",
    "longname",
    "matrix_shape",
    "alpha",
    "spectral_norm",
    "stable_rank",
    "matrix_rank",
    "ww_softrank",
    "rand_mp_softrank",
    "rand_num_spikes",
    "num_traps",
    "num_pl_spikes",
    "num_ERG_spikes",
    "trap_flag",
    "trap_rule",
    "unsupported_field_explanation",
)


def _null_explanation(field: str) -> str:
    return f"not returned by installed WeightWatcher {_ww_version()} for this analysis"


def add_weightwatcher_diagnostic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported installed-WeightWatcher diagnostics without computing them."""
    out = df.copy()
    if "longname" not in out.columns:
        out["longname"] = out["name"] if "name" in out.columns else pd.NA
    if "matrix_shape" not in out.columns:
        if {"M", "N"}.issubset(out.columns):
            out["matrix_shape"] = [json.dumps([None if pd.isna(m) else int(m), None if pd.isna(n) else int(n)]) for m, n in zip(out["M"], out["N"], strict=False)]
        else:
            out["matrix_shape"] = pd.NA
    supported_trap_metrics = [c for c in ("num_traps", "rand_num_spikes", "num_pl_spikes", "num_ERG_spikes") if c in out.columns]
    if "trap_flag" not in out.columns:
        if "num_traps" in out.columns:
            out["trap_flag"] = out["num_traps"].fillna(0).astype(float) > 0
            rule = "WeightWatcher randomize=True; trap_flag is num_traps > 0"
        elif "rand_num_spikes" in out.columns:
            out["trap_flag"] = out["rand_num_spikes"].fillna(0).astype(float) > 0
            rule = "WeightWatcher randomize=True; trap_flag is rand_num_spikes > 0"
        else:
            out["trap_flag"] = pd.NA
            rule = "unsupported: installed WeightWatcher returned no num_traps or rand_num_spikes column"
        out["trap_rule"] = rule
    if "unsupported_field_explanation" not in out.columns:
        missing = [field for field in WW_DIAGNOSTIC_FIELDS if field not in out.columns]
        out["unsupported_field_explanation"] = "; ".join(_null_explanation(f) for f in missing) if missing else ""
    for field in WW_DIAGNOSTIC_FIELDS:
        if field not in out.columns:
            out[field] = pd.NA
    out["trap_metric_columns"] = ",".join(supported_trap_metrics)
    return out


def spectral_summary(model: nn.Module, *, step: int, tokens_seen: int, optimizer: str, seed: int, pair_id: str) -> list[dict[str, object]]:
    df = add_weightwatcher_diagnostic_fields(weightwatcher_details(model, randomize=True))
    df["step"] = step; df["tokens_seen"] = tokens_seen; df["optimizer"] = optimizer; df["seed"] = seed; df["pair_id"] = pair_id
    return df.to_dict("records")


def weightwatcher_run_aggregates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out=[]
    for (step, tokens_seen, optimizer, seed, pair_id), g in df.groupby(["step", "tokens_seen", "optimizer", "seed", "pair_id"], dropna=False):
        traps = pd.to_numeric(g.get("trap_flag", pd.Series(dtype=float)), errors="coerce")
        alpha = pd.to_numeric(g.get("alpha", pd.Series(dtype=float)), errors="coerce")
        spectral = pd.to_numeric(g.get("spectral_norm", pd.Series(dtype=float)), errors="coerce")
        sr = pd.to_numeric(g.get("stable_rank", pd.Series(dtype=float)), errors="coerce")
        out.append({"step": step, "tokens_seen": tokens_seen, "optimizer": optimizer, "seed": seed, "pair_id": pair_id, "eligible_layer_count": int(len(g)), "mean_alpha": float(alpha.mean()) if alpha.notna().any() else math.nan, "median_alpha": float(alpha.median()) if alpha.notna().any() else math.nan, "mean_spectral_norm": float(spectral.mean()) if spectral.notna().any() else math.nan, "mean_stable_rank": float(sr.mean()) if sr.notna().any() else math.nan, "trap_layer_count": int(traps.fillna(False).astype(bool).sum()) if len(traps) else 0, "trap_layer_fraction": float(traps.fillna(False).astype(bool).mean()) if len(traps) else math.nan, "weightwatcher_version": _ww_version(), "weightwatcher_configuration": json.dumps({"detX": True, "randomize": True, "plot": False}, sort_keys=True)})
    return out


def fallback_spectral_summary(model: nn.Module, *, step: int = 0, tokens_seen: int = 0, optimizer: str = "smoke", seed: int = 0, pair_id: str = "smoke") -> list[dict[str, object]]:
    rows=[]
    for lid,(name,w) in enumerate(matrix_modules(model)):
        gram=w.detach().float().cpu() @ w.detach().float().cpu().T; eig=torch.linalg.eigvalsh(gram).clamp_min(0).numpy()
        rows.append({"layer_id":lid,"name":name,"longname":name,"num_evals":len(eig),"spectral_norm":float(eig.max() ** 0.5) if len(eig) else 0.0,"stable_rank":float(eig.sum()/(eig.max()+1e-12)) if len(eig) else 0.0,"step":step,"tokens_seen":tokens_seen,"optimizer":optimizer,"seed":seed,"pair_id":pair_id,"analysis_runtime":0.0,"weightwatcher_version":"","spectral_estimator":"fallback_non_scientific","spectral_estimator_version":"","valid_for_science":False,"warning":"smoke-test fallback; not WeightWatcher alpha"})
    return rows


@dataclass(frozen=True)
class ExternalWWTailConfigSpec:
    enable_tail_pgd: bool = True
    q: float = 1.0
    blend_eta: float = 0.5
    cayley_eta: float = 0.25
    min_tail: int = 5
    use_detx: bool = True
    warmup_epochs: int = 0
    ramp_epochs: int = 0
    verbose: bool = False


WWTailConfig = ExternalWWTailConfigSpec


STANDARD_WWPGD_BLEND_ETA = 0.5
STANDARD_WWPGD_WARMUP_EVENTS = 0
STANDARD_WWPGD_RAMP_EVENTS = 0


def external_wwpgd_config_from_experiment(cfg: object) -> ExternalWWTailConfigSpec:
    return ExternalWWTailConfigSpec(
        enable_tail_pgd=True,
        q=float(getattr(cfg, "q")),
        blend_eta=STANDARD_WWPGD_BLEND_ETA,
        cayley_eta=float(getattr(cfg, "cayley_eta")),
        min_tail=int(getattr(cfg, "min_tail")),
        use_detx=bool(getattr(cfg, "use_detx")),
        warmup_epochs=STANDARD_WWPGD_WARMUP_EVENTS,
        ramp_epochs=STANDARD_WWPGD_RAMP_EVENTS,
        verbose=bool(getattr(cfg, "verbose", False)),
    )


def resolved_external_wwpgd_config() -> ExternalWWTailConfigSpec:
    # Deprecated compatibility shim. New code should pass the resolved experiment
    # WWPGDConfig through external_wwpgd_config_from_experiment().
    return ExternalWWTailConfigSpec()


def external_wwpgd_manifest_fields(enabled: bool = True, requested_cfg: object | None = None) -> dict[str, object]:
    if not enabled:
        return {
            "wwpgd_package": "",
            "wwpgd_source_repository": "",
            "wwpgd_commit": "",
            "wwpgd_implementation": "none",
        }
    cfg = external_wwpgd_config_from_experiment(requested_cfg) if requested_cfg is not None else resolved_external_wwpgd_config()
    requested = dict(vars(requested_cfg)) if requested_cfg is not None and hasattr(requested_cfg, "__dataclass_fields__") else (dict(vars(requested_cfg)) if requested_cfg is not None and hasattr(requested_cfg, "__dict__") else {})
    resolved = dict(vars(cfg))
    return {
        "wwpgd_package": "ww_pgd",
        "wwpgd_source_repository": "CalculatedContent/WW_PGD",
        "wwpgd_commit": WWPGD_COMMIT,
        "wwpgd_implementation": "ww_pgd",
        "q": cfg.q,
        "blend_eta": cfg.blend_eta,
        "cayley_eta": cfg.cayley_eta,
        "min_tail": cfg.min_tail,
        "warmup": cfg.warmup_epochs,
        "ramp": cfg.ramp_epochs,
        "use_detx": cfg.use_detx,
        "requested_external_wwpgd_config": requested,
        "resolved_external_wwpgd_config": resolved,
    }


def _external_wwpgd_module():
    import ww_pgd
    return ww_pgd


def external_projected_layer_names(model: nn.Module) -> list[str]:
    return [name for name, _ in projected_matrix_modules(model)]


def _external_config_object(ww_pgd_module, cfg: ExternalWWTailConfigSpec):
    config_cls = getattr(ww_pgd_module, "WWTailConfig")
    kwargs = {
        "enable_tail_pgd": cfg.enable_tail_pgd,
        "q": cfg.q,
        "blend_eta": cfg.blend_eta,
        "cayley_eta": cfg.cayley_eta,
        "min_tail": cfg.min_tail,
        "use_detx": cfg.use_detx,
        "warmup_epochs": cfg.warmup_epochs,
        "ramp_epochs": cfg.ramp_epochs,
        "verbose": cfg.verbose,
    }
    try:
        return config_cls(**kwargs)
    except TypeError:
        import inspect
        sig = inspect.signature(config_cls)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return config_cls(**accepted)


def apply_external_wwpgd(
    model: nn.Module,
    *,
    event_index: int = 0,
    scheduled_token_fraction: float = 0.0,
    actual_step: int = 0,
    actual_tokens_seen: int = 0,
    cfg: ExternalWWTailConfigSpec | None = None,
) -> list[dict[str, object]]:
    cfg = cfg or resolved_external_wwpgd_config()
    ww_pgd_module = _external_wwpgd_module()
    external_cfg = _external_config_object(ww_pgd_module, cfg)
    layer_names = external_projected_layer_names(model)
    projector = getattr(ww_pgd_module, "ww_pgd_project")
    start = time.perf_counter()
    def layer_selector(mm: nn.Module, layer_name: str, row: object | None = None) -> nn.Module | None:
        if layer_name not in layer_names:
            return None
        cur: nn.Module = mm
        for part in layer_name.split("."):
            if part.isdigit() and isinstance(cur, (nn.ModuleList, nn.Sequential)):
                cur = cur[int(part)]
            elif hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                return None
        return cur if hasattr(cur, "weight") else None

    ww_logs: list[pd.DataFrame] = []
    try:
        result = projector(model, external_cfg, epoch=event_index, num_epochs=max(event_index + 1, cfg.ramp_epochs), global_step=actual_step, ww_logs=ww_logs, layer_selector=layer_selector)
    except TypeError:
        try:
            result = projector(model=model, config=external_cfg, layer_names=layer_names)
        except TypeError:
            result = projector(model, external_cfg, layer_names=layer_names)
    runtime = time.perf_counter() - start
    if isinstance(result, pd.DataFrame):
        rows = result.to_dict("records")
    elif isinstance(result, list):
        rows = list(result)
    elif ww_logs:
        key = "longname" if "longname" in ww_logs[-1].columns else "name"
        rows = [{"layer_name": str(row.get(key, row.get("name", "")))} for _, row in ww_logs[-1].iterrows() if str(row.get(key, row.get("name", ""))) in layer_names]
    elif result is None:
        rows = []
    else:
        rows = [dict(result)] if isinstance(result, dict) else [{"external_result": repr(result)}]
    if not rows:
        rows = [{"layer_name": name} for name in layer_names]
    for row in rows:
        row.setdefault("projection_event", event_index)
        row.setdefault("scheduled_token_fraction", scheduled_token_fraction)
        row.setdefault("actual_step", actual_step)
        row.setdefault("actual_tokens_seen", actual_tokens_seen)
        row.setdefault("projection_runtime", runtime / max(1, len(rows)))
        row.setdefault("wwpgd_implementation", "ww_pgd")
        row.setdefault("wwpgd_package", "ww_pgd")
        row.setdefault("wwpgd_commit", WWPGD_COMMIT)
        row.setdefault("q", cfg.q)
        row.setdefault("blend_eta", cfg.blend_eta)
        row.setdefault("cayley_eta", cfg.cayley_eta)
        row.setdefault("min_tail", cfg.min_tail)
        row.setdefault("warmup", cfg.warmup_epochs)
        row.setdefault("ramp", cfg.ramp_epochs)
        row.setdefault("use_detx", cfg.use_detx)
        row.setdefault("relative_frobenius_weight_change", row.get("relative_frobenius_change", 0.0))
        row.setdefault("relative_frobenius_change", row.get("relative_frobenius_weight_change", 0.0))
    return rows


def apply_wwpgd(model: nn.Module, target_alpha: float | None = None, strength: float | None = None, step: int = 0, warmup_steps: int = 0, ramp_steps: int = 0):
    if strength is not None:
        raise ValueError("deprecated strength is not a documented external WW_PGD parameter; use standard WWPGD or a documented external parameter ablation")
    return apply_external_wwpgd(model, event_index=step, actual_step=step)

COMPOSITE_SPECIFICATION_VERSION = "raw_and_composite_v1"


def raw_schema_v3_matrices(model: nn.Module):
    for i, block in enumerate(getattr(model, "blocks", [])):
        yield f"L{i:04d}_W_K", block.attn.key.weight.detach().float().cpu(), f"blocks.{i}.attn.key"
        yield f"L{i:04d}_W_Q", block.attn.query.weight.detach().float().cpu(), f"blocks.{i}.attn.query"
        yield f"L{i:04d}_W_V", block.attn.value.weight.detach().float().cpu(), f"blocks.{i}.attn.value"
        yield f"L{i:04d}_W_O", block.attn.proj.weight.detach().float().cpu(), f"blocks.{i}.attn.proj"
        yield f"L{i:04d}_W_MLP_IN", block.mlp[0].weight.detach().float().cpu(), f"blocks.{i}.mlp.0"
        yield f"L{i:04d}_W_MLP_OUT", block.mlp[2].weight.detach().float().cpu(), f"blocks.{i}.mlp.2"


def composite_matrices(model: nn.Module) -> dict[str, tuple[torch.Tensor, str, dict[str, tuple[int, ...]]]]:
    out = {}
    for i, block in enumerate(getattr(model, "blocks", [])):
        wk = block.attn.key.weight.detach().float().cpu(); wq = block.attn.query.weight.detach().float().cpu(); wv = block.attn.value.weight.detach().float().cpu(); wo = block.attn.proj.weight.detach().float().cpu()
        wi = block.mlp[0].weight.detach().float().cpu(); wout = block.mlp[2].weight.detach().float().cpu()
        shapes = {"W_K": tuple(wk.shape), "W_Q": tuple(wq.shape), "W_V": tuple(wv.shape), "W_O": tuple(wo.shape), "W_MLP_IN": tuple(wi.shape), "W_MLP_OUT": tuple(wout.shape)}
        out[f"L{i:04d}_KQ"] = (wk @ wq, "W_K @ W_Q", shapes)
        out[f"L{i:04d}_QK"] = (wq @ wk, "W_Q @ W_K", shapes)
        out[f"L{i:04d}_QK_effective"] = (wq.T @ wk, "W_Q.T @ W_K", shapes)
        out[f"L{i:04d}_KQ_effective"] = (wk.T @ wq, "W_K.T @ W_Q", shapes)
        n_head = block.attn.n_head; hd = block.attn.head_dim
        ov = torch.zeros(wo.size(0), wv.size(1))
        for h in range(n_head):
            sl = slice(h * hd, (h + 1) * hd)
            wqh, wkh, wvh, woh = wq[sl, :], wk[sl, :], wv[sl, :], wo[:, sl]
            ovh = woh @ wvh
            ov += ovh
            out[f"L{i:04d}_H{h:03d}_OV"] = (ovh, "W_O,h @ W_V,h", shapes)
            out[f"L{i:04d}_H{h:03d}_QK_effective"] = (wqh.T @ wkh, "W_Q,h.T @ W_K,h", shapes)
            out[f"L{i:04d}_H{h:03d}_KQ_effective"] = (wkh.T @ wqh, "W_K,h.T @ W_Q,h", shapes)
        out[f"L{i:04d}_OV"] = (ov, "sum_h W_O,h @ W_V,h", shapes)
        out[f"L{i:04d}_VO"] = (wv @ wo, "W_V @ W_O", shapes)
        out[f"L{i:04d}_MLP_IO"] = (wout @ wi, "W_MLP_OUT @ W_MLP_IN", shapes)
    return out


class MatrixHolder(nn.Module):
    def __init__(self, matrices: dict[str, torch.Tensor]):
        super().__init__()
        for name, mat in matrices.items():
            self.register_parameter(name, nn.Parameter(mat.clone(), requires_grad=False))


def composite_spectral_summary(model: nn.Module, *, step: int, tokens_seen: int, base_optimizer: str, extension: str, arm_name: str, seed: int, pair_id: str) -> list[dict[str, object]]:
    comps = composite_matrices(model)
    matrices = {k: v[0] for k, v in comps.items()}
    state = torch.random.get_rng_state()
    try:
        holder = MatrixHolder(matrices)
    finally:
        torch.random.set_rng_state(state)
    try:
        df = weightwatcher_details(holder)
    except Exception as e:
        rows = invalid_weightwatcher_rows(e, step=step, tokens_seen=tokens_seen, optimizer=arm_name, seed=seed, pair_id=pair_id)
        for r in rows:
            r.update({"base_optimizer": base_optimizer, "extension": extension, "arm_name": arm_name, "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION})
        return rows
    key = "longname" if "longname" in df.columns else "name"
    rows=[]
    for _, row in df.iterrows():
        cname = str(row.get(key, row.get("name", "")))
        if cname not in comps: continue
        _, formula, shapes = comps[cname]
        d = row.to_dict(); d.update({"step": step, "tokens_seen": tokens_seen, "base_optimizer": base_optimizer, "extension": extension, "arm_name": arm_name, "seed": seed, "pair_id": pair_id, "composite_name": cname, "formula": formula, "source_shapes": json.dumps(shapes, sort_keys=True), "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION})
        rows.append(d)
    return rows


def invalid_weightwatcher_rows(exc: BaseException, *, step: int, tokens_seen: int, optimizer: str, seed: int, pair_id: str, projection_event: int | None = None) -> list[dict[str, object]]:
    row = {
        "step": step,
        "tokens_seen": tokens_seen,
        "optimizer": optimizer,
        "seed": seed,
        "pair_id": pair_id,
        "spectral_estimator": "weightwatcher",
        "valid_for_science": False,
        "measurement_valid_for_science": False,
        "weightwatcher_exception_type": type(exc).__name__,
        "weightwatcher_exception_message": str(exc),
        "alpha": float("nan"),
        "D": float("nan"),
        "num_evals": float("nan"),
        "xmin": float("nan"),
        "detX_num": float("nan"),
        "weightwatcher_version": _ww_version(),
        "weightwatcher_configuration": '{"detX": true, "randomize": false, "plot": false}',
    }
    if projection_event is not None:
        row["projection_event"] = projection_event
        row["immediate_spectral_source"] = "weightwatcher_failed"
    return [row]


def _match_ww_row(df: pd.DataFrame, layer_name: str) -> dict[str, object] | None:
    for key in ("longname", "name"):
        if key in df.columns:
            matches = df[df[key].astype(str).eq(layer_name)]
            if len(matches):
                return matches.iloc[0].to_dict()
    return None


def measured_projection_spectral_rows(*args, **kwargs) -> list[dict[str, object]]:
    """Return paired WeightWatcher pre/post projection rows."""
    if args and isinstance(args[0], pd.DataFrame):
        pre = args[0]
        if len(args) > 1 and isinstance(args[1], pd.DataFrame):
            post = args[1]
            proj_rows = args[2] if len(args) > 2 else kwargs.get("projection_rows", [])
        else:
            model = args[1] if len(args) > 1 else kwargs.pop("model")
            proj_rows = kwargs.get("projection_rows", [])
            try:
                post = weightwatcher_details(model)
            except Exception as e:
                rows=[]
                for pr in proj_rows:
                    rows.append({**pr, "alpha_before": float("nan"), "alpha_after": float("nan"), "alpha_delta": float("nan"), "target_alpha": kwargs.get("target_alpha", float("nan")), "spectral_estimator": "weightwatcher", "immediate_spectral_source": "weightwatcher_failed", "measurement_valid_for_science": False, "valid_for_science": False, "weightwatcher_exception_type": type(e).__name__, "weightwatcher_exception_message": str(e)})
                return rows
        target_alpha = kwargs.get("target_alpha", args[3] if len(args) > 3 and isinstance(args[1], pd.DataFrame) else float("nan"))
        rows=[]
        for pr in proj_rows:
            lname=str(pr.get("layer_name", ""))
            before=_match_ww_row(pre, lname)
            after=_match_ww_row(post, lname)
            alpha_before = before.get("alpha", float("nan")) if before else float("nan")
            alpha_after = after.get("alpha", float("nan")) if after else float("nan")
            valid = bool(before and after and pd.notna(alpha_before) and pd.notna(alpha_after))
            out={**pr, **(after or {}), "layer_name": lname, "alpha_before": alpha_before, "alpha_after": alpha_after, "alpha_delta": (alpha_after-alpha_before if valid else float("nan")), "target_alpha": target_alpha, "spectral_estimator": "weightwatcher", "immediate_spectral_source": "weightwatcher_measured" if valid else "weightwatcher_unmatched", "measurement_valid_for_science": valid, "valid_for_science": valid}
            for fld in ("alpha","D","num_evals","xmin","detX_num"):
                out.setdefault(fld, float("nan"))
            rows.append(out)
        return rows
    model = args[0] if args else kwargs.pop("model")
    step=kwargs["step"]; tokens_seen=kwargs["tokens_seen"]; optimizer=kwargs["optimizer"]; seed=kwargs["seed"]; pair_id=kwargs["pair_id"]; projection_event=kwargs["projection_event"]; phase=kwargs.get("phase","post")
    try:
        rows = spectral_summary(model, step=step, tokens_seen=tokens_seen, optimizer=optimizer, seed=seed, pair_id=pair_id)
        for r in rows:
            r.update({"projection_event": projection_event,"projection_phase": phase,"immediate_spectral_source": "weightwatcher_measured","measurement_valid_for_science": bool(r.get("valid_for_science", True))})
        return rows
    except Exception as e:
        return invalid_weightwatcher_rows(e, step=step, tokens_seen=tokens_seen, optimizer=optimizer, seed=seed, pair_id=pair_id, projection_event=projection_event)
