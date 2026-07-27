from __future__ import annotations

import copy
import json
import hashlib
import math
import time
from dataclasses import dataclass
import dataclasses
from importlib import metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
from wwgpt.adaptive_wwpgd import matrix_type, block_index
from wwgpt.pip_wwpgd_adapter import (
    construct_pip_wwpgd_config,
    inspect_pip_wwpgd_api,
    resolve_pip_wwpgd_provenance,
    run_pip_wwpgd_candidate,
)

_WWPGD_PROVENANCE = resolve_pip_wwpgd_provenance()
WWPGD_COMMIT = str(
    _WWPGD_PROVENANCE.get("wwpgd_resolved_commit")
    or f"version:{_WWPGD_PROVENANCE['wwpgd_installed_version']}"
)
WWPGD_DIAGNOSTICS_SCHEMA_VERSION = 1
WWPGD_ADAPTER_MODE = "stock_candidate_displacement_scaling_v1"
SCIENTIFIC_SCHEMA_VERSION = 3
PROJECTED_LAYER_SUFFIXES = ("attn.key", "attn.query", "attn.value", "attn.proj", "mlp.0", "mlp.2")


def alpha_measurement_exclusion_reason(row: object, *, max_D: float | None,
                                       min_tail: int, require_projected: bool = True) -> str:
    """Apply the alpha-quality gate used by both control and analysis.

    ``detX_num`` is WeightWatcher's selected tail count.  Older WeightWatcher
    releases do not always return it, so ``num_evals`` is the explicit fallback
    eigenvalue count.  The estimator label is deliberately mandatory: an
    unlabeled or fallback estimate must never silently become scientific data.
    """
    get = row.get if hasattr(row, "get") else lambda key, default=None: default
    estimator = str(get("spectral_estimator", "")).strip().lower()
    if estimator != "weightwatcher":
        return "spectral_estimator_not_weightwatcher"
    projected_value = get("projected", get("included_in_projected_alpha_summary", False))
    projected = projected_value is True or str(projected_value).strip().lower() in {"true", "1"}
    if require_projected and not projected:
        return "nonprojected_matrix"
    try:
        alpha = float(get("alpha", float("nan")))
        xmin = float(get("xmin", float("nan")))
    except (TypeError, ValueError):
        return "invalid_alpha_or_xmin"
    if not math.isfinite(alpha):
        return "invalid_alpha"
    if not (math.isfinite(xmin) and xmin > 0):
        return "invalid_xmin"
    if max_D is not None:
        try:
            d_value = float(get("D", float("nan")))
        except (TypeError, ValueError):
            d_value = float("nan")
        if not math.isfinite(d_value):
            return "invalid_D"
        if d_value > float(max_D):
            return "D_above_max_D"
    tail = get("detX_num", None)
    try:
        tail_value = float(tail)
    except (TypeError, ValueError):
        tail_value = float("nan")
    if not math.isfinite(tail_value):
        try:
            tail_value = float(get("num_evals", float("nan")))
        except (TypeError, ValueError):
            tail_value = float("nan")
    if not math.isfinite(tail_value) or tail_value < int(min_tail):
        return "insufficient_tail_or_eigenvalue_count"
    return ""


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
    # Run pip-installed WeightWatcher on CPU for MPS/XLA training models.
    import weightwatcher

    live_device = _model_device(model)
    offloaded = live_device.type in {"mps", "xla"}
    analysis_model = _cpu_candidate_model(model) if offloaded else model
    analysis_device = _model_device(analysis_model)
    start = time.perf_counter()
    watcher = weightwatcher.WeightWatcher(model=analysis_model)
    details = watcher.analyze(detX=True, randomize=randomize, plot=False)
    if details is None:
        raise RuntimeError("WeightWatcher.analyze returned None")
    df = details.copy()
    df["analysis_runtime"] = time.perf_counter() - start
    df["weightwatcher_version"] = _ww_version()
    df["spectral_estimator"] = "weightwatcher"
    df["spectral_estimator_version"] = df["weightwatcher_version"]
    df["weightwatcher_configuration"] = json.dumps(
        {"detX": True, "randomize": bool(randomize), "plot": False},
        sort_keys=True,
    )
    df["analysis_execution_device"] = str(analysis_device)
    df["live_model_device"] = str(live_device)
    df["analysis_offloaded"] = offloaded
    df["valid_for_science"] = True
    return df


def nonmutating_weightwatcher_details(model: nn.Module, *, randomize: bool = False) -> pd.DataFrame:
    """Run WeightWatcher as a pure observation and restore all torch RNG/state."""
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    try:
        model.eval()
        return weightwatcher_details(model, randomize=randomize)
    finally:
        model.load_state_dict(state)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(was_training)


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


def spectral_summary(model: nn.Module, *, step: int, tokens_seen: int, optimizer: str, seed: int, pair_id: str,
                     randomize: bool = True) -> list[dict[str, object]]:
    df = add_weightwatcher_diagnostic_fields(weightwatcher_details(model, randomize=randomize))
    df["step"] = step; df["tokens_seen"] = tokens_seen; df["optimizer"] = optimizer; df["seed"] = seed; df["pair_id"] = pair_id
    return df.to_dict("records")


def alpha_measurement_rows(details: pd.DataFrame | None, model: nn.Module, *, step: int,
                           tokens_seen: int, seed: int, pair_id: str,
                           base_optimizer: str, extension: str, arm_name: str,
                           failure_reason: str = "") -> list[dict[str, object]]:
    """Normalize one nonrandomized WW result, including explicit exclusions.

    Every named weighted module is represented.  Non-projected architecture parts
    remain useful audit records, but cannot enter the projected-layer summary.
    """
    config = json.dumps({"detX": True, "plot": False, "randomize": False}, sort_keys=True)
    frame = details if details is not None else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        kind = ("embedding" if isinstance(module, nn.Embedding) else
                "layernorm" if isinstance(module, nn.LayerNorm) else
                "output_head" if name == "lm_head" else matrix_type(name) or "other")
        projected = is_projected_layer(name)
        matched = _match_ww_row(frame, name) if len(frame) else None
        data = matched.to_dict() if hasattr(matched, "to_dict") else dict(matched or {})
        alpha = pd.to_numeric(pd.Series([data.get("alpha")]), errors="coerce").iloc[0]
        reason = failure_reason
        if not reason and matched is None: reason = "missing_weightwatcher_row"
        if not reason and not math.isfinite(float(alpha)): reason = "invalid_alpha"
        if not reason and not projected: reason = f"excluded_nonprojected_{kind}"
        rows.append({
            "seed": seed, "trial_id": pair_id, "pair_id": pair_id,
            "base_optimizer": base_optimizer, "extension": extension, "arm_name": arm_name,
            "optimizer_step": step, "tokens_seen": tokens_seen, "layer_name": name,
            "matrix_type": kind, "block": block_index(name), "alpha": data.get("alpha"),
            "D": data.get("D"), "xmin": data.get("xmin"), "detX_num": data.get("detX_num"),
            "num_evals": data.get("num_evals"), "spectral_norm": data.get("spectral_norm"),
            "stable_rank": data.get("stable_rank"), "WeightWatcher version": data.get("weightwatcher_version", _ww_version()),
            "spectral_estimator": data.get("spectral_estimator", "weightwatcher" if matched is not None else ""),
            "projected": projected,
            "weightwatcher_configuration": data.get("weightwatcher_configuration", config),
            "analysis_runtime": data.get("analysis_runtime"),
            "valid_for_science": not bool(reason), "validity_exclusion_reason": reason,
            "validity/exclusion reason": reason,
            "included_in_projected_alpha_summary": projected and not bool(reason),
        })
    return rows


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
    target_alpha: float = 2.0
    blend_eta: float = 0.5
    cayley_eta: float = 0.25
    min_tail: int = 5
    use_detx: bool = True
    warmup_epochs: int = 0
    ramp_epochs: int = 0
    verbose: bool = False
    max_relative_frobenius_change: float | None = None
    candidate_device: str = "auto"


WWTailConfig = ExternalWWTailConfigSpec


STANDARD_WWPGD_BLEND_ETA = 0.5
STANDARD_WWPGD_WARMUP_EVENTS = 0
STANDARD_WWPGD_RAMP_EVENTS = 0


def target_alpha_to_external_rank_exponent(target_alpha: float) -> float:
    """Translate the public spectral target at the private dependency boundary."""
    return 1.0 / (target_alpha - 1.0)


def external_wwpgd_config_from_experiment(cfg: object) -> ExternalWWTailConfigSpec:
    target_alpha = float(getattr(cfg, "target_alpha"))
    if not math.isfinite(target_alpha) or target_alpha <= 1.0:
        raise ValueError("target_alpha must be finite and greater than 1")
    return ExternalWWTailConfigSpec(
        enable_tail_pgd=True,
        target_alpha=target_alpha,
        blend_eta=float(getattr(cfg, "blend_eta", STANDARD_WWPGD_BLEND_ETA)),
        cayley_eta=float(getattr(cfg, "cayley_eta")),
        min_tail=int(getattr(cfg, "min_tail")),
        use_detx=bool(getattr(cfg, "use_detx")),
        warmup_epochs=STANDARD_WWPGD_WARMUP_EVENTS,
        ramp_epochs=STANDARD_WWPGD_RAMP_EVENTS,
        verbose=bool(getattr(cfg, "verbose", False)),
        max_relative_frobenius_change=getattr(cfg, "max_relative_frobenius_change", None),
        candidate_device=str(getattr(cfg, "candidate_device", "auto")),
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
    from dataclasses import asdict as _asdict
    requested = _asdict(requested_cfg) if requested_cfg is not None and hasattr(requested_cfg, "__dataclass_fields__") else (dict(vars(requested_cfg)) if requested_cfg is not None and hasattr(requested_cfg, "__dict__") else {})
    _, mapping = construct_pip_wwpgd_config(cfg)
    resolved = mapping["resolved"]
    derived = target_alpha_to_external_rank_exponent(cfg.target_alpha)
    return {
        **_WWPGD_PROVENANCE,
        "wwpgd_package": "ww_pgd",
        "wwpgd_source_repository": "CalculatedContent/WW_PGD",
        "wwpgd_commit": WWPGD_COMMIT,
        "wwpgd_diagnostics_schema_version": WWPGD_DIAGNOSTICS_SCHEMA_VERSION,
        "wwpgd_implementation": "ww_pgd",
        "wwpgd_adapter_mode": WWPGD_ADAPTER_MODE,
        "wwpgd_adaptive_implementation": "nanogpt-experiments scales stock WW_PGD candidate displacements per layer",
        "target_alpha": float(getattr(requested_cfg, "target_alpha", 2.0)),
        "derived_external_rank_exponent": derived,
        "derivation_formula": "1 / (target_alpha - 1)",
        "external_parameter_name": "q",
        "external_rank_exponent_was_configured": False,
        "blend_eta": cfg.blend_eta,
        "cayley_eta": cfg.cayley_eta,
        "min_tail": cfg.min_tail,
        "warmup": cfg.warmup_epochs,
        "ramp": cfg.ramp_epochs,
        "use_detx": cfg.use_detx,
        "candidate_device": cfg.candidate_device,
        "requested_external_wwpgd_config": requested,
        "resolved_external_wwpgd_config": resolved,
    }


def _external_wwpgd_module():
    import ww_pgd
    return ww_pgd


def external_projected_layer_names(model: nn.Module) -> list[str]:
    return [name for name, _ in projected_matrix_modules(model)]


def _external_config_object(ww_pgd_module, cfg: ExternalWWTailConfigSpec):
    del ww_pgd_module
    return construct_pip_wwpgd_config(cfg)[0]



def _assert_stock_wwpgd_api(projector: object) -> None:
    api = inspect_pip_wwpgd_api()
    if projector is not api["projector"]:
        raise RuntimeError("projector must be the unmodified pip-installed ww_pgd public function")


@dataclass(frozen=True)
class StockWWPGDCandidate:
    pre_projection_details: pd.DataFrame
    original_weights: dict[str, torch.Tensor]
    candidate_weights: dict[str, torch.Tensor]
    original_to_candidate_relative_change: dict[str, float]
    stock_candidate_changed: dict[str, bool]
    runtime: float
    stock_config: ExternalWWTailConfigSpec
    internal_diagnostics: list[dict[str, object]] = dataclasses.field(default_factory=list)
    stock_commit: str = WWPGD_COMMIT
    candidate_execution_device: str = "live"
    live_model_device: str = "cpu"
    candidate_offloaded: bool = False


def _module_by_name(model: nn.Module, layer_name: str) -> nn.Module | None:
    cur: nn.Module = model
    for part in layer_name.split("."):
        if part.isdigit() and isinstance(cur, (nn.ModuleList, nn.Sequential)):
            cur = cur[int(part)]
        elif hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            return None
    return cur if hasattr(cur, "weight") else None


def _selected_layer_selector(selected_names: set[str]):
    def layer_selector(mm: nn.Module, layer_name: str, row: object | None = None) -> nn.Module | None:
        if not is_projected_layer(layer_name) or layer_name not in selected_names:
            return None
        return _module_by_name(mm, layer_name)
    return layer_selector


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def resolve_candidate_execution_device(model: nn.Module, requested: str) -> str:
    requested = str(requested or "auto").lower()
    if requested not in {"auto", "live", "cpu"}:
        raise ValueError("candidate_device must be auto, live, or cpu")
    if requested == "cpu":
        return "cpu"
    if requested == "live":
        return "live"
    return "cpu" if _model_device(model).type in {"mps", "xla"} else "live"


def _cpu_candidate_model(model: nn.Module) -> nn.Module:
    clone = copy.deepcopy(model).to(torch.device("cpu"))
    clone.load_state_dict(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    )
    clone.train(model.training)
    return clone


def build_stock_wwpgd_candidate(
    model: nn.Module,
    *,
    event_index: int = 0,
    actual_step: int = 0,
    cfg: ExternalWWTailConfigSpec | None = None,
    selected_names: set[str] | None = None,
    layer_selector: object | None = None,
) -> StockWWPGDCandidate:
    cfg = cfg or resolved_external_wwpgd_config()
    full_cfg = ExternalWWTailConfigSpec(
        enable_tail_pgd=cfg.enable_tail_pgd,
        target_alpha=cfg.target_alpha,
        blend_eta=cfg.blend_eta,
        cayley_eta=cfg.cayley_eta,
        min_tail=cfg.min_tail,
        use_detx=cfg.use_detx,
        warmup_epochs=cfg.warmup_epochs,
        ramp_epochs=cfg.ramp_epochs,
        verbose=cfg.verbose,
        max_relative_frobenius_change=cfg.max_relative_frobenius_change,
        candidate_device=cfg.candidate_device,
    )
    ww_pgd_module = _external_wwpgd_module()
    projector = getattr(ww_pgd_module, "ww_pgd_project")
    _assert_stock_wwpgd_api(projector)
    selected_names = selected_names or set(external_projected_layer_names(model))
    originals = {name: weight.detach().clone() for name, weight in projected_matrix_modules(model)}
    live_device = _model_device(model)
    execution_mode = resolve_candidate_execution_device(model, full_cfg.candidate_device)
    execution_model = model if execution_mode == "live" else _cpu_candidate_model(model)
    selector = layer_selector or _selected_layer_selector(set(selected_names))
    start = time.perf_counter()
    candidates: dict[str, torch.Tensor] = {}
    result: dict[str, object] = {}
    try:
        with torch.no_grad():
            result = run_pip_wwpgd_candidate(
                execution_model,
                _external_config_object(ww_pgd_module, full_cfg),
                epoch=event_index,
                num_epochs=max(event_index + 1, 1),
                global_step=actual_step,
                layer_selector=selector,
            )
            candidates = {
                name: weight.detach().clone().cpu()
                for name, weight in projected_matrix_modules(execution_model)
            }
    finally:
        if execution_mode == "live":
            with torch.no_grad():
                for name, weight in projected_matrix_modules(model):
                    weight.copy_(originals[name].to(weight.device, dtype=weight.dtype))
                    if not torch.equal(weight.detach().cpu(), originals[name].cpu()):
                        raise RuntimeError(
                            f"failed to restore original WW_PGD weight bitwise for {name}"
                        )
    runtime = time.perf_counter() - start
    missing_candidates = sorted(set(originals) - set(candidates))
    if missing_candidates:
        raise RuntimeError(f"WWPGD candidate is missing projected matrices: {missing_candidates}")
    ww_logs = result.get("ww_logs", [])
    diagnostic_logs = list(result.get("diagnostic_logs", []))
    usable = [item for item in ww_logs if isinstance(item, pd.DataFrame) and not item.empty]
    if len(usable) != 1:
        raise RuntimeError(
            "stock WW_PGD candidate generation expected exactly one usable "
            f"ww_logs DataFrame, got {len(usable)}"
        )
    relative_change: dict[str, float] = {}
    changed: dict[str, bool] = {}
    with torch.no_grad():
        for name, original in originals.items():
            candidate = candidates[name].to(original.device, dtype=original.dtype)
            displacement = (candidate - original).float()
            relative_change[name] = float(
                torch.linalg.norm(displacement)
                / max(float(torch.linalg.norm(original.float())), 1e-12)
            )
            changed[name] = not torch.equal(candidates[name].cpu(), original.cpu())
    common = {
        "candidate_execution_device": execution_mode,
        "live_model_device": str(live_device),
        "candidate_offloaded": execution_mode != "live",
    }
    if result.get("native_internal_diagnostics"):
        for row in diagnostic_logs:
            row.setdefault("diagnostics_schema_version", 1)
            row.setdefault("diagnostics_mode", "native")
            row.setdefault("native_internal_diagnostics", True)
            row.setdefault("valid_observable_diagnostic", True)
            row.setdefault("unsupported_internal_fields", json.dumps([]))
            row.update(common)
    else:
        unsupported = [
            "k_pl", "k_detx", "k_star", "selected_lambda_threshold",
            "selected_tail_size", "TraceLog", "cayley_ratios", "clipping_counts",
            "shaped_movement",
        ]
        frame = usable[0]
        name_column = "longname" if "longname" in frame.columns else "name"
        by_name = {str(row.get(name_column, "")): row for _, row in frame.iterrows()}
        for name, original in originals.items():
            observed = by_name.get(name, {})
            diagnostic_logs.append(
                {
                    "diagnostics_schema_version": 1,
                    "diagnostics_mode": "compatibility",
                    "native_internal_diagnostics": False,
                    "valid_observable_diagnostic": bool(
                        math.isfinite(relative_change[name])
                    ),
                    "status": "unsupported_internal_fields",
                    "unsupported_internal_fields": json.dumps(unsupported),
                    "layer_name": name,
                    "layer_shape": list(original.shape),
                    "alpha": observed.get("alpha"),
                    "D": observed.get("D"),
                    "xmin": observed.get("xmin"),
                    "detX_num": observed.get("detX_num"),
                    "num_evals": observed.get("num_evals"),
                    "candidate_changed": changed[name],
                    "original_to_candidate_relative_frobenius_change": relative_change[name],
                    "original_frobenius_norm": float(original.float().norm()),
                    "candidate_frobenius_norm": float(candidates[name].float().norm()),
                    "target_alpha": full_cfg.target_alpha,
                    "candidate_relative_frobenius_change": relative_change[name],
                    "configured_blend_eta": full_cfg.blend_eta,
                    "configured_cayley_eta": full_cfg.cayley_eta,
                    "configured_min_tail": full_cfg.min_tail,
                    "configured_use_detx": full_cfg.use_detx,
                    "projection_runtime": runtime,
                    "warning_message": (
                        "private WWPGD internals are unsupported by the installed package"
                    ),
                    **common,
                    **_WWPGD_PROVENANCE,
                }
            )
    return StockWWPGDCandidate(
        usable[0].copy(),
        originals,
        candidates,
        relative_change,
        changed,
        runtime,
        full_cfg,
        diagnostic_logs,
        WWPGD_COMMIT,
        execution_mode,
        str(live_device),
        execution_mode != "live",
    )


def apply_external_wwpgd(
    model: nn.Module,
    *,
    event_index: int = 0,
    scheduled_token_fraction: float = 0.0,
    actual_step: int = 0,
    actual_tokens_seen: int = 0,
    cfg: ExternalWWTailConfigSpec | None = None,
    precomputed_details: pd.DataFrame | None = None,
    layer_hardness: dict[str, float] | None = None,
    global_event_hardness: float | None = None,
    layer_max_relative_change: dict[str, float | None] | None = None,
    stock_candidate: StockWWPGDCandidate | None = None,
    sham_seed: int | None = None,
) -> list[dict[str, object]]:
    if precomputed_details is not None:
        raise TypeError("stock WW_PGD adapter does not accept precomputed_details")
    cfg = cfg or resolved_external_wwpgd_config()
    if stock_candidate is not None:
        candidate = stock_candidate
        if layer_hardness is None:
            layer_hardness = {n: 1.0 for n in candidate.original_weights}
    elif layer_hardness is None:
        candidate = build_stock_wwpgd_candidate(model, event_index=event_index, actual_step=actual_step, cfg=cfg)
        layer_hardness = {n: 1.0 for n in candidate.original_weights}
        global_event_hardness = 1.0
        layer_max_relative_change = {}
    else:
        candidate = build_stock_wwpgd_candidate(model, event_index=event_index, actual_step=actual_step, cfg=cfg, selected_names=set(layer_hardness))
    geh = 1.0 if global_event_hardness is None else float(global_event_hardness)
    rows=[]
    with torch.no_grad():
        live = dict(projected_matrix_modules(model))
        for name, orig in candidate.original_weights.items():
            if name not in layer_hardness and layer_hardness:
                continue
            req_h = max(0.0, min(1.0, float(layer_hardness.get(name, 0.0)) * geh))
            cand = candidate.candidate_weights[name]
            weight = live[name]
            real_disp = cand.to(orig.device, dtype=orig.dtype) - orig
            disp = real_disp
            displacement_cosine = 1.0 if float(torch.linalg.norm(real_disp.float())) > 0 else float("nan")
            displacement_kind = "target_directed"
            if sham_seed is not None:
                # Generate on CPU so the scientific control stream is independent
                # of accelerator type and does not consume the training RNG.
                digest = hashlib.sha256(f"{int(sham_seed)}:{event_index}:{name}".encode()).digest()
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
                d32 = real_disp.detach().float().cpu()
                random = torch.randn(d32.shape, generator=generator, dtype=torch.float32)
                d2 = torch.sum(d32 * d32)
                if float(d2) > 0.0:
                    random = random - (torch.sum(random * d32) / d2) * d32
                random_norm = torch.linalg.norm(random)
                dnorm = torch.linalg.norm(d32)
                if float(random_norm) > 0.0 and float(dnorm) > 0.0:
                    random = random * (dnorm / random_norm)
                else:
                    random.zero_()
                disp = random.to(orig.device, dtype=orig.dtype)
                denom = float(torch.linalg.norm(random) * dnorm)
                displacement_cosine = float(torch.sum(random * d32) / denom) if denom > 0.0 else float("nan")
                displacement_kind = "norm_matched_sham"
            requested = orig + req_h * disp
            req_rel = float(torch.linalg.norm((requested - orig).float()) / max(float(torch.linalg.norm(orig.float())), 1e-12))
            limit = (layer_max_relative_change or {}).get(name, cfg.max_relative_frobenius_change)
            scale = 1.0
            if limit is not None and math.isfinite(req_rel) and req_rel > float(limit):
                scale = float(limit) / max(req_rel, 1e-12)
            applied = orig + scale * (requested - orig)
            app_rel = float(torch.linalg.norm((applied - orig).float()) / max(float(torch.linalg.norm(orig.float())), 1e-12))
            if req_h == 0.0:
                applied = orig
                app_rel = 0.0
                scale = 1.0
            weight.copy_(applied.to(weight.device, dtype=weight.dtype))
            changed = app_rel > 0.0 and not torch.equal(weight.detach().cpu(), orig.cpu())
            rows.append({"layer_name": name, "matrix_type": matrix_type(name), "block": block_index(name), "projection_event": event_index,
                "scheduled_token_fraction": scheduled_token_fraction, "actual_step": actual_step, "actual_tokens_seen": actual_tokens_seen,
                "projection_runtime": candidate.runtime / max(1, len(candidate.original_weights)), "wwpgd_implementation": "ww_pgd", "wwpgd_adapter_mode": WWPGD_ADAPTER_MODE,
                "wwpgd_package": "ww_pgd", "wwpgd_commit": WWPGD_COMMIT, "target_alpha": cfg.target_alpha, "blend_eta": cfg.blend_eta, "cayley_eta": cfg.cayley_eta, "min_tail": cfg.min_tail,
                "warmup": 0, "ramp": 0, "use_detx": cfg.use_detx, "stock_candidate_changed": candidate.stock_candidate_changed.get(name, False),
                "stock_candidate_relative_frobenius_change": candidate.original_to_candidate_relative_change.get(name, 0.0),
                "combined_hardness_requested": req_h, "combined_hardness_applied": req_h * scale, "trust_region_limit": limit, "trust_region_scale": scale,
                "relative_frobenius_change_requested": req_rel, "relative_frobenius_change_applied": app_rel, "relative_frobenius_change": app_rel,
                "relative_frobenius_weight_change": app_rel, "changed": changed, "projection_attempted": req_h > 0.0, "projected": changed})
            rows[-1].update({"displacement_kind": displacement_kind,
                             "real_candidate_displacement_cosine": displacement_cosine,
                             "sham_seed": int(sham_seed) if sham_seed is not None else None})
    return rows

def apply_wwpgd(model: nn.Module, target_alpha: float = 2.0, *, step: int = 0):
    """Apply WW-PGD using its sole public spectral target."""
    target_alpha = float(target_alpha)
    if not math.isfinite(target_alpha) or target_alpha <= 1.0:
        raise ValueError("target_alpha must be finite and greater than 1")
    cfg = ExternalWWTailConfigSpec(target_alpha=target_alpha)
    return apply_external_wwpgd(model, event_index=step, actual_step=step, cfg=cfg)

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
