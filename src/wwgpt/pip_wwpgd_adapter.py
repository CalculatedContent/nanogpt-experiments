"""Explicit boundary to the pip-installed ``ww_pgd`` public API."""
from __future__ import annotations

import inspect
import json
from importlib import metadata
from pathlib import Path
from typing import Any

REQUIRED_PROJECTOR_PARAMETERS = frozenset(
    {"model", "cfg", "epoch", "num_epochs", "global_step", "ww_logs", "layer_selector"}
)
REQUIRED_CONFIG_OPTIONS = frozenset(
    {"enable_tail_pgd", "q", "blend_eta", "cayley_eta", "min_tail", "use_detx", "warmup_epochs", "ramp_epochs", "verbose"}
)


def _distribution(*names: str) -> metadata.Distribution | None:
    for name in names:
        try:
            return metadata.distribution(name)
        except metadata.PackageNotFoundError:
            pass
    return None


def inspect_pip_wwpgd_api() -> dict[str, Any]:
    """Inspect, but never modify, the authoritative installed API."""
    import ww_pgd

    config = getattr(ww_pgd, "WWTailConfig", None)
    projector = getattr(ww_pgd, "ww_pgd_project", None)
    if config is None or projector is None:
        raise RuntimeError("pip-installed ww_pgd must expose WWTailConfig and ww_pgd_project")
    config_signature = inspect.signature(config)
    projector_signature = inspect.signature(projector)
    projector_names = set(projector_signature.parameters)
    missing = sorted(REQUIRED_PROJECTOR_PARAMETERS - projector_names)
    if missing:
        raise RuntimeError(f"incompatible pip-installed ww_pgd projector; missing parameters: {missing}")
    return {
        "module": ww_pgd,
        "config_class": config,
        "projector": projector,
        "config_signature_object": config_signature,
        "projector_signature_object": projector_signature,
        "native_internal_diagnostics": "diagnostic_logs" in projector_names,
    }


def resolve_pip_wwpgd_provenance() -> dict[str, Any]:
    """Return package metadata, including PEP 610 VCS provenance when supplied."""
    api = inspect_pip_wwpgd_api()
    ww_pgd = api["module"]
    dist = _distribution("ww-pgd", "ww_pgd")
    direct: dict[str, Any] = {}
    if dist is not None:
        raw = dist.read_text("direct_url.json")
        if raw:
            try:
                direct = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                direct = {}
    vcs = direct.get("vcs_info") or {}
    if vcs:
        mode = "pip-vcs"
    elif (direct.get("dir_info") or {}).get("editable"):
        mode = "pip-editable"
    elif direct:
        mode = "pip-direct-url"
    else:
        mode = "pypi"
    ww_dist_name = dist.metadata.get("Name") if dist is not None else "ww-pgd"
    ww_version = dist.version if dist is not None else getattr(ww_pgd, "__version__", "unknown")

    import weightwatcher

    weightwatcher_dist = _distribution("weightwatcher")
    return {
        "wwpgd_distribution_name": str(ww_dist_name),
        "wwpgd_installed_version": str(ww_version),
        "wwpgd_module_path": str(Path(ww_pgd.__file__).resolve()),
        "wwpgd_source_url": direct.get("url"),
        "wwpgd_install_mode": mode,
        "wwpgd_resolved_commit": vcs.get("commit_id"),
        "wwpgd_projector_signature": str(api["projector_signature_object"]),
        "wwpgd_config_signature": str(api["config_signature_object"]),
        "wwpgd_native_internal_diagnostics": api["native_internal_diagnostics"],
        "wwpgd_dependency_pinned": False,
        "weightwatcher_installed_version": str(weightwatcher_dist.version if weightwatcher_dist else getattr(weightwatcher, "__version__", "unknown")),
        "weightwatcher_module_path": str(Path(weightwatcher.__file__).resolve()),
    }


def construct_pip_wwpgd_config(spec: object) -> tuple[object, dict[str, Any]]:
    """Map every mathematical experiment option into the installed config."""
    api = inspect_pip_wwpgd_api()
    target_alpha = float(getattr(spec, "target_alpha"))
    if target_alpha <= 1.0:
        raise ValueError("target_alpha must be greater than 1")
    requested = {
        "enable_tail_pgd": bool(getattr(spec, "enable_tail_pgd")),
        "q": 1.0 / (target_alpha - 1.0),
        "blend_eta": float(getattr(spec, "blend_eta")),
        "cayley_eta": float(getattr(spec, "cayley_eta")),
        "min_tail": int(getattr(spec, "min_tail")),
        "use_detx": bool(getattr(spec, "use_detx")),
        "warmup_epochs": int(getattr(spec, "warmup_epochs")),
        "ramp_epochs": int(getattr(spec, "ramp_epochs")),
        "verbose": bool(getattr(spec, "verbose")),
    }
    limit = getattr(spec, "max_relative_frobenius_change", None)
    params = api["config_signature_object"].parameters
    has_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    missing = sorted(name for name in REQUIRED_CONFIG_OPTIONS if name not in params and not has_kwargs)
    if missing:
        raise RuntimeError(f"incompatible pip-installed ww_pgd WWTailConfig; required options unsupported: {missing}")
    if limit is not None:
        if "max_relative_frobenius_change" not in params and not has_kwargs:
            raise RuntimeError("pip-installed ww_pgd cannot enforce requested max_relative_frobenius_change")
        requested["max_relative_frobenius_change"] = float(limit)
    config = api["config_class"](**requested)
    resolved = {name: getattr(config, name, value) for name, value in requested.items()}
    ignored = [name for name, value in requested.items() if resolved[name] != value]
    if ignored:
        raise RuntimeError(f"pip-installed ww_pgd silently changed requested options: {ignored}")
    return config, {"requested": requested, "resolved": resolved}


def run_pip_wwpgd_candidate(model: object, config: object, **kwargs: Any) -> dict[str, Any]:
    """Invoke the installed projector exactly once with its supported diagnostics."""
    api = inspect_pip_wwpgd_api()
    ww_logs: list[Any] = []
    diagnostics: list[dict[str, Any]] = []
    call = dict(kwargs, ww_logs=ww_logs)
    if api["native_internal_diagnostics"]:
        call["diagnostic_logs"] = diagnostics
    api["projector"](model, config, **call)
    return {"ww_logs": ww_logs, "diagnostic_logs": diagnostics, "native_internal_diagnostics": api["native_internal_diagnostics"]}
