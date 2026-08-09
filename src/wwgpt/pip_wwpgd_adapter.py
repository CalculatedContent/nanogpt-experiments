"""Explicit boundary to the pip-installed ``ww_pgd`` public API."""
from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Iterator, Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

WWPGD_COMMIT_PIN_ENV = "WWPGD_COMMIT_PIN"
WWPGD_COMMIT_PIN_MIN_LENGTH = 12
WWPGD_COMMIT_PIN_MAX_LENGTH = 64
_HEXADECIMAL_COMMIT = re.compile(r"^[0-9a-f]+$")

REQUIRED_PROJECTOR_PARAMETERS = frozenset(
    {"model", "cfg", "epoch", "num_epochs", "global_step", "ww_logs", "layer_selector"}
)
REQUIRED_CONFIG_OPTIONS = frozenset(
    {
        "enable_tail_pgd",
        "q",
        "blend_eta",
        "cayley_eta",
        "min_tail",
        "use_detx",
        "warmup_epochs",
        "ramp_epochs",
        "verbose",
    }
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
        raise RuntimeError(
            f"incompatible pip-installed ww_pgd projector; missing parameters: {missing}"
        )
    return {
        "module": ww_pgd,
        "config_class": config,
        "projector": projector,
        "config_signature_object": config_signature,
        "projector_signature_object": projector_signature,
        "native_internal_diagnostics": "diagnostic_logs" in projector_names,
    }


def _requested_wwpgd_commit_pin() -> str | None:
    requested = (os.environ.get(WWPGD_COMMIT_PIN_ENV) or "").strip().lower()
    return requested or None


def _validate_requested_commit_pin(requested_pin: str) -> str:
    pin = str(requested_pin).strip().lower()
    if not (
        WWPGD_COMMIT_PIN_MIN_LENGTH
        <= len(pin)
        <= WWPGD_COMMIT_PIN_MAX_LENGTH
    ) or _HEXADECIMAL_COMMIT.fullmatch(pin) is None:
        raise ValueError(
            f"{WWPGD_COMMIT_PIN_ENV} must be a hexadecimal Git SHA prefix "
            f"between {WWPGD_COMMIT_PIN_MIN_LENGTH} and "
            f"{WWPGD_COMMIT_PIN_MAX_LENGTH} characters"
        )
    return pin


def _resolve_pip_wwpgd_base_provenance() -> dict[str, Any]:
    """Resolve package metadata that is stable for the lifetime of this process."""
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
    ww_version = (
        dist.version
        if dist is not None
        else getattr(ww_pgd, "__version__", "unknown")
    )

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
        "wwpgd_native_internal_diagnostics": api[
            "native_internal_diagnostics"
        ],
        "weightwatcher_installed_version": str(
            weightwatcher_dist.version
            if weightwatcher_dist
            else getattr(weightwatcher, "__version__", "unknown")
        ),
        "weightwatcher_module_path": str(Path(weightwatcher.__file__).resolve()),
    }


def verify_wwpgd_commit_pin(
    provenance: Mapping[str, Any],
    *,
    requested_pin: str | None = None,
) -> dict[str, Any]:
    """Return explicit floating/verified provenance or fail on an invalid pin.

    A commit pin is runtime verification of a floating Git dependency. It does
    not rewrite the installation specification and therefore never makes
    ``wwpgd_dependency_pinned`` true.
    """
    info = dict(provenance)
    raw_pin = _requested_wwpgd_commit_pin() if requested_pin is None else requested_pin
    pin = str(raw_pin or "").strip().lower() or None
    info.update(
        {
            "wwpgd_commit_pin_env": WWPGD_COMMIT_PIN_ENV,
            "wwpgd_commit_pin_requested": pin,
            "wwpgd_commit_pin_verified": False,
            "wwpgd_commit_pin_status": "floating" if pin is None else "requested",
            "wwpgd_dependency_pinned": False,
        }
    )
    if pin is None:
        return info

    pin = _validate_requested_commit_pin(pin)
    info["wwpgd_commit_pin_requested"] = pin
    resolved = str(info.get("wwpgd_resolved_commit") or "").strip().lower()
    if not resolved:
        raise RuntimeError(
            f"{WWPGD_COMMIT_PIN_ENV}={pin!r} is set but the installed ww_pgd "
            "package has no VCS commit in PEP 610 direct_url.json. Install "
            "ww_pgd from Git or clear the pin."
        )
    if _HEXADECIMAL_COMMIT.fullmatch(resolved) is None:
        raise RuntimeError(
            "the installed ww_pgd PEP 610 commit_id is not a hexadecimal Git SHA: "
            f"{resolved!r}"
        )
    if not resolved.startswith(pin):
        raise RuntimeError(
            f"ww_pgd commit pin mismatch: {WWPGD_COMMIT_PIN_ENV}={pin!r} "
            f"but resolved commit is {resolved!r}"
        )
    info["wwpgd_resolved_commit"] = resolved
    info["wwpgd_commit_pin_verified"] = True
    info["wwpgd_commit_pin_status"] = "verified"
    return info


def resolve_and_verify_pip_wwpgd_provenance(
    provenance: Mapping[str, Any] | None = None,
    *,
    requested_pin: str | None = None,
) -> dict[str, Any]:
    """Resolve fresh provenance and apply the optional runtime commit check."""
    base = (
        _resolve_pip_wwpgd_base_provenance()
        if provenance is None
        else dict(provenance)
    )
    return verify_wwpgd_commit_pin(base, requested_pin=requested_pin)


class _LiveWWPGDProvenance(Mapping[str, Any]):
    """Mapping whose pin fields reflect the environment at access time.

    ``wwgpt.ww`` retains a provenance object at import time. Keeping the package
    metadata stable while evaluating the pin dynamically prevents notebooks or
    long-lived processes from verifying one environment state and recording a
    different, stale state in their manifests.
    """

    def __init__(self, base: Mapping[str, Any]):
        self._base = dict(base)

    def snapshot(self) -> dict[str, Any]:
        return verify_wwpgd_commit_pin(self._base)

    def __getitem__(self, key: str) -> Any:
        return self.snapshot()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.snapshot())

    def __len__(self) -> int:
        return len(self.snapshot())

    def __repr__(self) -> str:
        return repr(self.snapshot())


def resolve_pip_wwpgd_provenance() -> Mapping[str, Any]:
    """Return live manifest provenance with explicit pin verification status."""
    return _LiveWWPGDProvenance(_resolve_pip_wwpgd_base_provenance())


def assert_wwpgd_commit_pin(
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility name for explicit runtime verification."""
    return resolve_and_verify_pip_wwpgd_provenance(provenance)


def enforce_wwpgd_commit_pin() -> None:
    """Fail at a shared adapter boundary only when a pin was requested."""
    if _requested_wwpgd_commit_pin() is not None:
        resolve_and_verify_pip_wwpgd_provenance()


def construct_pip_wwpgd_config(spec: object) -> tuple[object, dict[str, Any]]:
    """Map every mathematical experiment option into the installed config."""
    enforce_wwpgd_commit_pin()
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
    has_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in params.values()
    )
    missing = sorted(
        name
        for name in REQUIRED_CONFIG_OPTIONS
        if name not in params and not has_kwargs
    )
    if missing:
        raise RuntimeError(
            "incompatible pip-installed ww_pgd WWTailConfig; required options "
            f"unsupported: {missing}"
        )
    if limit is not None:
        if "max_relative_frobenius_change" not in params and not has_kwargs:
            raise RuntimeError(
                "pip-installed ww_pgd cannot enforce requested "
                "max_relative_frobenius_change"
            )
        requested["max_relative_frobenius_change"] = float(limit)
    config = api["config_class"](**requested)
    resolved = {
        name: getattr(config, name, value) for name, value in requested.items()
    }
    ignored = [
        name for name, value in requested.items() if resolved[name] != value
    ]
    if ignored:
        raise RuntimeError(
            f"pip-installed ww_pgd silently changed requested options: {ignored}"
        )
    return config, {"requested": requested, "resolved": resolved}


def run_pip_wwpgd_candidate(
    model: object,
    config: object,
    **kwargs: Any,
) -> dict[str, Any]:
    """Invoke the installed projector exactly once with its supported diagnostics."""
    enforce_wwpgd_commit_pin()
    api = inspect_pip_wwpgd_api()
    ww_logs: list[Any] = []
    diagnostics: list[dict[str, Any]] = []
    call = dict(kwargs, ww_logs=ww_logs)
    if api["native_internal_diagnostics"]:
        call["diagnostic_logs"] = diagnostics
    api["projector"](model, config, **call)
    return {
        "ww_logs": ww_logs,
        "diagnostic_logs": diagnostics,
        "native_internal_diagnostics": api["native_internal_diagnostics"],
    }
