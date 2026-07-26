from __future__ import annotations

import dataclasses
import inspect
import json
import math
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any


_INTERNAL_ONLY_FIELDS = (
    "k_pl",
    "k_detx",
    "k_star",
    "selected_lambda_threshold",
    "selected_tail_size",
    "target_normalization_A",
    "trace_log_before",
    "trace_log_target",
    "trace_log_after_cayley_before_retraction",
    "trace_log_after_retraction",
    "trace_log_retraction_residual",
    "trace_log_retraction_absolute_error",
    "trace_log_retraction_relative_error",
    "trace_log_retraction_tolerance",
    "trace_log_retraction_pass",
    "trace_log_after_final_blend",
    "trace_log_final_blend_delta_from_original",
    "cayley_raw_ratio_min",
    "cayley_raw_ratio_max",
    "cayley_applied_ratio_min",
    "cayley_applied_ratio_max",
    "cayley_low_clip_count",
    "cayley_high_clip_count",
    "cayley_nonfinite_raw_ratio_count",
    "cayley_denominator_min_absolute_value",
    "shaped_weight_frobenius_norm",
    "shaped_relative_frobenius_change",
    "maximum_absolute_singular_value_change",
    "mean_absolute_singular_value_change",
    "maximum_relative_tail_eigenvalue_change",
    "mean_relative_tail_eigenvalue_change",
)


def _distribution() -> metadata.Distribution | None:
    for name in ("ww-pgd", "ww_pgd"):
        try:
            return metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def resolve_wwpgd_provenance() -> dict[str, Any]:
    """Describe the package pip actually installed without imposing a revision."""
    import ww_pgd

    dist = _distribution()
    version = getattr(ww_pgd, "__version__", None)
    if not version and dist is not None:
        version = dist.version
    version = str(version or "unknown")

    source_url = ""
    resolved_commit = ""
    install_mode = "installed-package"
    if dist is not None:
        raw = dist.read_text("direct_url.json")
        if raw:
            try:
                direct = json.loads(raw)
                source_url = str(direct.get("url") or "")
                vcs = direct.get("vcs_info") or {}
                resolved_commit = str(vcs.get("commit_id") or "")
                if vcs:
                    install_mode = "pip-vcs"
                elif (direct.get("dir_info") or {}).get("editable"):
                    install_mode = "pip-editable"
                else:
                    install_mode = "pip-direct-url"
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

    if not resolved_commit:
        module_path = Path(getattr(ww_pgd, "__file__", "")).resolve()
        for parent in (module_path.parent, *module_path.parents):
            if not (parent / ".git").exists():
                continue
            try:
                resolved_commit = subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                source_url = source_url or str(parent)
                install_mode = "editable-git-checkout"
            except (OSError, subprocess.CalledProcessError):
                pass
            break

    return {
        "wwpgd_source_repository": "CalculatedContent/WW_PGD",
        "wwpgd_source_url": source_url,
        "wwpgd_installed_version": version,
        "wwpgd_resolved_commit": resolved_commit,
        "wwpgd_install_mode": install_mode,
    }


def _row_dict(row: object) -> dict[str, Any]:
    if hasattr(row, "to_dict"):
        return dict(row.to_dict())
    if isinstance(row, dict):
        return dict(row)
    return {}


def _compatibility_diagnostic(
    *,
    layer_name: str,
    row: object,
    module: object,
    cfg: object,
    epoch: int,
    global_step: int | None,
) -> dict[str, Any]:
    data = _row_dict(row)
    weight = getattr(module, "weight", None)
    matrix_rows = int(weight.shape[0]) if weight is not None and weight.ndim >= 1 else None
    matrix_columns = int(weight.reshape(weight.shape[0], -1).shape[1]) if weight is not None and weight.ndim >= 2 else None
    candidate_norm = float(weight.detach().float().norm().cpu()) if weight is not None else None
    reason = (
        "installed ww_pgd exposes ww_logs but not diagnostic_logs; exact internal "
        "tail midpoint, Cayley-ratio, and TraceLog-retraction fields are unsupported"
    )
    result: dict[str, Any] = {
        "diagnostics_schema_version": 1,
        "source_package": "ww_pgd",
        "diagnostic_source": "ww_pgd_ww_logs_compatibility_adapter",
        "native_internal_diagnostics": False,
        "layer_name": layer_name,
        "global_step": global_step,
        "event_index": epoch,
        "epoch_id": epoch,
        "status": "unsupported_internal_fields",
        "skip_reason": reason,
        "valid_diagnostic": False,
        "changed": False,
        "matrix_rows": matrix_rows,
        "matrix_columns": matrix_columns,
        "num_singular_values": min(matrix_rows, matrix_columns) if matrix_rows and matrix_columns else None,
        "original_weight_frobenius_norm": None,
        "candidate_weight_frobenius_norm": candidate_norm,
        "alpha": data.get("alpha"),
        "D": data.get("D"),
        "xmin": data.get("xmin"),
        "detX_num": data.get("detX_num"),
        "num_evals": data.get("num_evals"),
        "num_pl_spikes": data.get("num_pl_spikes"),
        "num_ERG_spikes": data.get("num_ERG_spikes"),
        "rand_num_spikes": data.get("rand_num_spikes"),
        "num_traps": data.get("num_traps"),
        "configured_min_tail": getattr(cfg, "min_tail", None),
        "used_detx": getattr(cfg, "use_detx", None),
        "internal_rank_exponent": getattr(cfg, "q", None),
        "configured_cayley_eta": getattr(cfg, "cayley_eta", None),
        "effective_cayley_eta": None,
        "cayley_low_clip_bound": 0.1,
        "cayley_high_clip_bound": 10.0,
        "configured_blend_eta": getattr(cfg, "blend_eta", None),
        "effective_blend_eta": None,
        "candidate_relative_frobenius_change": None,
        "adapter_candidate_relative_frobenius_change": None,
        "adapter_candidate_changed": None,
        "unsupported_internal_fields": ",".join(_INTERNAL_ONLY_FIELDS),
    }
    for field in _INTERNAL_ONLY_FIELDS:
        result.setdefault(field, None)
    return result


def install_wwpgd_api_compatibility() -> dict[str, Any]:
    """Make the pip-installed public WW-PGD API usable without requiring a fork."""
    import ww_pgd

    provenance = resolve_wwpgd_provenance()
    projector = getattr(ww_pgd, "ww_pgd_project")
    if getattr(projector, "__wwgpt_compatibility_wrapper__", False):
        return provenance

    signature = inspect.signature(projector)
    required = {"epoch", "num_epochs", "global_step", "ww_logs", "layer_selector"}
    missing = sorted(required - set(signature.parameters))
    if missing:
        raise RuntimeError(
            "the pip-installed ww_pgd package does not expose the required public API; "
            f"missing parameters: {missing}"
        )
    if "diagnostic_logs" in signature.parameters:
        return provenance

    original = projector

    def compatible_projector(
        model,
        cfg,
        *,
        epoch,
        num_epochs,
        global_step=None,
        ww_logs=None,
        layer_selector=None,
        diagnostic_logs=None,
    ) -> None:
        admitted: list[tuple[str, object, object]] = []

        def recording_selector(mm, layer_name, row=None):
            if layer_selector is None:
                return None
            module = layer_selector(mm, layer_name, row)
            if module is not None:
                admitted.append((str(layer_name), row, module))
            return module

        original(
            model,
            cfg,
            epoch=epoch,
            num_epochs=num_epochs,
            global_step=global_step,
            ww_logs=ww_logs,
            layer_selector=recording_selector if layer_selector is not None else None,
        )
        if diagnostic_logs is not None:
            diagnostic_logs.extend(
                _compatibility_diagnostic(
                    layer_name=name,
                    row=row,
                    module=module,
                    cfg=cfg,
                    epoch=int(epoch),
                    global_step=global_step,
                )
                for name, row, module in admitted
            )

    compatible_projector.__name__ = getattr(original, "__name__", "ww_pgd_project")
    compatible_projector.__doc__ = getattr(original, "__doc__", None)
    compatible_projector.__wrapped__ = original
    compatible_projector.__wwgpt_compatibility_wrapper__ = True
    ww_pgd.ww_pgd_project = compatible_projector
    return provenance


def patch_wwgpt_ww_module(ww_module: object, provenance: dict[str, Any]) -> None:
    """Attach runtime provenance and candidate-level movement to compatibility rows."""
    if getattr(ww_module, "__wwgpt_runtime_patched__", False):
        return

    runtime_identifier = str(
        provenance.get("wwpgd_resolved_commit")
        or f"version:{provenance.get('wwpgd_installed_version', 'unknown')}"
    )
    # Legacy code imports this name. It is now runtime provenance, never a pin.
    ww_module.WWPGD_COMMIT = runtime_identifier

    original_manifest = ww_module.external_wwpgd_manifest_fields

    def manifest_fields(enabled: bool = True, requested_cfg: object | None = None) -> dict[str, Any]:
        fields = original_manifest(enabled, requested_cfg)
        fields.update(provenance)
        fields["wwpgd_commit"] = runtime_identifier if enabled else ""
        fields["wwpgd_dependency_pinned"] = False
        fields["wwpgd_native_internal_diagnostics"] = bool(
            "diagnostic_logs" in inspect.signature(__import__("ww_pgd").ww_pgd_project).parameters
            and not getattr(__import__("ww_pgd").ww_pgd_project, "__wwgpt_compatibility_wrapper__", False)
        )
        return fields

    original_build = ww_module.build_stock_wwpgd_candidate

    def build_candidate(*args, **kwargs):
        candidate = original_build(*args, **kwargs)
        rows: list[dict[str, Any]] = []
        for raw in candidate.internal_diagnostics:
            row = dict(raw)
            layer_name = str(row.get("layer_name", ""))
            relative = candidate.original_to_candidate_relative_change.get(layer_name)
            changed = candidate.stock_candidate_changed.get(layer_name)
            row["adapter_candidate_relative_frobenius_change"] = relative
            if row.get("candidate_relative_frobenius_change") is None:
                row["candidate_relative_frobenius_change"] = relative
            row["adapter_candidate_changed"] = changed
            row["wwpgd_resolved_commit"] = provenance.get("wwpgd_resolved_commit", "")
            row["wwpgd_installed_version"] = provenance.get("wwpgd_installed_version", "unknown")
            row["wwpgd_install_mode"] = provenance.get("wwpgd_install_mode", "installed-package")
            rows.append(row)
        return dataclasses.replace(candidate, internal_diagnostics=rows, stock_commit=runtime_identifier)

    ww_module.external_wwpgd_manifest_fields = manifest_fields
    ww_module.build_stock_wwpgd_candidate = build_candidate
    ww_module.__wwgpt_runtime_patched__ = True
