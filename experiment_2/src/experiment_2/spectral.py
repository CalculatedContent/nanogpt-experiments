from __future__ import annotations

import csv
import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .model import GPT, projected_modules


LAYER_MEASUREMENT_FIELDS = [
    "step",
    "tokens_seen",
    "layer_name",
    "matrix_type",
    "block",
    "alpha",
    "D",
    "xmin",
    "num_evals",
    "alpha_error",
    "alpha_velocity",
    "credit_ema",
    "credit_observations",
    "bad_windows",
    "cooldown_until",
    "eligible",
    "selected",
    "global_paused",
    "last_projection_status",
]

CONTROLLER_WINDOW_FIELDS = [
    "step",
    "previous_step",
    "train_loss",
    "val_loss",
    "baseline_train_loss",
    "baseline_val_loss",
    "adaptive_progress",
    "baseline_progress",
    "progress_advantage",
    "loss_gap",
    "active_layers_in_window",
    "credited_layers",
    "global_bad_windows",
    "global_pause_until",
    "decision_reason",
]

PROJECTION_FIELDS = [
    "optimizer_step",
    "tokens_seen",
    "projection_event",
    "projection_status",
    "error_type",
    "error_message",
    "consecutive_failures",
    "layer_name",
    "matrix_type",
    "block",
    "alpha_before",
    "alpha_after_candidate",
    "alpha_error_before",
    "alpha_error_after_candidate",
    "alpha_improvement",
    "probe_loss_before",
    "probe_loss_after_candidate",
    "probe_loss_delta",
    "target_alpha",
    "adamw_delta_norm",
    "projection_delta_norm_requested",
    "projection_to_adamw_ratio_requested",
    "alignment_cosine",
    "relative_frobenius_change_requested",
    "relative_frobenius_change_applied",
    "trust_region_scale",
    "update_ratio_scale",
    "changed",
    "credit_ema",
    "cooldown_until",
    "candidate_analyzed_matrix_count",
    "projection_runtime_seconds",
]

_TORCH_LINALG_ERRORS = tuple(
    error_type
    for error_type in (getattr(torch._C, "_LinAlgError", None),)
    if isinstance(error_type, type)
)


@contextmanager
def preserve_global_rng() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    mps_state = None
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        getter = getattr(torch.mps, "get_rng_state", None)
        if getter is not None:
            try:
                mps_state = getter()
            except Exception:
                mps_state = None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if mps_state is not None:
            setter = getattr(torch.mps, "set_rng_state", None)
            if setter is not None:
                setter(mps_state)


def append_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _is_retryable_projection_error(exc: BaseException) -> bool:
    if isinstance(exc, np.linalg.LinAlgError):
        return True
    if _TORCH_LINALG_ERRORS and isinstance(exc, _TORCH_LINALG_ERRORS):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "linalg",
        "svd",
        "singular value",
        "failed to converge",
        "ill-conditioned",
        "ill conditioned",
    )
    return isinstance(exc, RuntimeError) and any(marker in text for marker in markers)


def _module_by_name(model: nn.Module, name: str) -> nn.Module | None:
    current: nn.Module = model
    for part in name.split("."):
        if part.isdigit() and isinstance(current, (nn.ModuleList, nn.Sequential)):
            current = current[int(part)]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current if hasattr(current, "weight") else None


class _ProjectedMatrixHolder(nn.Module):
    """CPU-only model containing only the matrices selected for one intervention."""

    def __init__(self, model: GPT, selected: set[str] | None = None):
        super().__init__()
        self.safe_to_live: dict[str, str] = {}
        self.metadata: dict[str, tuple[str, int]] = {}
        for live_name, matrix_type, block, live_module in projected_modules(model):
            if selected is not None and live_name not in selected:
                continue
            safe_name = f"L{block:02d}_{matrix_type}"
            layer = nn.Linear(
                live_module.weight.shape[1], live_module.weight.shape[0], bias=False
            )
            layer.weight = nn.Parameter(
                live_module.weight.detach().float().cpu().clone(), requires_grad=False
            )
            self.add_module(safe_name, layer)
            self.safe_to_live[safe_name] = live_name
            self.metadata[safe_name] = (matrix_type, block)


def _match_weightwatcher_row(frame: pd.DataFrame, safe_name: str) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    for _, row in frame.iterrows():
        text = " ".join(str(row.get(column, "")) for column in ("longname", "name"))
        if safe_name == text or safe_name in text or text.endswith(safe_name):
            return row.to_dict()
    return {}


def _analyze_holder(holder: _ProjectedMatrixHolder) -> pd.DataFrame:
    import weightwatcher as ww

    with preserve_global_rng(), torch.no_grad():
        frame = ww.WeightWatcher(model=holder).analyze(
            detX=True,
            randomize=False,
            plot=False,
        )
    if frame is None or len(frame) == 0:
        raise RuntimeError("WeightWatcher returned no rows")
    return frame.copy()


def measure_model_layers(
    model: GPT,
    *,
    step: int,
    tokens_seen: int,
) -> list[dict[str, Any]]:
    holder = _ProjectedMatrixHolder(model)
    frame = _analyze_holder(holder)
    rows: list[dict[str, Any]] = []
    for safe_name, live_name in holder.safe_to_live.items():
        observed = _match_weightwatcher_row(frame, safe_name)
        matrix_type, block = holder.metadata[safe_name]
        rows.append(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "layer_name": live_name,
                "matrix_type": matrix_type,
                "block": block,
                "alpha": observed.get("alpha"),
                "D": observed.get("D"),
                "xmin": observed.get("xmin"),
                "num_evals": observed.get("num_evals"),
            }
        )
    return rows
