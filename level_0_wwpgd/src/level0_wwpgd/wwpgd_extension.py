from __future__ import annotations

import copy
import csv
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .model import GPT

PROJECTION_FIELDS = [
    "optimizer_step",
    "tokens_seen",
    "projection_event",
    "layer_name",
    "matrix_type",
    "block",
    "alpha_before",
    "D",
    "xmin",
    "num_evals",
    "target_alpha",
    "blend_eta",
    "cayley_eta",
    "min_tail",
    "use_detx",
    "candidate_device",
    "candidate_relative_frobenius_change",
    "relative_frobenius_change_requested",
    "relative_frobenius_change_applied",
    "trust_region_scale",
    "changed",
    "projection_runtime_seconds",
]


def projected_modules(model: GPT) -> list[tuple[str, str, int, nn.Linear]]:
    result: list[tuple[str, str, int, nn.Linear]] = []
    for block_index, block in enumerate(model.blocks):
        result.extend(
            [
                (f"blocks.{block_index}.attn.q_proj", "W_Q", block_index, block.attn.q_proj),
                (f"blocks.{block_index}.attn.k_proj", "W_K", block_index, block.attn.k_proj),
                (f"blocks.{block_index}.attn.v_proj", "W_V", block_index, block.attn.v_proj),
                (f"blocks.{block_index}.attn.out_proj", "W_O", block_index, block.attn.out_proj),
                (f"blocks.{block_index}.mlp.fc", "W_MLP_IN", block_index, block.mlp.fc),
                (f"blocks.{block_index}.mlp.proj", "W_MLP_OUT", block_index, block.mlp.proj),
            ]
        )
    return result


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


def _match_weightwatcher_row(frame: pd.DataFrame, layer_name: str) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    for _, row in frame.iterrows():
        text = " ".join(str(row.get(column, "")) for column in ("longname", "name"))
        if layer_name == text or layer_name in text or text.endswith(layer_name):
            return row.to_dict()
    return {}


def append_projection_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROJECTION_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in PROJECTION_FIELDS})


class WWPGDExtension:
    """Fresh stock WWPGD event projection after each scheduled AdamW update."""

    def __init__(self, model: GPT, config: dict[str, Any]):
        self.model = model
        self.config = dict(config)
        self.call_count = 0
        self.projected_matrix_count = 0
        self.runtime_seconds = 0.0
        self._resolved_config = None

    def _adapter(self):
        try:
            from wwgpt.ww import (
                _external_config_object,
                _external_wwpgd_module,
                external_wwpgd_config_from_experiment,
                external_wwpgd_manifest_fields,
                run_pip_wwpgd_candidate,
            )
        except Exception as exc:
            raise RuntimeError(
                "The repository WWPGD adapter is unavailable. Install the repository root "
                "with `python -m pip install -e .` before running this experiment."
            ) from exc
        return (
            _external_config_object,
            _external_wwpgd_module,
            external_wwpgd_config_from_experiment,
            external_wwpgd_manifest_fields,
            run_pip_wwpgd_candidate,
        )

    def resolved_config(self):
        if self._resolved_config is None:
            _, _, resolver, _, _ = self._adapter()
            requested = SimpleNamespace(
                enabled=True,
                extension="wwpgd",
                target_alpha=float(self.config["target_alpha"]),
                blend_eta=float(self.config["blend_eta"]),
                cayley_eta=float(self.config["cayley_eta"]),
                min_tail=int(self.config["min_tail"]),
                use_detx=bool(self.config["use_detx"]),
                verbose=False,
                candidate_device="cpu",
                max_relative_frobenius_change=self.config.get(
                    "max_relative_frobenius_change"
                ),
            )
            self._resolved_config = resolver(requested)
        return self._resolved_config

    def manifest_fields(self) -> dict[str, Any]:
        _, _, _, manifest_builder, _ = self._adapter()
        requested = SimpleNamespace(
            enabled=True,
            extension="wwpgd",
            target_alpha=float(self.config["target_alpha"]),
            blend_eta=float(self.config["blend_eta"]),
            cayley_eta=float(self.config["cayley_eta"]),
            min_tail=int(self.config["min_tail"]),
            use_detx=bool(self.config["use_detx"]),
            verbose=False,
            candidate_device="cpu",
            max_relative_frobenius_change=self.config.get(
                "max_relative_frobenius_change"
            ),
        )
        fields = manifest_builder(True, requested)
        fields.update(
            {
                "extension": "wwpgd",
                "wwpgd_apply_mode": "event_projection",
                "wwpgd_interval": int(self.config["interval"]),
                "wwpgd_scope": "transformer_block_matrices_only",
                "wwpgd_candidate_device": "cpu",
            }
        )
        return fields

    def _cpu_candidate_model(self) -> GPT:
        with preserve_global_rng():
            clone = GPT(copy.deepcopy(self.model.cfg)).to(torch.device("cpu"))
        clone.load_state_dict(
            {name: value.detach().cpu() for name, value in self.model.state_dict().items()}
        )
        clone.train(self.model.training)
        return clone

    def after_optimizer_step(
        self,
        *,
        optimizer_step: int,
        total_optimizer_steps: int,
        tokens_seen: int,
    ) -> list[dict[str, Any]]:
        interval = int(self.config["interval"])
        if optimizer_step % interval != 0:
            return []

        (
            external_config_object,
            external_module,
            _,
            _,
            run_candidate,
        ) = self._adapter()
        resolved = self.resolved_config()
        candidate_model = self._cpu_candidate_model()
        selected = projected_modules(candidate_model)
        selected_names = {name for name, _, _, _ in selected}

        def selector(mm: nn.Module, layer_name: str, row: object | None = None):
            del row
            match = next(
                (
                    name
                    for name in selected_names
                    if layer_name == name
                    or str(layer_name).endswith(name)
                    or name in str(layer_name)
                ),
                None,
            )
            return _module_by_name(mm, match) if match is not None else None

        started = time.perf_counter()
        with preserve_global_rng(), torch.no_grad():
            result = run_candidate(
                candidate_model,
                external_config_object(external_module(), resolved),
                epoch=self.call_count,
                num_epochs=max(self.call_count + 1, 1),
                global_step=optimizer_step,
                layer_selector=selector,
            )
        runtime = time.perf_counter() - started
        result = result if isinstance(result, dict) else {}
        usable_logs = [
            item
            for item in result.get("ww_logs", [])
            if isinstance(item, pd.DataFrame) and not item.empty
        ]
        if len(usable_logs) != 1:
            raise RuntimeError(
                "stock WWPGD projection expected exactly one nonempty ww_logs frame; "
                f"found {len(usable_logs)}"
            )
        details = usable_logs[0]

        live = {name: module for name, _, _, module in projected_modules(self.model)}
        candidates = {
            name: module.weight.detach().cpu().clone()
            for name, _, _, module in projected_modules(candidate_model)
        }
        rows: list[dict[str, Any]] = []
        limit = self.config.get("max_relative_frobenius_change")

        with torch.no_grad():
            for layer_name, matrix_type, block, live_module in projected_modules(self.model):
                original = live_module.weight.detach().clone()
                candidate = candidates[layer_name].to(
                    original.device, dtype=original.dtype
                )
                requested_delta = candidate - original
                denominator = max(float(original.float().norm()), 1e-12)
                requested_relative = float(
                    requested_delta.float().norm() / denominator
                )
                trust_scale = 1.0
                if (
                    limit is not None
                    and math.isfinite(requested_relative)
                    and requested_relative > float(limit)
                ):
                    trust_scale = float(limit) / max(requested_relative, 1e-12)
                applied = original + trust_scale * requested_delta
                live_module.weight.copy_(applied)
                applied_relative = float(
                    (live_module.weight.detach() - original).float().norm()
                    / denominator
                )
                observed = _match_weightwatcher_row(details, layer_name)
                rows.append(
                    {
                        "optimizer_step": optimizer_step,
                        "tokens_seen": tokens_seen,
                        "projection_event": self.call_count,
                        "layer_name": layer_name,
                        "matrix_type": matrix_type,
                        "block": block,
                        "alpha_before": observed.get("alpha"),
                        "D": observed.get("D"),
                        "xmin": observed.get("xmin"),
                        "num_evals": observed.get("num_evals"),
                        "target_alpha": self.config["target_alpha"],
                        "blend_eta": self.config["blend_eta"],
                        "cayley_eta": self.config["cayley_eta"],
                        "min_tail": self.config["min_tail"],
                        "use_detx": self.config["use_detx"],
                        "candidate_device": "cpu",
                        "candidate_relative_frobenius_change": requested_relative,
                        "relative_frobenius_change_requested": requested_relative,
                        "relative_frobenius_change_applied": applied_relative,
                        "trust_region_scale": trust_scale,
                        "changed": applied_relative > 0.0,
                        "projection_runtime_seconds": runtime / max(len(live), 1),
                    }
                )

        self.call_count += 1
        self.projected_matrix_count += sum(bool(row["changed"]) for row in rows)
        self.runtime_seconds += runtime
        if (
            self.call_count == 1
            or optimizer_step % int(self.config.get("log_interval", 10)) == 0
            or optimizer_step == total_optimizer_steps
        ):
            mean_change = float(
                np.mean([row["relative_frobenius_change_applied"] for row in rows])
            )
            print(
                "[level0-wwpgd] "
                f"step={optimizer_step}/{total_optimizer_steps} "
                f"event={self.call_count - 1} matrices={len(rows)} "
                f"mean_relative_change={mean_change:.3e} runtime_s={runtime:.2f}",
                flush=True,
            )
        return rows
