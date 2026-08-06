from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import torch

from .model import GPT, transformer_matrix_items


def write_manifest(
    run_dir: Path,
    *,
    cfg: dict[str, Any],
    optimizer_name: str,
    profile: dict[str, Any],
    warmup_steps: int,
    device: torch.device,
    model: GPT,
    data_root: Path,
    data_metadata: dict[str, Any],
    tokens_per_step: int,
    batch_size: int,
    grad_accum_steps: int,
    planned_tokens: int,
    planned_epochs: float,
    target_epochs: float,
    fingerprint: str,
    train_tokens: int,
    val_tokens: int,
    test_tokens: int,
) -> Path:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    matrix_parameters = sum(
        weight.numel() for _, _, _, weight in transformer_matrix_items(model)
    )
    manifest = {
        "schema_version": 3,
        "protocol_fingerprint": fingerprint,
        "config": cfg,
        "optimizer_name": optimizer_name,
        "optimizer_profile": profile,
        "warmup_steps": int(warmup_steps),
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "parameter_count": total_parameters,
        "transformer_matrix_parameter_count": matrix_parameters,
        "data_root": str(data_root.resolve()),
        "data_manifest": data_metadata,
        "tokens_per_optimizer_step": int(tokens_per_step),
        "effective_batch_sequences": int(batch_size) * int(grad_accum_steps),
        "planned_training_tokens": int(planned_tokens),
        "planned_train_epochs": float(planned_epochs),
        "target_train_epochs": float(target_epochs),
        "test_evaluation_policy": "final_and_validation_selected_checkpoint_only",
        "evaluation_sampling": "fixed_probes_with_rng_streams_independent_of_training",
        "precision_policy": "float32",
        "mps_compile_default": False,
        "split_sizes": {
            "train": int(train_tokens),
            "validation": int(val_tokens),
            "test": int(test_tokens),
        },
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
