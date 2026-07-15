from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def unique_dir(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"{prefix}_{utc_stamp()}_{uuid.uuid4().hex[:8]}"
    if path.exists():
        raise FileExistsError(f"unique path collision: {path}")
    path.mkdir(parents=True)
    return path


def write_json(path: Path, data: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict[str, Any]:
    return {
        "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        "backend": "pytorch",
        "precision": "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32",
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "operating_system": platform.platform(),
        "processor": platform.processor(),
        "cuda_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "xla_version": None,
    }
