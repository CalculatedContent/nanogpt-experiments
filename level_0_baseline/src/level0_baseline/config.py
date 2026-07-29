from __future__ import annotations
import os
from pathlib import Path
import yaml

DEFAULT_ROOT = Path("/tmp/nanogpt-level0")

def roots() -> dict[str, Path]:
    root = Path(os.getenv("NANOGPT_LEVEL0_ROOT", DEFAULT_ROOT))
    return {
        "root": root,
        "data": Path(os.getenv("NANOGPT_LEVEL0_DATA_ROOT", root / "data")),
        "results": Path(os.getenv("NANOGPT_LEVEL0_RESULTS_ROOT", root / "results")),
        "cache": Path(os.getenv("NANOGPT_LEVEL0_CACHE_ROOT", root / "cache")),
    }

def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_map = {
        "NANOGPT_LEVEL0_SEED": ("training", "seed", int),
        "NANOGPT_LEVEL0_OPTIMIZER": ("training", "optimizer", str),
        "NANOGPT_LEVEL0_MAX_STEPS": ("training", "max_steps", int),
        "NANOGPT_LEVEL0_BATCH_SIZE": ("training", "batch_size", int),
        "NANOGPT_LEVEL0_LR": ("training", "learning_rate", float),
        "NANOGPT_LEVEL0_EVAL_INTERVAL": ("training", "eval_interval", int),
    }
    for name, (section, key, cast) in env_map.items():
        if name in os.environ:
            cfg[section][key] = cast(os.environ[name])
    return cfg
