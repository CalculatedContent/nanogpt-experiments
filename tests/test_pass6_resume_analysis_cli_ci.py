from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from wwgpt.analysis import discover_canonical_runs, paired_extension_effects


def _run(root: Path, pair: str, arm: str, base: str, ext: str, seed: int, profile: str = "reproduction_fineweb"):
    d = root / "experiments" / "level_00" / "multiplier_20" / pair / arm / "run_001"
    d.mkdir(parents=True)
    man = {
        "scientific_schema_version": 3,
        "valid_for_science": True,
        "seed": seed,
        "pair_id": pair,
        "optimizer": arm,
        "arm_name": arm,
        "base_optimizer": base,
        "extension": ext,
        "initialization_hash": f"init-{seed}",
        "tokenizer_hash": "tok",
        "data_hash": f"data-{profile}",
        "validation_probe_hash": "v",
        "training_probe_hash": "t",
        "realized_tokens": 100,
        "level": 0,
        "token_multiplier": 20,
        "experiment_profile": profile,
    }
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "run_complete.json").write_text("{}")
    pd.DataFrame([{"step": 1, "tokens_seen": 100, "validation_loss": 1.0}]).to_csv(d / "metrics.csv", index=False)
    return d


def test_all_six_arm_discovery(tmp_path: Path):
    arms = [
        ("adamw", "adamw", "none"),
        ("adamw_wwpgd", "adamw", "wwpgd"),
        ("muon", "muon", "none"),
        ("muon_wwpgd", "muon", "wwpgd"),
        ("stableadamw", "stableadamw", "none"),
        ("stableadamw_wwpgd", "stableadamw", "wwpgd"),
    ]
    for arm, base, ext in arms:
        _run(tmp_path, "pair_1", arm, base, ext, 1)
    found = discover_canonical_runs(tmp_path)
    assert {r["optimizer_family"] for r in found} == {a for a, _, _ in arms}


def test_paired_effects_never_cross_base_optimizer():
    df = pd.DataFrame([
        {"scientific_schema_version": 3, "level": 0, "token_multiplier": 20, "base_optimizer": "adamw", "extension": "none", "seed": 1, "loss": 10.0},
        {"scientific_schema_version": 3, "level": 0, "token_multiplier": 20, "base_optimizer": "adamw", "extension": "wwpgd", "seed": 1, "loss": 9.0},
        {"scientific_schema_version": 3, "level": 0, "token_multiplier": 20, "base_optimizer": "muon", "extension": "none", "seed": 1, "loss": 1.0},
        {"scientific_schema_version": 3, "level": 0, "token_multiplier": 20, "base_optimizer": "muon", "extension": "wwpgd", "seed": 1, "loss": 3.0},
    ])
    out = paired_extension_effects(df, "loss")
    assert set(out["paired_comparison"]) == {"AdamW+WW-PGD - AdamW", "Muon+WW-PGD - Muon"}
    assert set(out["wwpgd_minus_none_loss"]) == {-1.0, 2.0}


def test_profile_isolation_and_no_composite_pooling_by_default(tmp_path: Path):
    _run(tmp_path / "repro", "pair_1", "adamw", "adamw", "none", 1, "reproduction_fineweb")
    _run(tmp_path / "scale", "pair_1", "adamw", "adamw", "none", 1, "scaling")
    assert len(discover_canonical_runs(tmp_path / "repro")) == 0  # no within-base pair, not pooled
    assert discover_canonical_runs(tmp_path) == []


def test_cli_help_lists_profiles_and_commands():
    res = subprocess.run([sys.executable, "-m", "wwgpt.cli", "--help"], text=True, capture_output=True, check=True)
    help_text = res.stdout
    for text in ["reproduction_tiny", "reproduction_fineweb", "scaling", "run-multiseed", "analyze-results", "run-strength-scan"]:
        assert text in help_text


def test_ci_workflow_contains_acceptance_commands():
    ci = Path(".github/workflows/ci.yml").read_text()
    for cmd in ["python -m compileall -q src tests", "ruff check src tests", "pytest -q -m \"not slow\""]:
        assert cmd in ci

def test_clean_install_imports_required_packages():
    import importlib

    for module in ["wwgpt", "optimi", "ww_pgd", "weightwatcher", "tiktoken"]:
        importlib.import_module(module)
