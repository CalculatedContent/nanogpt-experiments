from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from wwgpt.analysis import discover_canonical_runs, paired_extension_effects


def _run(
    root: Path,
    pair: str,
    arm: str,
    base: str,
    ext: str,
    seed: int,
    profile: str = "reproduction_fineweb",
    *,
    level: int = 0,
    token_multiplier: int = 20,
):
    d = (
        root
        / "experiments"
        / f"level_{level:02d}"
        / f"multiplier_{token_multiplier}"
        / pair
        / arm
        / "run_001"
    )
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
        "level": level,
        "token_multiplier": token_multiplier,
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


def test_nested_level_and_multiplier_discovery_is_not_level0_specific(tmp_path: Path):
    _run(
        tmp_path,
        "pair_9",
        "adamw",
        "adamw",
        "none",
        9,
        level=2,
        token_multiplier=7,
    )
    _run(
        tmp_path,
        "pair_9",
        "adamw_wwpgd",
        "adamw",
        "wwpgd",
        9,
        level=2,
        token_multiplier=7,
    )
    found = discover_canonical_runs(tmp_path)
    assert {(row["level"], row["token_multiplier"]) for row in found} == {(2, 7)}
    assert {row["extension"] for row in found} == {"none", "wwpgd"}


def test_multiple_levels_with_same_seed_remain_distinct_design_cells(tmp_path: Path):
    for level in (0, 1, 2):
        pair = f"pair_level_{level}"
        _run(tmp_path, pair, "adamw", "adamw", "none", 1337, level=level)
        _run(
            tmp_path,
            pair,
            "adamw_wwpgd",
            "adamw",
            "wwpgd",
            1337,
            level=level,
        )
    found = discover_canonical_runs(tmp_path)
    assert len(found) == 6
    assert {(row["level"], row["extension"]) for row in found} == {
        (level, extension)
        for level in (0, 1, 2)
        for extension in ("none", "wwpgd")
    }


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
    for profile_root, profile in (
        (tmp_path / "repro", "reproduction_fineweb"),
        (tmp_path / "scale", "scaling"),
    ):
        _run(profile_root, "pair_1", "adamw", "adamw", "none", 1, profile)
        _run(
            profile_root,
            "pair_1",
            "adamw_wwpgd",
            "adamw",
            "wwpgd",
            1,
            profile,
        )
        assert len(discover_canonical_runs(profile_root)) == 2
    assert discover_canonical_runs(tmp_path) == []


def test_cli_help_lists_profiles_and_commands():
    res = subprocess.run([sys.executable, "-m", "wwgpt.cli", "--help"], text=True, capture_output=True, check=True)
    help_text = res.stdout
    for text in ["reproduction_tiny", "reproduction_fineweb", "scaling", "run-multiseed", "analyze-results"]:
        assert text in help_text
    assert "run-strength-scan" not in help_text
    assert "analyze-strength-scan" not in help_text
    assert "audit-strength-scan" not in help_text


def test_ci_workflow_contains_acceptance_commands():
    ci = Path(".github/workflows/ci.yml").read_text()
    for cmd in [
        "python -m compileall -q src tests",
        "ruff check src tests",
        "pytest -q -ra --tb=short",
    ]:
        assert cmd in ci
    assert '-m "not slow"' not in ci


def test_ci_has_required_schema_v3_papermill_notebook_job():
    ci = Path(".github/workflows/ci.yml").read_text()
    assert "analysis-notebooks:" in ci
    assert "continue-on-error" not in ci
    assert "|| true" not in ci
    assert "tests/fixtures/schema_v3_results" in ci
    assert "./scripts/run_analysis_notebooks.sh" in ci
    assert "schema-v2" not in ci
    assert "test_schema_v2_analysis.py" not in ci
    assert "actions/upload-artifact" in ci
    assert "if: always()" in ci
    for env_name in [
        "WWGPT_RESULTS_ROOT",
        "WWGPT_NOTEBOOK_OUTPUT_DIR",
        "WWGPT_ANALYSIS_PLAN",
        "WWGPT_LEVEL",
        "WWGPT_TOKEN_MULTIPLIER",
        "MPLBACKEND",
    ]:
        assert env_name in ci


def test_clean_install_imports_required_packages():
    import importlib

    for module in ["wwgpt", "optimi", "ww_pgd", "weightwatcher", "tiktoken"]:
        importlib.import_module(module)
