from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO / "experiment_2"


def test_notebooks_are_valid_v4_json() -> None:
    expected = {
        "01_protocol_audit.ipynb",
        "02_compare_paired.ipynb",
        "03_layer_controller_diagnostics.ipynb",
        "04_cross_scale_summary.ipynb",
    }
    found = {path.name for path in (EXPERIMENT / "notebooks").glob("*.ipynb")}
    assert found == expected
    for path in sorted((EXPERIMENT / "notebooks").glob("*.ipynb")):
        payload = json.loads(path.read_text())
        assert payload["nbformat"] == 4
        assert payload["cells"]
        assert all(cell["cell_type"] in {"markdown", "code", "raw"} for cell in payload["cells"])


def test_shell_entrypoints_are_isolated_and_source_safe() -> None:
    expected = {"prepare_data.sh", "run_one.sh", "run_pair.sh"}
    found = {path.name for path in (EXPERIMENT / "scripts").glob("*.sh")}
    assert found == expected
    for path in sorted((EXPERIMENT / "scripts").glob("*.sh")):
        source = path.read_text()
        assert source.startswith("#!/usr/bin/env bash")
        assert 'BASH_SOURCE[0]' in source
        assert "do not source" in source
        assert "set -euo pipefail" in source


def test_experiment_tree_does_not_shadow_existing_level_directories() -> None:
    assert EXPERIMENT.is_dir()
    assert not (EXPERIMENT / "level_0_baseline").exists()
    assert not (EXPERIMENT / "level_0_wwpgd").exists()
    assert not (EXPERIMENT / "level_1_baseline").exists()
    assert not (EXPERIMENT / "level_1_wwpgd").exists()
