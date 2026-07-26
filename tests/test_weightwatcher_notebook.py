from __future__ import annotations

import json
import shutil
from pathlib import Path

import nbformat
import pandas as pd
import pytest
from nbclient import NotebookClient

from wwgpt.analysis import audit_spectral_validity, load_run_artifacts, normalize_spectral_records

FIXTURE_ROOT = Path("tests/fixtures/schema_v2_results/experiments/level_00/multiplier_20").resolve()
NOTEBOOK = Path("notebooks/03_weightwatcher_analysis.ipynb")


def test_weightwatcher_notebook_parses_compiles_and_has_no_stale_globals():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    assert "OPTIMIZER_COLORS" not in source
    assert all(cell.get("id") for cell in nb.cells)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, f"{NOTEBOOK}:{cell.get('id')}", "exec")
            assert cell.execution_count is None
            assert cell.outputs == []


def test_weightwatcher_notebook_executes_fixture_and_exports_expected_files(tmp_path, monkeypatch):
    results = tmp_path / "schema_v2_results"
    shutil.copytree(FIXTURE_ROOT, results)
    monkeypatch.setenv("WWGPT_RESULTS_ROOT", str(results))
    output = tmp_path / "notebook-output"
    monkeypatch.setenv("WWGPT_NOTEBOOK_OUTPUT_DIR", str(output))
    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(nb, timeout=120, kernel_name="python3").execute(cwd=str(Path.cwd()))

    analysis = output / "tables"
    expected_csv = {"scientific_alpha.csv", "trap_diagnostics.csv"}
    for name in expected_csv:
        path = analysis / name
        assert path.exists() and path.stat().st_size >= 0

def test_spectral_validity_audit_refuses_unverifiable_rows():
    run = next(FIXTURE_ROOT.glob("pair_1337*/adamw/run_*"))
    spectral = normalize_spectral_records(load_run_artifacts(run)["spectral"]).assign(
        seed=1337,
        pair_id="pair_1337_20260715",
        optimizer_family="adamw",
        valid_for_science=True,
        scientific_schema_version=2,
    )
    spectral.loc[0, "spectral_estimator"] = "fallback_non_scientific"
    audit = audit_spectral_validity(spectral)
    assert not audit["valid_for_weightwatcher_science"].all()
    assert audit.loc[0, "invalid_reasons"].find("spectral_estimator_not_weightwatcher") >= 0
