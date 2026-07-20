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
    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(nb, timeout=120, kernel_name="python3").execute(cwd=str(Path.cwd()))

    analysis = results / "analysis"
    expected_csv = {
        "spectral_records_scientific.csv",
        "metrics_records_scientific.csv",
        "projection_records_scientific.csv",
        "spectral_validity_audit.csv",
        "projected_layer_alpha_summary.csv",
        "run_snapshot_alpha_summary.csv",
        "weightwatcher_plot_manifest.csv",
    }
    for name in expected_csv:
        path = analysis / name
        assert path.exists() and path.stat().st_size >= 0

    manifest = pd.read_csv(analysis / "weightwatcher_plot_manifest.csv")
    assert not manifest.empty
    assert manifest["png"].map(lambda p: Path(p).exists()).all()
    assert manifest["pdf"].map(lambda p: Path(p).exists()).all()
    assert manifest["data"].map(lambda p: Path(p).exists()).all()
    assert manifest["metadata"].map(lambda p: Path(p).exists()).all()
    metadata = json.loads(Path(manifest["metadata"].iloc[0]).read_text())
    assert metadata["png_dpi"] >= 300
    assert "sample standard deviation" in metadata["band_definition"] or "Student-t" in metadata["band_definition"]


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
