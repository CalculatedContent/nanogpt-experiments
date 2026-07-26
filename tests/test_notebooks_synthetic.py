from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient


@pytest.mark.notebook
def test_synthetic_notebook_executes_offline(tmp_path: Path):
    """Execute a tiny notebook that exercises repo imports without network or large data."""
    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("from wwgpt.config import ModelConfig\nfrom wwgpt.model import GPT"),
            nbformat.v4.new_code_cell("model = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, vocab_size=16))\nassert model.parameter_report().total_parameters > 0"),
            nbformat.v4.new_code_cell("from pathlib import Path\nfrom wwgpt.analysis import load_run_artifacts\nroot = Path('tests/fixtures/schema_v3_results/experiments/level_00/multiplier_1/pair_7')\nruns = [load_run_artifacts(root / arm / 'run_fixture') for arm in ('adamw', 'adamw_wwpgd')]\nassert all(r['manifest']['scientific_schema_version'] == 3 for r in runs)\nassert all(not r['alpha_measurements'].empty for r in runs)"),
        ]
    )
    NotebookClient(nb, timeout=30, kernel_name="python3").execute(cwd=str(Path.cwd()))
