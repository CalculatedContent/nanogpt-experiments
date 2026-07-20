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
        ]
    )
    NotebookClient(nb, timeout=30, kernel_name="python3").execute(cwd=str(Path.cwd()))
