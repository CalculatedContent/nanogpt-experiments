from __future__ import annotations

import inspect
import re
from pathlib import Path

import torch

import wwgpt
from wwgpt._wwpgd_compat import _compatibility_diagnostic
from wwgpt.config import WWPGDConfig
from wwgpt.ww import external_wwpgd_manifest_fields


def test_wwpgd_dependency_is_installed_by_pip_without_revision_pin() -> None:
    text = Path("pyproject.toml").read_text()
    assert "ww-pgd @ git+https://github.com/CalculatedContent/WW_PGD.git" in text
    assert not re.search(r"WW_PGD\.git@", text)


def test_installed_public_api_is_compatible_with_candidate_builder() -> None:
    import ww_pgd

    signature = inspect.signature(ww_pgd.ww_pgd_project)
    for parameter in (
        "epoch",
        "num_epochs",
        "global_step",
        "ww_logs",
        "layer_selector",
        "diagnostic_logs",
    ):
        assert parameter in signature.parameters


def test_manifest_records_runtime_provenance_not_a_pin() -> None:
    fields = external_wwpgd_manifest_fields(
        True,
        WWPGDConfig(enabled=True, extension="wwpgd"),
    )
    assert fields["wwpgd_dependency_pinned"] is False
    assert fields["wwpgd_installed_version"]
    assert fields["wwpgd_source_repository"] == "CalculatedContent/WW_PGD"
    resolved = fields.get("wwpgd_resolved_commit") or ""
    assert fields["wwpgd_commit"] == resolved or fields["wwpgd_commit"].startswith("version:")
    assert wwgpt.WWPGD_PROVENANCE["wwpgd_installed_version"] == fields["wwpgd_installed_version"]


def test_compatibility_rows_are_explicit_about_unsupported_internal_fields() -> None:
    module = torch.nn.Linear(4, 3, bias=False)
    cfg = type(
        "Config",
        (),
        {"min_tail": 5, "use_detx": True, "q": 1.0, "cayley_eta": 0.25, "blend_eta": 0.5},
    )()
    row = _compatibility_diagnostic(
        layer_name="blocks.0.attn.query",
        row={"alpha": 2.3, "D": 0.04, "xmin": 0.1, "detX_num": 3, "num_evals": 3},
        module=module,
        cfg=cfg,
        epoch=2,
        global_step=25,
    )
    assert row["diagnostics_schema_version"] == 1
    assert row["native_internal_diagnostics"] is False
    assert row["status"] == "unsupported_internal_fields"
    assert row["valid_diagnostic"] is False
    assert row["alpha"] == 2.3
    assert row["k_star"] is None
    assert row["trace_log_retraction_residual"] is None
    assert "k_star" in row["unsupported_internal_fields"]
