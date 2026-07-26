from __future__ import annotations

import inspect
import re
from pathlib import Path

import wwgpt
from wwgpt.config import WWPGDConfig
from wwgpt.pip_wwpgd_adapter import inspect_pip_wwpgd_api, resolve_pip_wwpgd_provenance
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
    ):
        assert parameter in signature.parameters
    assert ww_pgd.ww_pgd_project is inspect_pip_wwpgd_api()["projector"]


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
    assert resolve_pip_wwpgd_provenance()["wwpgd_installed_version"] == fields["wwpgd_installed_version"]


def test_import_does_not_monkeypatch_installed_projector() -> None:
    import ww_pgd

    before = ww_pgd.ww_pgd_project
    assert wwgpt.__version__
    assert ww_pgd.ww_pgd_project is before
