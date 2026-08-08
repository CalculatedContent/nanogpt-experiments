"""Optional WWPGD_COMMIT_PIN env — Class C logging/repro only; default floating."""
from __future__ import annotations

import pytest

from wwgpt.pip_wwpgd_adapter import (
    WWPGD_COMMIT_PIN_ENV,
    assert_wwpgd_commit_pin,
)


def test_commit_pin_noop_when_env_unset(monkeypatch):
    monkeypatch.delenv(WWPGD_COMMIT_PIN_ENV, raising=False)
    info = assert_wwpgd_commit_pin()
    assert info.get("wwpgd_commit_pin_requested") in (None, "")
    assert info.get("wwpgd_dependency_pinned") is False


def test_commit_pin_accepts_matching_prefix(monkeypatch):
    monkeypatch.delenv(WWPGD_COMMIT_PIN_ENV, raising=False)
    base = assert_wwpgd_commit_pin()
    resolved = str(base.get("wwpgd_resolved_commit") or "")
    if not resolved:
        pytest.skip("installed ww_pgd has no VCS commit to pin against")
    monkeypatch.setenv(WWPGD_COMMIT_PIN_ENV, resolved[:12])
    info = assert_wwpgd_commit_pin()
    assert info["wwpgd_dependency_pinned"] is True
    assert str(info["wwpgd_commit_pin_requested"]).startswith(resolved[:12])


def test_commit_pin_rejects_mismatch(monkeypatch):
    monkeypatch.setenv(WWPGD_COMMIT_PIN_ENV, "0" * 40)
    with pytest.raises(RuntimeError, match="commit pin mismatch|no VCS commit"):
        assert_wwpgd_commit_pin()
