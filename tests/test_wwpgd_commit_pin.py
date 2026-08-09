"""Deterministic coverage for optional WWPGD runtime commit verification."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import wwgpt.pip_wwpgd_adapter as adapter
import wwgpt.ww as ww

RESOLVED_COMMIT = "abcdef0123456789abcdef0123456789abcdef01"


def _provenance(*, resolved: str | None = RESOLVED_COMMIT) -> dict[str, object]:
    return {
        "wwpgd_distribution_name": "ww-pgd",
        "wwpgd_installed_version": "test",
        "wwpgd_resolved_commit": resolved,
        "wwpgd_install_mode": "pip-vcs" if resolved else "pypi",
    }


def test_unset_pin_reports_floating_install_without_false_pin_claim() -> None:
    info = adapter.verify_wwpgd_commit_pin(_provenance(), requested_pin="")
    assert info["wwpgd_commit_pin_requested"] is None
    assert info["wwpgd_commit_pin_verified"] is False
    assert info["wwpgd_commit_pin_status"] == "floating"
    assert info["wwpgd_dependency_pinned"] is False


@pytest.mark.parametrize(
    "requested",
    [RESOLVED_COMMIT, RESOLVED_COMMIT[:12], RESOLVED_COMMIT[:20].upper()],
)
def test_full_or_valid_hex_prefix_is_verified(requested: str) -> None:
    info = adapter.verify_wwpgd_commit_pin(
        _provenance(),
        requested_pin=requested,
    )
    assert info["wwpgd_commit_pin_requested"] == requested.lower()
    assert info["wwpgd_commit_pin_verified"] is True
    assert info["wwpgd_commit_pin_status"] == "verified"
    assert info["wwpgd_resolved_commit"] == RESOLVED_COMMIT
    # The install requirement remains floating even after runtime verification.
    assert info["wwpgd_dependency_pinned"] is False


@pytest.mark.parametrize(
    "requested",
    [
        "a" * 11,
        "g" * 12,
        "abc def012345",
        "a" * 65,
    ],
)
def test_malformed_or_dangerously_short_pin_is_rejected(requested: str) -> None:
    with pytest.raises(ValueError, match="hexadecimal Git SHA prefix"):
        adapter.verify_wwpgd_commit_pin(
            _provenance(),
            requested_pin=requested,
        )


def test_mismatch_is_rejected_without_reverse_prefix_acceptance() -> None:
    with pytest.raises(RuntimeError, match="commit pin mismatch"):
        adapter.verify_wwpgd_commit_pin(
            _provenance(),
            requested_pin="0" * 12,
        )

    # A requested value longer than the resolved commit is not accepted merely
    # because the requested text starts with the resolved value.
    with pytest.raises(RuntimeError, match="commit pin mismatch"):
        adapter.verify_wwpgd_commit_pin(
            _provenance(resolved=RESOLVED_COMMIT[:12]),
            requested_pin=RESOLVED_COMMIT,
        )


def test_requested_pin_requires_pep610_vcs_commit() -> None:
    with pytest.raises(RuntimeError, match="no VCS commit"):
        adapter.verify_wwpgd_commit_pin(
            _provenance(resolved=None),
            requested_pin=RESOLVED_COMMIT[:12],
        )


def test_nonhexadecimal_resolved_commit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="not a hexadecimal Git SHA"):
        adapter.verify_wwpgd_commit_pin(
            _provenance(resolved="not-a-git-sha"),
            requested_pin=RESOLVED_COMMIT[:12],
        )


def test_live_provenance_tracks_environment_changes_after_import(monkeypatch) -> None:
    live = adapter._LiveWWPGDProvenance(_provenance())

    monkeypatch.delenv(adapter.WWPGD_COMMIT_PIN_ENV, raising=False)
    first = dict(live)
    assert first["wwpgd_commit_pin_status"] == "floating"

    monkeypatch.setenv(
        adapter.WWPGD_COMMIT_PIN_ENV,
        RESOLVED_COMMIT[:12].upper(),
    )
    second = dict(live)
    assert second["wwpgd_commit_pin_requested"] == RESOLVED_COMMIT[:12]
    assert second["wwpgd_commit_pin_verified"] is True

    monkeypatch.setenv(adapter.WWPGD_COMMIT_PIN_ENV, "0" * 12)
    with pytest.raises(RuntimeError, match="commit pin mismatch"):
        dict(live)


def test_config_and_candidate_boundaries_both_enforce_requested_pin(monkeypatch) -> None:
    class BoundaryReached(RuntimeError):
        pass

    def fail() -> None:
        raise BoundaryReached("shared pin boundary reached")

    monkeypatch.setattr(adapter, "enforce_wwpgd_commit_pin", fail)

    with pytest.raises(BoundaryReached, match="shared pin boundary reached"):
        adapter.construct_pip_wwpgd_config(object())

    with pytest.raises(BoundaryReached, match="shared pin boundary reached"):
        adapter.run_pip_wwpgd_candidate(object(), object())


def test_manifest_uses_fresh_verified_provenance_not_import_time_state(monkeypatch) -> None:
    live = adapter._LiveWWPGDProvenance(_provenance())
    monkeypatch.setattr(ww, "_WWPGD_PROVENANCE", live)
    monkeypatch.setattr(
        ww,
        "construct_pip_wwpgd_config",
        lambda _cfg: (object(), {"resolved": {"q": 1.0}}),
    )
    requested = SimpleNamespace(
        enabled=True,
        extension="wwpgd",
        target_alpha=2.0,
        blend_eta=0.5,
        cayley_eta=0.25,
        min_tail=5,
        use_detx=True,
        verbose=False,
        candidate_device="cpu",
        max_relative_frobenius_change=None,
    )

    monkeypatch.delenv(adapter.WWPGD_COMMIT_PIN_ENV, raising=False)
    floating = ww.external_wwpgd_manifest_fields(True, requested)
    assert floating["wwpgd_commit_pin_status"] == "floating"
    assert floating["wwpgd_commit_pin_verified"] is False

    monkeypatch.setenv(
        adapter.WWPGD_COMMIT_PIN_ENV,
        RESOLVED_COMMIT[:12],
    )
    verified = ww.external_wwpgd_manifest_fields(True, requested)
    assert verified["wwpgd_commit_pin_requested"] == RESOLVED_COMMIT[:12]
    assert verified["wwpgd_commit_pin_verified"] is True
    assert verified["wwpgd_dependency_pinned"] is False

    monkeypatch.setenv(adapter.WWPGD_COMMIT_PIN_ENV, "0" * 12)
    with pytest.raises(RuntimeError, match="commit pin mismatch"):
        ww.external_wwpgd_manifest_fields(True, requested)


def test_supported_wwpgd_runners_delegate_to_verified_shared_boundaries() -> None:
    expected = {
        "src/wwgpt/ww.py": (
            "construct_pip_wwpgd_config",
            "run_pip_wwpgd_candidate",
        ),
        "level_0_wwpgd/src/level0_wwpgd/wwpgd_extension.py": (
            "external_wwpgd_manifest_fields",
            "run_pip_wwpgd_candidate",
        ),
        "experiment_2/src/experiment_2/adaptive_extension.py": (
            "external_wwpgd_manifest_fields",
            "run_pip_wwpgd_candidate",
        ),
    }
    for path, required_names in expected.items():
        text = Path(path).read_text(encoding="utf-8")
        for required_name in required_names:
            assert required_name in text, f"{path} bypasses {required_name}"
