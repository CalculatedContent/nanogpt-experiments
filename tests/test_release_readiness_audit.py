from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import wwgpt.reproducibility as reproducibility


def test_audit_cli_exits_nonzero_for_invalid_experiment(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wwgpt.cli",
            "audit-experiment",
            "--experiment-root",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    summary = tmp_path / "analysis" / "integrity_summary.json"
    assert summary.is_file()
    assert json.loads(summary.read_text())["valid_for_publication"] is False


def test_reproducibility_strict_rejects_invalid_audit(monkeypatch, tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    audit = analysis / "integrity_summary.json"
    audit.write_text(
        json.dumps(
            {
                "valid_for_publication": False,
                "failures": [{"reason": "forced-invalid"}],
            }
        )
    )
    monkeypatch.setattr(reproducibility, "analyze_results", lambda *_args, **_kwargs: analysis)
    monkeypatch.setattr(
        reproducibility,
        "generate_experiment_health",
        lambda *_args, **_kwargs: {"ready_for_analysis": True, "reports": []},
    )
    monkeypatch.setattr(reproducibility, "audit_experiment", lambda *_args, **_kwargs: audit)
    monkeypatch.setattr(reproducibility, "completed_runs", lambda *_args, **_kwargs: [])
    with pytest.raises(RuntimeError, match="invalid experiment"):
        reproducibility.write_reproducibility_report(tmp_path, strict=True)


def test_reproducibility_cli_forwards_strict_and_analysis_plan() -> None:
    source = Path("src/wwgpt/cli.py").read_text()
    assert "strict=args.strict" in source
    assert "analysis_plan=args.analysis_plan" in source


def test_postprocessing_cli_exits_after_flushing_completed_artifacts() -> None:
    source = Path("src/wwgpt/cli.py").read_text()
    assert "def _exit_after_flush(status: int = 0)" in source
    assert 'elif args.cmd=="analyze-results":\n        print(' in source
    assert 'elif args.cmd=="generate-reproducibility-report":' in source
    assert source.count("_exit_after_flush(0)") >= 3
