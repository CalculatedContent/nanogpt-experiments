from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _fake_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def test_run_one_pair_uses_level_specific_scientific_workflow(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    _fake_executable(
        bin_dir / "wwgpt",
        f'printf "%s\\n" "$*" >> "{calls}"\n',
    )
    results = tmp_path / "results"
    (results / "experiments" / "level_01" / "multiplier_20").mkdir(
        parents=True
    )
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    subprocess.run(
        [
            "bash",
            "scripts/run_one_pair.sh",
            "1",
            str(tmp_path / "data"),
            str(results),
            "20",
            "cpu",
            "1337",
        ],
        check=True,
        env=env,
    )

    rows = calls.read_text().splitlines()
    assert rows[0].startswith("prepare-data ")
    assert "--config configs/level1_adaptive_alpha.yaml" in rows[0]
    run = next(row for row in rows if row.startswith("run-multiseed "))
    assert "--level 1" in run
    assert "--optimizer adamw" in run
    assert "--extensions none,wwpgd" in run
    assert "--device cpu" in run
    level_layout = str(results / "experiments" / "level_01" / "multiplier_20")
    assert any(
        row.startswith("check-health ") and level_layout in row for row in rows
    )
    assert any(
        row.startswith("analyze-results ") and level_layout in row for row in rows
    )
    assert any(
        row.startswith("audit-experiment ") and level_layout in row for row in rows
    )
    report = next(
        row for row in rows if row.startswith("generate-reproducibility-report ")
    )
    assert level_layout in report
    assert "--analysis-plan configs/analysis_plan_exploratory.yaml" in report
    assert "--strict" in report


def test_level0_2_runner_defaults_to_bounded_pilot(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    _fake_executable(
        bin_dir / "wwgpt",
        f'printf "wwgpt %s\\n" "$*" >> "{calls}"\n',
    )
    _fake_executable(
        bin_dir / "python",
        f'printf "python %s\\n" "$*" >> "{calls}"\n',
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WWGPT_LEVELS": "0,1,2",
        "WWGPT_RUN_NOTEBOOKS": "0",
    }

    subprocess.run(
        [
            "bash",
            "scripts/run_level0_2_experiment.sh",
            str(tmp_path / "data"),
            str(tmp_path / "results"),
            "20",
            "cpu",
        ],
        check=True,
        env=env,
    )

    rows = calls.read_text().splitlines()
    prepares = [row for row in rows if row.startswith("wwgpt prepare-data ")]
    actual_runs = [
        row
        for row in rows
        if row.startswith("wwgpt run-multiseed ") and "--dry-run" not in row
    ]
    assert len(prepares) == 3
    assert len(actual_runs) == 3
    for level in (0, 1, 2):
        assert any(
            f"--config configs/level{level}_adaptive_alpha.yaml" in row
            for row in prepares
        )
        run = next(row for row in actual_runs if f"--level {level}" in row)
        assert "--seeds 1337" in run
        assert "--extensions none,wwpgd" in run


def test_level0_2_runner_postprocesses_each_level_and_combined_root(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    _fake_executable(
        bin_dir / "wwgpt",
        f'printf "wwgpt %s\n" "$*" >> "{calls}"\n',
    )
    _fake_executable(
        bin_dir / "python",
        f'printf "python %s\n" "$*" >> "{calls}"\n',
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WWGPT_LEVELS": "0,1,2",
        "WWGPT_RUN_NOTEBOOKS": "0",
    }
    results = tmp_path / "results"
    subprocess.run(
        [
            "bash",
            "scripts/run_level0_2_experiment.sh",
            str(tmp_path / "data"),
            str(results),
            "20",
            "cpu",
        ],
        check=True,
        env=env,
    )
    rows = calls.read_text().splitlines()
    for command in (
        "wwgpt check-health ",
        "wwgpt analyze-results ",
        "wwgpt audit-experiment ",
        "wwgpt generate-reproducibility-report ",
    ):
        selected = [row for row in rows if row.startswith(command)]
        assert len(selected) == 4
        for level in (0, 1, 2):
            layout = f"level_{level:02d}/multiplier_20"
            assert any(layout in row for row in selected)
        combined_root = str(results / "adaptive" / "adamw" / "flat" / "fixed")
        assert any(
            combined_root in row and "/experiments/level_" not in row
            for row in selected
        )
    reports = [
        row
        for row in rows
        if row.startswith("wwgpt generate-reproducibility-report ")
    ]
    assert all("--strict" in row for row in reports)
    assert any(row.startswith("python -m wwgpt.cross_level_analysis ") for row in rows)


def test_macbook_preflight_uses_active_python_and_text_dry_runs() -> None:
    source = Path("scripts/macbook_preflight.sh").read_text()
    assert 'PYTHON_BIN="${PYTHON_BIN:-python3}"' in source
    assert "macbook-preflight-level${level}-${mode}.txt" in source
    assert "macbook-preflight-level${level}-${mode}.json" not in source


def test_bounded_level012_runner_uses_one_clean_postprocessing_process() -> None:
    source = Path("scripts/run_bounded_level012_acceptance.sh").read_text()
    assert "set -euo pipefail" in source
    assert 'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"' in source
    assert 'export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"' in source
    assert 'python - "$RESULTS_ROOT" "$PLAN"' in source
    assert "from wwgpt.analysis import analyze_results" in source
    assert "from wwgpt.reproducibility import write_reproducibility_report" in source
    assert "repeated_report = write_reproducibility_report" in source
    assert "LEVEL_0_1_2_RELEASE_ACCEPTANCE_PASS" in source
    assert "wwgpt generate-reproducibility-report" not in source
