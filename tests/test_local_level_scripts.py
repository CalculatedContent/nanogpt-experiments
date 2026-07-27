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
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    subprocess.run(
        [
            "bash",
            "scripts/run_one_pair.sh",
            "1",
            str(tmp_path / "data"),
            str(tmp_path / "results"),
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
    assert any(row.startswith("check-health ") for row in rows)
    assert any(row.startswith("analyze-results ") for row in rows)
    assert any(row.startswith("audit-experiment ") for row in rows)
    assert any(row.startswith("generate-reproducibility-report ") for row in rows)


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
