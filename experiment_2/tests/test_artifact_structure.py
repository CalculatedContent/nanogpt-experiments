from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO / "experiment_2"


def test_notebooks_are_valid_v4_json() -> None:
    expected = {
        "01_protocol_audit.ipynb",
        "02_compare_paired.ipynb",
        "03_layer_controller_diagnostics.ipynb",
        "04_cross_scale_summary.ipynb",
    }
    found = {path.name for path in (EXPERIMENT / "notebooks").glob("*.ipynb")}
    assert found == expected
    for path in sorted((EXPERIMENT / "notebooks").glob("*.ipynb")):
        payload = json.loads(path.read_text())
        assert payload["nbformat"] == 4
        assert payload["cells"]
        assert all(
            cell["cell_type"] in {"markdown", "code", "raw"}
            for cell in payload["cells"]
        )


def test_shell_entrypoints_are_isolated_and_source_safe() -> None:
    expected = {"prepare_data.sh", "run_one.sh", "run_pair.sh"}
    found = {path.name for path in (EXPERIMENT / "scripts").glob("*.sh")}
    assert found == expected
    for path in sorted((EXPERIMENT / "scripts").glob("*.sh")):
        source = path.read_text()
        assert source.startswith("#!/usr/bin/env bash")
        assert 'BASH_SOURCE[0]' in source
        assert "do not source" in source
        assert "set -euo pipefail" in source


def test_run_one_avoids_empty_optional_array_expansion() -> None:
    source = (EXPERIMENT / "scripts/run_one.sh").read_text()
    # macOS Bash 3.2 can treat an initialized-but-empty array expansion as
    # unbound under `set -u`. Keep the two invocations explicit.
    assert "BASELINE_ARGS" not in source
    assert '--baseline-run "$BASELINE_RUN"' in source


def test_run_one_dispatches_baseline_and_adaptive_with_nounset(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    pair_root = tmp_path / "pair"
    fake_bin = tmp_path / "bin"
    data_root.mkdir()
    fake_bin.mkdir()
    (data_root / "meta.json").write_text("{}")

    args_log = tmp_path / "python-args.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$EXPERIMENT2_FAKE_ARGS_LOG\"\n"
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["NANOGPT_EXPERIMENT2_DATA_ROOT"] = str(data_root)
    env["NANOGPT_EXPERIMENT2_DEVICE"] = "cpu"
    env["EXPERIMENT2_FAKE_ARGS_LOG"] = str(args_log)

    runner = EXPERIMENT / "scripts/run_one.sh"
    subprocess.run(
        ["/bin/bash", str(runner), "level0", "adamw", "1337", str(pair_root)],
        cwd=REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    baseline_args = args_log.read_text().splitlines()
    assert "--arm" in baseline_args
    assert "adamw" in baseline_args
    assert "--baseline-run" not in baseline_args

    completion = pair_root / "baseline/results/adamw_seed_1337/run_complete.json"
    completion.parent.mkdir(parents=True)
    completion.write_text('{"completed": true}')

    subprocess.run(
        [
            "/bin/bash",
            str(runner),
            "level0",
            "adaptive_wwpgd",
            "1337",
            str(pair_root),
        ],
        cwd=REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    adaptive_args = args_log.read_text().splitlines()
    assert "--arm" in adaptive_args
    assert "adaptive_wwpgd" in adaptive_args
    assert "--baseline-run" in adaptive_args
    assert str(completion.parent) in adaptive_args


def test_experiment_tree_does_not_shadow_existing_level_directories() -> None:
    assert EXPERIMENT.is_dir()
    assert not (EXPERIMENT / "level_0_baseline").exists()
    assert not (EXPERIMENT / "level_0_wwpgd").exists()
    assert not (EXPERIMENT / "level_1_baseline").exists()
    assert not (EXPERIMENT / "level_1_wwpgd").exists()
