from __future__ import annotations

import shlex
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ISOLATED_SCRIPTS = [
    REPO_ROOT / "level_0_baseline/scripts/run_one.sh",
    REPO_ROOT / "level_0_baseline/scripts/run_multiseed.sh",
    REPO_ROOT / "level_0_wwpgd/scripts/prepare_data.sh",
    REPO_ROOT / "level_0_wwpgd/scripts/run_one.sh",
    REPO_ROOT / "level_0_wwpgd/scripts/run_multiseed.sh",
    REPO_ROOT / "level_0_wwpgd/scripts/run_smoke.sh",
    REPO_ROOT / "scripts/run_isolated_level0_pair.sh",
]


def test_isolated_shell_scripts_are_executable_and_parse() -> None:
    for script in ISOLATED_SCRIPTS:
        assert script.is_file(), script
        assert script.stat().st_mode & stat.S_IXUSR, (
            f"{script.relative_to(REPO_ROOT)} is not executable in git"
        )
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_isolated_shell_scripts_refuse_sourcing_without_mutating_parent() -> None:
    for script in ISOLATED_SCRIPTS:
        quoted = shlex.quote(str(script))
        command = f"""
set +e
set +u
set +o pipefail
source {quoted}
status=$?
[[ "$status" -eq 2 ]] || exit 20
[[ "$-" != *e* ]] || exit 21
[[ "$-" != *u* ]] || exit 22
pipefail_state="$(set -o | awk '$1 == "pipefail" {{ print $2 }}')"
[[ "$pipefail_state" == "off" ]] || exit 23
printf 'parent-shell-alive\n'
"""
        result = subprocess.run(
            ["bash", "-c", command],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"source guard failed for {script.relative_to(REPO_ROOT)}:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "parent-shell-alive" in result.stdout


def test_multiseed_runners_do_not_depend_on_nested_execute_bits() -> None:
    baseline = (REPO_ROOT / "level_0_baseline/scripts/run_multiseed.sh").read_text()
    wwpgd = (REPO_ROOT / "level_0_wwpgd/scripts/run_multiseed.sh").read_text()
    assert 'bash "$SCRIPT_DIR/run_one.sh"' in baseline
    assert 'bash "$SCRIPT_DIR/run_one.sh"' in wwpgd


def test_isolated_readmes_use_child_bash_processes() -> None:
    baseline = (REPO_ROOT / "level_0_baseline/README.md").read_text()
    wwpgd = (REPO_ROOT / "level_0_wwpgd/README.md").read_text()
    for text in (baseline, wwpgd):
        assert "do not source" in text.lower()
        assert "set -euo pipefail" not in text
    assert "bash scripts/run_isolated_level0_pair.sh baseline" in baseline
    assert "bash scripts/run_isolated_level0_pair.sh wwpgd" in wwpgd
