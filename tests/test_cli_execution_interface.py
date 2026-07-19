import json
import re
import subprocess
import sys


def _run_cli(*args):
    return subprocess.run([sys.executable, "-m", "wwgpt.cli", *args], check=True, text=True, capture_output=True)


def _json_payload(stdout: str):
    match = re.search(r"\{.*\}\s*$", stdout, re.S)
    assert match, stdout
    return json.loads(match.group(0))


def test_prepare_data_profile_dry_run_loads_profile(tmp_path):
    cp = _run_cli(
        "prepare-data",
        "--profile",
        "reproduction_tiny",
        "--level",
        "0",
        "--data-root",
        str(tmp_path / "data"),
        "--token-multiplier",
        "20",
        "--dry-run",
    )
    payload = _json_payload(cp.stdout)
    assert payload["dry_run"] is True
    assert payload["config_path"] == "configs/reproduction_tiny.yaml"
    assert payload["resolved_config"]["data_mode"] == "tiny_shakespeare_char_reproduction"
    assert payload["resolved_config"]["model"]["block_size"] == 64


def test_run_multiseed_dry_run_is_canonical_six_arm(tmp_path):
    cp = _run_cli(
        "run-multiseed",
        "--level",
        "0",
        "--data-root",
        str(tmp_path / "data"),
        "--results-root",
        str(tmp_path / "results"),
        "--token-multiplier",
        "20",
        "--seeds",
        "1,2",
        "--max-steps",
        "7",
        "--dry-run",
    )
    payload = _json_payload(cp.stdout)
    assert payload["number_of_trials"] == 2
    assert payload["number_of_arms"] == 6
    assert payload["arms"] == ["adamw", "adamw_wwpgd", "muon", "muon_wwpgd", "stable_adamw", "stable_adamw_wwpgd"]
    assert payload["seeds"] == [1, 2]
    assert payload["resolved_config"]["train"]["max_steps"] == 7


def test_run_multiseed_rejects_noncanonical_options(tmp_path):
    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "wwgpt.cli",
            "run-multiseed",
            "--level",
            "0",
            "--data-root",
            str(tmp_path / "data"),
            "--results-root",
            str(tmp_path / "results"),
            "--token-multiplier",
            "20",
            "--optimizer",
            "muon",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )
    assert cp.returncode != 0
    assert "canonical-only" in cp.stderr
