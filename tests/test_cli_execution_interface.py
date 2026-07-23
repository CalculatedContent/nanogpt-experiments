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
    assert payload["token_budgets"]["parameter_count_convention"] in {"total", "transformer_body"}
    assert payload["level_multiplier_table"]
    first = payload["level_multiplier_table"][0]
    for key in ["requested_tokens", "realized_tokens", "selected_parameter_count", "realized_tokens_per_selected_parameter", "sequence_count", "optimizer_step_count"]:
        assert key in first
    for key in ["total_unique_trainable_parameters", "non_position_parameters", "transformer_body_parameters", "token_embedding_parameters", "position_embedding_parameters", "output_head_parameters", "tied_weight_accounting"]:
        assert key in first["parameter_report"]


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


def test_run_multiseed_dry_run_reports_step_resolution(tmp_path):
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
        "--max-steps",
        "2",
        "--dry-run",
    )
    payload = _json_payload(cp.stdout)
    assert payload["cli_max_steps"] == 2
    assert payload["configured_max_steps"] == 2
    assert payload["resolved_optimizer_steps"] == 2
    assert payload["optimizer_step_limit_source"] == "cli_max_steps"
    assert payload["resolved_train_tokens"] == payload["resolved_optimizer_steps"] * payload["token_budgets"]["tokens_per_step"]
    override = tmp_path / "results" / "cli_overrides_config.yaml"
    assert override.exists()
    assert "max_steps: 2" in override.read_text()


def test_run_canonical_trials_accepts_max_steps_override_in_dry_run(tmp_path):
    cp = _run_cli(
        "run-canonical-trials",
        "--level",
        "0",
        "--data-root",
        str(tmp_path / "data"),
        "--results-root",
        str(tmp_path / "results"),
        "--token-multiplier",
        "20",
        "--max-steps",
        "2",
        "--dry-run",
    )
    payload = _json_payload(cp.stdout)
    assert payload["cli_max_steps"] == 2
    assert payload["resolved_config"]["train"]["max_steps"] == 2
    assert payload["number_of_arms"] == 6


def test_run_multiseed_dry_run_reports_ww_interval_alias(tmp_path):
    cp = _run_cli(
        "run-multiseed",
        "--level", "0",
        "--data-root", str(tmp_path / "data"),
        "--results-root", str(tmp_path / "results"),
        "--token-multiplier", "20",
        "--max-steps", "24",
        "--ww-interval", "8",
        "--dry-run",
    )
    payload = _json_payload(cp.stdout)
    assert payload["effective_wwpgd_interval"] == 8
    assert payload["estimated_projection_event_count"] == 3
    assert payload["resolved_config"]["train"]["wwpgd_interval"] == 8


def test_run_multiseed_conflicting_interval_aliases_fail(tmp_path):
    cp = subprocess.run(
        [
            sys.executable, "-m", "wwgpt.cli", "run-multiseed",
            "--level", "0",
            "--data-root", str(tmp_path / "data"),
            "--results-root", str(tmp_path / "results"),
            "--token-multiplier", "20",
            "--ww-interval", "2",
            "--wwpgd-interval", "4",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )
    assert cp.returncode != 0
    assert "conflicting WW-PGD interval aliases" in cp.stderr
