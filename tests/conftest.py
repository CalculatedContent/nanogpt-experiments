from __future__ import annotations

from pathlib import Path


def pytest_collection_modifyitems(config, items):
    import pytest

    unit_files = {
        "test_analysis_helpers.py", "test_analysis_statistics.py", "test_config_profiles.py",
        "test_config_validation_pass1.py", "test_core.py", "test_critical_fixes.py",
        "test_external_wwpgd_pass3.py", "test_integrity_publication_grade.py",
        "test_lr_scheduler_policy.py", "test_memmap_data.py", "test_nanogpt_baseline.py",
        "test_optim_pass2.py", "test_publication_plots.py", "test_scaling_accounting.py",
        "test_schema_v3.py", "test_trial_integrity.py", "test_weightwatcher_diagnostics.py",
    }
    integration_files = {
        "test_canonical_trials.py", "test_cli_execution_interface.py", "test_data_source_reproducibility.py",
        "test_pass5_dataset_e2e.py", "test_pass6_resume_analysis_cli_ci.py",
        "test_schema_v2_analysis.py", "test_strength_scan.py", "test_wwpgd_training_cadence.py",
    }
    accelerator_files = {"test_accelerator_device.py"}
    slow_tests = {
        "tests/test_core.py::test_notebooks_parse",
        "tests/test_strength_scan.py::test_notebooks_parse_strength",
    }

    for item in items:
        name = Path(str(item.fspath)).name
        nodeid = item.nodeid
        if name in accelerator_files:
            item.add_marker(pytest.mark.accelerator)
        elif name in integration_files:
            item.add_marker(pytest.mark.integration)
        elif name in unit_files:
            item.add_marker(pytest.mark.unit)
        if nodeid in slow_tests:
            item.add_marker(pytest.mark.slow(reason="executes/parses the full notebook set; kept out of normal CI"))
