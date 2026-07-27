import csv

import pytest

from wwgpt.checkpointing import append_csv_records
from wwgpt.train import _read_complete_csv_rows


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_cached_fast_rows_are_not_duplicated_into_controller_csv(tmp_path):
    controller = tmp_path / "wwpgd_controller.csv"
    relaxation = tmp_path / "wwpgd_endpoint_relaxation.csv"

    slow_measurement_75 = {
        "optimizer_step": 75,
        "layer_name": "blocks.0.attn.value",
        "action_type": "slow_measurement",
    }
    fast_relaxation_76 = {
        "optimizer_step": 76,
        "layer_name": "blocks.0.attn.value",
        "endpoint_measurement_step": 75,
        "action_type": "fast_endpoint_relaxation",
    }
    slow_measurement_100 = {
        "optimizer_step": 100,
        "layer_name": "blocks.0.attn.value",
        "action_type": "slow_measurement",
    }
    fast_relaxation_100 = {
        "optimizer_step": 100,
        "layer_name": "blocks.0.attn.value",
        "endpoint_measurement_step": 75,
        "action_type": "fast_endpoint_relaxation",
    }

    append_csv_records(controller, [slow_measurement_75])
    append_csv_records(controller, [fast_relaxation_76])
    # A custom schedule may put fast and slow records on the same flush boundary.
    # The fast row must still be routed only to its dedicated artifact.
    append_csv_records(controller, [fast_relaxation_100, slow_measurement_100])
    append_csv_records(relaxation, [fast_relaxation_76, fast_relaxation_100])

    assert _rows(controller) == [
        {
            "optimizer_step": "75",
            "layer_name": "blocks.0.attn.value",
            "action_type": "slow_measurement",
        },
        {
            "optimizer_step": "100",
            "layer_name": "blocks.0.attn.value",
            "action_type": "slow_measurement",
        },
    ]
    relaxation_rows = _rows(relaxation)
    assert [
        (
            row["optimizer_step"],
            row["layer_name"],
            row["endpoint_measurement_step"],
            row["action_type"],
        )
        for row in relaxation_rows
    ] == [
        ("76", "blocks.0.attn.value", "75", "fast_endpoint_relaxation"),
        ("100", "blocks.0.attn.value", "75", "fast_endpoint_relaxation"),
    ]


def test_cached_terminal_rows_are_completion_summary_safe(tmp_path):
    relaxation = tmp_path / "wwpgd_endpoint_relaxation.csv"
    measurement = tmp_path / "wwpgd_endpoint_measurements.csv"

    changed_row = {
        "optimizer_step": 76,
        "layer_name": "blocks.0.attn.value",
        "endpoint_measurement_step": 75,
        "controller_gain_requested": 0.02,
        "controller_gain_applied": 0.01,
        "requested_relative_frobenius_change": 0.002,
        "applied_relative_frobenius_change": 0.001,
        "changed": True,
        "converged": False,
        "invalidated": False,
        "invalidation_reason": "",
        "action_type": "fast_endpoint_relaxation",
    }
    terminal_row = {
        "optimizer_step": 77,
        "layer_name": "blocks.0.attn.value",
        "endpoint_measurement_step": 75,
        "changed": False,
        "converged": True,
        "invalidated": False,
        "invalidation_reason": "endpoint_converged",
        "action_type": "fast_endpoint_relaxation",
    }

    append_csv_records(relaxation, [changed_row])
    # Terminal rows intentionally omit movement fields. They are a valid subset of
    # the established relaxation schema and must remain parseable at completion.
    append_csv_records(relaxation, [terminal_row])
    relaxation_rows = _read_complete_csv_rows(relaxation)

    # These are the exact access patterns used by the completion summary.
    changed = [row for row in relaxation_rows if row.get("changed")]
    requested_gains = [
        float(row["controller_gain_requested"])
        for row in relaxation_rows
        if row.get("controller_gain_requested") is not None
    ]
    applied_gains = [
        float(row["controller_gain_applied"])
        for row in relaxation_rows
        if row.get("controller_gain_applied") is not None
    ]
    applied_changes = [
        float(row["applied_relative_frobenius_change"])
        for row in changed
        if row.get("applied_relative_frobenius_change") is not None
    ]

    assert len(changed) == 1
    assert requested_gains == [0.02]
    assert applied_gains == [0.01]
    assert applied_changes == [0.001]
    assert relaxation_rows[1]["controller_gain_requested"] is None
    assert relaxation_rows[1]["changed"] is False
    assert relaxation_rows[1]["converged"] is True

    append_csv_records(
        measurement,
        [
            {
                "optimizer_step": 75,
                "layer_name": "blocks.0.attn.value",
                "alpha_side": "above_target",
                "cache_activated": True,
                "action_type": "slow_measurement",
            },
            {
                "optimizer_step": 75,
                "layer_name": "blocks.0.attn.query",
                "alpha_side": "below_target",
                "cache_activated": False,
                "action_type": "slow_measurement",
            },
        ],
    )
    measurement_rows = _read_complete_csv_rows(measurement)
    activations = [row for row in measurement_rows if row.get("cache_activated")]
    above_activations = sum(
        row.get("alpha_side") == "above_target" and row.get("cache_activated")
        for row in measurement_rows
    )
    below_activations = sum(
        row.get("alpha_side") == "below_target" and row.get("cache_activated")
        for row in measurement_rows
    )

    assert len(activations) == 1
    assert activations[0]["layer_name"] == "blocks.0.attn.value"
    assert above_activations == 1
    assert below_activations == 0
    assert measurement_rows[0]["cache_activated"] is True
    assert measurement_rows[1]["cache_activated"] is False


def test_non_relaxation_csv_schemas_remain_strict(tmp_path):
    path = tmp_path / "metrics.csv"
    append_csv_records(path, [{"step": 1, "loss": 2.0, "note": "first"}])

    with pytest.raises(ValueError, match="CSV schema changed"):
        append_csv_records(path, [{"step": 2, "loss": 1.5}])

    with pytest.raises(ValueError, match="CSV schema changed"):
        append_csv_records(
            path,
            [{"step": 2, "loss": 1.5, "note": "second", "unexpected": 1}],
        )
