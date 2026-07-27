import csv

import pytest

from wwgpt.checkpointing import append_csv_records


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
    assert _rows(relaxation) == [
        {
            "optimizer_step": "76",
            "layer_name": "blocks.0.attn.value",
            "endpoint_measurement_step": "75",
            "action_type": "fast_endpoint_relaxation",
        },
        {
            "optimizer_step": "100",
            "layer_name": "blocks.0.attn.value",
            "endpoint_measurement_step": "75",
            "action_type": "fast_endpoint_relaxation",
        },
    ]


def test_unrelated_csv_schema_mismatches_remain_errors(tmp_path):
    path = tmp_path / "metrics.csv"
    append_csv_records(path, [{"step": 1, "loss": 2.0}])

    with pytest.raises(ValueError, match="CSV schema changed"):
        append_csv_records(path, [{"step": 2, "validation_loss": 1.5}])
