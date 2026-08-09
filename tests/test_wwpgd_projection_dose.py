"""Regression coverage for exact, resume-safe WW-PGD projection-dose telemetry."""
from __future__ import annotations

import csv
import math

import pytest

import wwgpt.checkpointing as checkpointing
from wwgpt.checkpointing import append_csv_records, csv_commit, reconcile_csv_artifacts


DOSE_DEFINITION = (
    "applied_projection_delta_frobenius_over_preprojection_weight_frobenius"
)


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _projection_row(
    *,
    event: int | float,
    step: int,
    dose: float,
    changed: bool,
    layer: str = "blocks.0.attn.query",
):
    return {
        "layer_name": layer,
        "projection_event": event,
        "actual_step": step,
        "relative_frobenius_change_applied": dose,
        "changed": changed,
    }


def test_projection_csv_reports_exact_realized_dose_and_first_application(tmp_path):
    path = tmp_path / "wwpgd_projection.csv"
    source = _projection_row(event=0, step=10, dose=0.125, changed=True)

    append_csv_records(path, [source])

    row = _read_rows(path)[0]
    assert row["dose_definition"] == DOSE_DEFINITION
    assert float(row["dose_value"]) == pytest.approx(0.125)
    assert float(row["dose_relative_frobenius"]) == pytest.approx(0.125)
    assert row["dose_value"] == row["relative_frobenius_change_applied"]
    assert row["dose_applied"] == "True"
    assert int(row["layer_application_index"]) == 1
    assert row["is_first_applied_projection"] == "True"
    assert int(row["projection_event_index_base"]) == 0
    assert int(row["projection_event_number"]) == 1
    assert row["is_first_scheduled_projection_event"] == "True"
    assert "is_first_projection_event" not in row
    assert "dose_value" not in source  # durable enrichment must not mutate callers


def test_first_scheduled_event_can_precede_first_actual_application(tmp_path):
    path = tmp_path / "wwpgd_projection.csv"
    append_csv_records(
        path,
        [
            _projection_row(event=0, step=10, dose=0.0, changed=False),
            _projection_row(event=1, step=20, dose=0.05, changed=True),
        ],
    )

    rows = _read_rows(path)
    assert rows[0]["is_first_scheduled_projection_event"] == "True"
    assert rows[0]["dose_applied"] == "False"
    assert rows[0]["layer_application_index"] == ""
    assert rows[0]["is_first_applied_projection"] == "False"
    assert float(rows[0]["dose_value"]) == 0.0

    assert rows[1]["is_first_scheduled_projection_event"] == "False"
    assert rows[1]["dose_applied"] == "True"
    assert int(rows[1]["layer_application_index"]) == 1
    assert rows[1]["is_first_applied_projection"] == "True"
    assert int(rows[1]["projection_event_number"]) == 2

    first_applied = next(row for row in rows if row["is_first_applied_projection"] == "True")
    assert int(first_applied["projection_event"]) == 1
    assert int(first_applied["actual_step"]) == 20


def test_layer_application_index_survives_sequential_appends(tmp_path):
    path = tmp_path / "wwpgd_projection.csv"
    append_csv_records(path, [_projection_row(event=0, step=10, dose=0.0, changed=False)])
    append_csv_records(path, [_projection_row(event=1, step=20, dose=0.05, changed=True)])
    append_csv_records(path, [_projection_row(event=2, step=30, dose=0.04, changed=True)])

    rows = _read_rows(path)
    assert len(rows) == 3
    assert rows[0]["layer_application_index"] == ""
    assert int(rows[1]["layer_application_index"]) == 1
    assert rows[1]["is_first_applied_projection"] == "True"
    assert int(rows[2]["layer_application_index"]) == 2
    assert rows[2]["is_first_applied_projection"] == "False"
    assert int(rows[2]["projection_event_number"]) == 3


def test_application_count_index_avoids_full_history_rescan(monkeypatch, tmp_path):
    path = tmp_path / "wwpgd_projection.csv"
    append_csv_records(path, [_projection_row(event=0, step=10, dose=0.05, changed=True)])
    index_path = path.with_name(f".{path.name}.application_counts.json")
    assert index_path.is_file()

    def fail_scan(_path):
        raise AssertionError("valid sidecar index should avoid a full CSV rescan")

    monkeypatch.setattr(checkpointing, "_scan_projection_application_counts", fail_scan)
    append_csv_records(path, [_projection_row(event=1, step=20, dose=0.04, changed=True)])
    assert int(_read_rows(path)[-1]["layer_application_index"]) == 2


def test_invalid_sidecar_metadata_rebuilds_from_csv(tmp_path):
    path = tmp_path / "wwpgd_projection.csv"
    append_csv_records(path, [_projection_row(event=0, step=10, dose=0.05, changed=True)])
    index_path = path.with_name(f".{path.name}.application_counts.json")
    index_path.write_text('{"schema_version": "broken", "counts": {}}')

    append_csv_records(path, [_projection_row(event=1, step=20, dose=0.04, changed=True)])
    assert int(_read_rows(path)[-1]["layer_application_index"]) == 2


def test_reconciled_uncommitted_suffix_does_not_advance_application_count(tmp_path):
    path = tmp_path / "wwpgd_projection.csv"
    append_csv_records(path, [_projection_row(event=0, step=10, dose=0.05, changed=True)])
    committed = csv_commit(path)

    # Simulate rows flushed after the last durable checkpoint, then a restart.
    append_csv_records(path, [_projection_row(event=1, step=20, dose=0.04, changed=True)])
    assert int(_read_rows(path)[-1]["layer_application_index"]) == 2
    reconcile_csv_artifacts(tmp_path, {path.name: committed})

    append_csv_records(path, [_projection_row(event=2, step=30, dose=0.03, changed=True)])
    rows = _read_rows(path)
    assert len(rows) == 2
    assert int(rows[-1]["layer_application_index"]) == 2
    assert rows[-1]["is_first_applied_projection"] == "False"


def test_first_application_is_tracked_independently_per_layer(tmp_path):
    path = tmp_path / "wwpgd_projection.csv"
    append_csv_records(
        path,
        [
            _projection_row(
                event=0,
                step=10,
                dose=0.05,
                changed=True,
                layer="blocks.0.attn.query",
            ),
            _projection_row(
                event=0,
                step=10,
                dose=0.03,
                changed=True,
                layer="blocks.0.attn.key",
            ),
        ],
    )
    append_csv_records(
        path,
        [
            _projection_row(
                event=1,
                step=20,
                dose=0.02,
                changed=True,
                layer="blocks.0.attn.query",
            )
        ],
    )

    rows = _read_rows(path)
    assert [int(rows[index]["layer_application_index"]) for index in (0, 1, 2)] == [1, 1, 2]
    assert [rows[index]["is_first_applied_projection"] for index in (0, 1, 2)] == [
        "True",
        "True",
        "False",
    ]


def test_non_projection_csv_rows_are_unchanged(tmp_path):
    path = tmp_path / "metrics.csv"
    append_csv_records(path, [{"step": 1, "loss": 2.5}])
    assert _read_rows(path) == [{"step": "1", "loss": "2.5"}]


def test_old_projection_schema_is_rejected_before_append(tmp_path):
    path = tmp_path / "wwpgd_projection.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer_name",
                "projection_event",
                "actual_step",
                "relative_frobenius_change_applied",
                "changed",
            ],
        )
        writer.writeheader()
        writer.writerow(_projection_row(event=0, step=10, dose=0.05, changed=True))

    before = path.read_bytes()
    with pytest.raises(ValueError, match="older telemetry schema"):
        append_csv_records(path, [_projection_row(event=1, step=20, dose=0.04, changed=True)])
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "row, message",
    [
        ({"relative_frobenius_change_applied": 0.1, "changed": True}, "projection_event"),
        (_projection_row(event=-1, step=1, dose=0.1, changed=True), "non-negative"),
        (_projection_row(event=0.5, step=1, dose=0.1, changed=True), "must be an integer"),
        (_projection_row(event=0, step=1, dose=-0.1, changed=True), "finite and non-negative"),
        (_projection_row(event=0, step=1, dose=math.nan, changed=True), "finite and non-negative"),
        (_projection_row(event=0, step=1, dose=0.1, changed=True, layer=""), "layer_name"),
    ],
)
def test_invalid_projection_telemetry_fails_before_writing(tmp_path, row, message):
    path = tmp_path / "wwpgd_projection.csv"
    with pytest.raises(ValueError, match=message):
        append_csv_records(path, [row])
    assert not path.exists()
