# WW-PGD projection dose telemetry

The main scientific runner’s `wwpgd_projection.csv` rows report the realized intervention size and distinguish scheduled events from actual per-layer applications.

## Scope

This schema is applied by the shared durable artifact writer used by `run_scientific_single`. It covers the scientific Level 0–2 WW-PGD, delayed-onset, adaptive, and norm-matched-sham arms that write the root experiment artifact `wwpgd_projection.csv`.

It does not silently redefine the separate invalid-for-science `run_single` smoke artifact or the isolated `level_0_wwpgd` package’s independent projection schema. Those paths have different lifecycle contracts and require separate schema changes if they are ever promoted to scientific evidence.

## Realized dose

For each projected matrix, let `W_before` be the live matrix immediately before the WW-PGD projection and `W_after` the matrix after the applied, trust-region-limited projection. The reported dose is

```text
dose_value = ||W_after - W_before||_F / max(||W_before||_F, 1e-12)
```

The row fields are:

- `dose_definition = applied_projection_delta_frobenius_over_preprojection_weight_frobenius`
- `dose_value`: the realized relative Frobenius change
- `dose_relative_frobenius`: compatibility alias of `dose_value`
- `dose_applied`: true only when the live matrix actually changed and the realized dose is positive

This is an observed dose. It is not `blend_eta`, requested hardness, or a configured trust-region cap.

## Scheduled event indexing

Within the main scientific runner, `projection_event` is zero-based. Rows also include:

- `projection_event_index_base = 0`
- `projection_event_number = projection_event + 1`
- `is_first_scheduled_projection_event = (projection_event == 0)`

The scheduled-event flag is intentionally not called `is_first_projection_event`. Warm-up, delayed onset, adaptive gates, zero hardness, unchanged candidates, and trust-region behavior can make the first scheduled event different from the first actual application.

## First actual application

Actual applications are counted independently for each `layer_name` from the durable CSV prefix:

- `layer_application_index`: one-based application count for an applied row; empty when no dose was applied
- `is_first_applied_projection`: true exactly when `layer_application_index == 1`

The count is reconstructed from the durable projection-row history before every append. Checkpoint recovery first reconciles the CSV to its committed prefix, so the count remains correct after resume or process restart without adding process-local counters to the optimizer, controller, model, or checkpoint state.

## Compatibility

These fields extend the main scientific projection CSV schema but do not change model weights, optimizer state, projection mathematics, spectral targets, or defaults. An incomplete scientific run created with an older projection CSV header is rejected before append; start a fresh run under the new schema instead.
