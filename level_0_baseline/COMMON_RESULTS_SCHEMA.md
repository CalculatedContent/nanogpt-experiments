# Level Zero baseline reference-store schema

The baseline notebooks publish their core results to a common, optimizer-neutral
directory so later experiments can compare against exactly the same baseline
artifacts without reopening individual run folders.

## Default location

```text
/tmp/nanogpt-level0-baselines/baseline_reference
```

Override it with:

```bash
export NANOGPT_LEVEL0_BASELINE_STORE=/path/to/baseline_reference
```

The store is generated from completed runs only. Each optimizer notebook calls
`export_optimizer_core_results(...)` after its three seeds are available. The
comparison notebook reads only this store.

## Directory structure

```text
baseline_reference/
  store_manifest.json
  all_runs/
    trajectory_metrics.csv
    epoch_checkpoint_metrics.csv
    spectral_metrics.csv
    terminal_test_metrics.csv
  summaries/
    trajectory_bollinger_summary.csv
    epoch_bollinger_summary.csv
    spectral_bollinger_summary.csv
    terminal_test_student_t_summary.csv
  per_optimizer/
    sgd_momentum/
      manifest.json
      trajectory_runs.csv
      epoch_runs.csv
      spectral_runs.csv
      terminal_test_runs.csv
      trajectory_bollinger_summary.csv
      epoch_bollinger_summary.csv
      spectral_bollinger_summary.csv
      terminal_test_student_t_summary.csv
    adamw/
    muon/
```

## Raw tables

### `trajectory_metrics.csv`

One row per seed and normal evaluation step. It contains train/validation loss,
perplexity, exact next-token accuracy, optimization diagnostics, throughput, and
the optimizer identity. Test fields are populated only at preregistered
integer-epoch monitoring points.

### `epoch_checkpoint_metrics.csv`

One row per optimizer, seed, and nominal epoch. The default grid is epochs
`1, 2, 3, 4, 5`. Every row contains:

- train, validation, and test cross-entropy;
- train, validation, and test perplexity;
- train, validation, and test exact next-token top-1 accuracy;
- validation and test generalization gaps;
- gradient, weight, update, and throughput diagnostics;
- the model-only epoch-checkpoint path;
- `test_monitoring_only=1`.

The test probe is fixed per seed, is independent of training minibatch sampling,
and is never used for optimizer updates or checkpoint selection.

### `spectral_metrics.csv`

One row per seed and WeightWatcher measurement. It includes the raw aggregate
columns returned by the run, including alpha, `ERG_gap`, fit distance `D`,
stable rank, MP soft rank, entropy, log/spectral norms, and spike counts when
available. Missing fits remain missing.

### `terminal_test_metrics.csv`

One row per optimizer, seed, and checkpoint policy (`final` or
`validation_selected`).

## Summary tables

The three Bollinger summary files are long-form tables with:

```text
optimizer, optimizer_label, x, metric, n, mean, sd, lower, upper, sigma
```

where:

```text
lower = mean - 2 * sample_sd
upper = mean + 2 * sample_sd
```

These are across-seed variability envelopes, not rolling time-series
Bollinger bands and not confidence intervals.

`terminal_test_student_t_summary.csv` contains the mean, sample standard
deviation, standard error, and two-sided 95% Student-t interval across the three
seeds.

## Rebuilding without notebooks

After all runs complete:

```bash
level0-export-baselines \
  --results-root /tmp/nanogpt-level0-baselines/results \
  --store-root /tmp/nanogpt-level0-baselines/baseline_reference \
  --optimizers sgd_momentum,adamw,muon \
  --seeds 1337,2027,4099
```

The exporter rewrites tables atomically and records source run directories,
seeds, protocol fingerprints, model settings, token budgets, and data
manifests.
