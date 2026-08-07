# Hyperparameter rationale and source trail

The comparison fixes architecture, data identity, seed set, effective batch, and
training-token budget. Optimizer profiles are allowed to differ because forcing
one numerical learning rate across SGD, AdamW, and Muon would not be a
meaningful control.

## AdamW

The AdamW profile follows the small-GPT defaults in the original nanoGPT
training code:

- peak learning rate `6e-4`;
- cosine decay to `6e-5`;
- betas `(0.9, 0.95)`;
- weight decay `0.1`;
- gradient clipping at `1.0`.

Source: <https://github.com/karpathy/nanoGPT/blob/master/train.py>

The warm-up is expressed as one percent of this longer five-pass run rather
than retaining the old absolute 20-step value.

## Muon

The Muon partition and update follow the official implementation:

- Muon is applied only to hidden 2D weight matrices;
- embeddings, output/head parameters, gains, biases, and other non-hidden
  parameters use auxiliary AdamW;
- hidden learning rate `0.02`;
- momentum `0.95` with Nesterov look-ahead;
- five quintic Newton-Schulz iterations;
- matrix-shape scaling by `sqrt(max(1, rows / columns))`.

Source: <https://github.com/KellerJordan/Muon>

The Newton-Schulz kernel runs in float32 here rather than bfloat16 because this
suite targets Apple MPS and prioritizes a conservative, uniform numerical path.
Hidden and auxiliary learning rates share the same warm-up/decay progress but
keep separate magnitudes.

## SGD with momentum

There is no canonical plain-SGDM nanoGPT recipe comparable to the AdamW
defaults. The selected profile is therefore a conservative, explicit baseline
rather than a claim of a globally optimal value:

- Nesterov momentum `0.90`;
- peak learning rate `0.05`;
- ten-percent warm-up;
- cosine decay to `0.005`;
- weight decay `0.01`;
- gradient clipping at `1.0`.

The longer warm-up is intentional because SGDM lacks AdamW's coordinate-wise
normalization and is more exposed to early gradient-scale transients. Any later
learning-rate sweep should be treated as a separate pilot and should be
completed before looking at confirmatory outcomes.

## Test-by-epoch measurement policy

The optimizer comparison requires test loss and exact next-token accuracy on a
common epoch grid. The grid is fixed before training at nominal epochs
`1, 2, 3, 4, 5`.

At each point the code:

- evaluates the same fixed per-seed test probe;
- writes `epoch_metrics.csv`;
- saves a model-only epoch checkpoint;
- never uses the test value for updates, stopping, learning-rate changes, or
  checkpoint selection.

Validation loss remains the only checkpoint-selection signal. The terminal
test table still reports both the final and validation-selected checkpoints.

## Statistical presentation

Three seeds are enough to expose gross optimizer instability, but they do not
justify normal-theory error bars. Therefore:

- trajectories show mean ± 2 sample standard deviations as an ensemble
  variability envelope;
- integer-epoch train/test comparisons use the same mean ± 2-SD definition;
- final held-out metrics show 95% Student-t confidence intervals with 2 degrees
  of freedom;
- validation selects checkpoints;
- epoch-test measurements are monitoring-only.

## WeightWatcher and ERG

WeightWatcher is invoked as:

```python
watcher.analyze(ERG=True, plot=False, randomize=False, min_evals=20)
```

The raw result table is retained. The suite aggregates the returned `alpha`,
`ERG_gap`, and other spectral columns, but never substitutes a fallback alpha
or fabricates an ERG gap when WeightWatcher does not return one.

Source: <https://github.com/CalculatedContent/WeightWatcher>
