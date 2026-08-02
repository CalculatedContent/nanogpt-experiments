# Experiment 2: reference-guided layer-wise adaptive WWPGD

Experiment 2 is a new, isolated research tree. It does not alter the existing
Level 0 or Level 1 experiments.

It runs the same question at two scales:

| scale | layers | heads | width | context | steps |
|---|---:|---:|---:|---:|---:|
| `level0` | 4 | 4 | 128 | 256 | 2,000 |
| `level1` | 8 | 8 | 512 | 512 | 10,000 |

Each scale has two matched arms:

1. `adamw`: the reference trajectory;
2. `adaptive_wwpgd`: the same AdamW trajectory plus a controller-selected
   WWPGD intervention.

## Scientific objective

The target alpha remains **2.0 for every projected Transformer matrix**, but the
controller no longer projects every matrix on every optimizer step.

A layer becomes eligible only when its measured alpha lies outside a hysteresis
band around the target. The controller then selects a small rotating cohort of
the largest-error layers. For every candidate correction it records:

- alpha before and candidate alpha after;
- whether the candidate actually moves alpha toward 2;
- AdamW update norm;
- WWPGD correction norm;
- cosine alignment between the AdamW update and WWPGD correction;
- correction-to-AdamW update ratio;
- immediate loss change on a fixed, independent training-control probe when
  that layer's candidate correction is applied in isolation;
- applied trust-region scale;
- layer credit, cooldown, and projection status.

A candidate is rejected when it does not improve alpha, strongly opposes AdamW,
exceeds the update-ratio guardrail after scaling, worsens the fixed control-probe
loss beyond the configured tolerance, or becomes non-finite. Candidate probe
evaluations are transactional: the live weight is restored before the next layer
is tested, and accepted corrections are committed only after all selected layers
have been evaluated.

## Reference-guided safety controller

The paired runner completes the matched AdamW baseline first. The adaptive arm
then reads that baseline's metrics at identical evaluation steps.

Over each evaluation window it compares adaptive validation-loss progress with
the matched baseline progress. The resulting window advantage is assigned to
the active layer cohort as an **approximate shared credit**, not as a causal
per-layer attribution. Repeatedly harmful cohorts enter cooldown. If adaptive
validation loss falls behind the baseline by the configured margin for multiple
windows, all WWPGD interventions pause temporarily.

This makes the controller deliberately reference-guided. It is an experimental
feedback controller, not a claim that WWPGD can infer a counterfactual baseline
without running one.

## Data

Both scales reuse the immutable GPT-2-BPE FineWeb-Edu corpus:

```text
/tmp/nanogpt-level0-bpe/data
```

Prepare it only if missing:

```bash
conda activate ww_prod310
cd ~/Desktop/work/nanoGPT/nanogpt-experiments
bash experiment_2/scripts/prepare_data.sh
```

## Install

```bash
conda activate ww_prod310
cd ~/Desktop/work/nanoGPT/nanogpt-experiments
python -m pip install -e .
python -m pip install -e 'experiment_2[data,analysis,test]'
python -m pip check
```

## Run Level 0

Start with one matched seed:

```bash
caffeinate -dimsu env \
  NANOGPT_EXPERIMENT2_SEEDS="1337" \
  NANOGPT_EXPERIMENT2_DEVICE="mps" \
  bash experiment_2/scripts/run_pair.sh level0 all
```

Run the canonical five matched seeds:

```bash
caffeinate -dimsu env \
  NANOGPT_EXPERIMENT2_SEEDS="1337,2027,4099,7919,104729" \
  NANOGPT_EXPERIMENT2_DEVICE="mps" \
  bash experiment_2/scripts/run_pair.sh level0 all
```

## Run Level 1

Calibrate with one matched seed before launching a multi-day campaign:

```bash
caffeinate -dimsu env \
  NANOGPT_EXPERIMENT2_SEEDS="1337" \
  NANOGPT_EXPERIMENT2_DEVICE="mps" \
  bash experiment_2/scripts/run_pair.sh level1 all
```

Then run the canonical five matched seeds:

```bash
caffeinate -dimsu env \
  NANOGPT_EXPERIMENT2_SEEDS="1337,2027,4099,7919,104729" \
  NANOGPT_EXPERIMENT2_DEVICE="mps" \
  bash experiment_2/scripts/run_pair.sh level1 all
```

The runner is safe to execute with `bash`; do not source it. It records the
current pair in:

```text
/tmp/nanogpt-experiment2-level0-current-pair
/tmp/nanogpt-experiment2-level1-current-pair
```

## Run phases separately

```bash
bash experiment_2/scripts/run_pair.sh level0 baseline
bash experiment_2/scripts/run_pair.sh level0 adaptive
bash experiment_2/scripts/run_pair.sh level0 verify
```

The adaptive phase never reruns completed baseline seeds.

## Outputs

Each adaptive run writes:

- `metrics.csv`;
- `layer_measurements.csv`;
- `controller_windows.csv`;
- `projection_events.csv`;
- `controller_summary.json`;
- best, periodic, and final checkpoints;
- selected-checkpoint test metrics;
- `run_complete.json`.

## Notebooks

```bash
jupyter lab experiment_2/notebooks/01_protocol_audit.ipynb
jupyter lab experiment_2/notebooks/02_compare_paired.ipynb
jupyter lab experiment_2/notebooks/03_layer_controller_diagnostics.ipynb
jupyter lab experiment_2/notebooks/04_cross_scale_summary.ipynb
```

The diagnostics include loss curves with error bands, paired seed differences,
all layer alpha trajectories, target-band occupancy, projection alignment,
cohort credit, cooldowns, global pauses, and projection status counts.
