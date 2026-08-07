# Level Zero nanoGPT optimizer baselines

This folder contains the canonical **baseline-only Level Zero experiment** for
`CalculatedContent/nanogpt-experiments`.

It compares three optimizers under one matched nanoGPT protocol:

1. **SGD with Nesterov momentum**;
2. **AdamW**;
3. **Muon on hidden two-dimensional weights**, with auxiliary AdamW for the
   token and position embeddings, output head, normalization parameters, and
   all other non-Muon parameters.

Each optimizer is run with the same three seeds, model, tokenizer, data splits,
effective batch size, evaluation probes, and total token budget. Only the
optimizer-specific hyperparameters and learning-rate schedules differ.

The complete experiment produces:

- nine training runs: three optimizers × three seeds;
- train, validation, and preregistered test metrics by epoch;
- WeightWatcher alpha, ERG gap, fit-quality, rank, and norm diagnostics;
- final and validation-selected checkpoints;
- final-checkpoint text samples;
- one shared baseline-results directory;
- three optimizer-specific notebooks;
- a fourth notebook that compares all three baselines with across-seed error
  bands.

No full scientific result is committed to the repository. The nine real runs
must be executed on the target MacBook Pro.

---

## Fastest complete workflow

Run these commands from a terminal at the repository root:

```bash
git pull --ff-only

export NANOGPT_LEVEL0_ROOT="$HOME/nanogpt-level0-baselines"
mkdir -p "$NANOGPT_LEVEL0_ROOT"

bash level_0_baseline/scripts/setup_mac.sh
bash level_0_baseline/scripts/prepare_data.sh
bash level_0_baseline/scripts/smoke_test.sh

caffeinate -dimsu bash level_0_baseline/scripts/run_all_baselines.sh \
  2>&1 | tee "$NANOGPT_LEVEL0_ROOT/run_all_baselines.log"
```

After the nine runs finish, open the comparison notebook:

```bash
level_0_baseline/.venv-level0/bin/jupyter lab \
  level_0_baseline/notebooks/04_compare_baselines.ipynb
```

The comparison notebook reads the common result store at:

```text
$NANOGPT_LEVEL0_ROOT/baseline_reference
```

The default root is `/tmp/nanogpt-level0-baselines`, but a directory under
`$HOME` is recommended because `/tmp` may be cleaned by the operating system.

**Run every shell script with `bash`. Do not source the scripts.**

---

## Fixed experimental protocol

| Component | Value |
|---|---:|
| Dataset | Pinned FineWeb-Edu `sample-10BT` stream |
| Tokenizer | GPT-2 BPE |
| Vocabulary size | 50,257 |
| Training split | 10,000,000 tokens |
| Validation split | 1,000,000 tokens |
| Test split | 1,000,000 tokens |
| Split construction | Document-disjoint |
| Context length | 256 tokens |
| Transformer blocks | 4 |
| Attention heads | 4 |
| Embedding width | 128 |
| Dropout | 0.0 |
| Micro-batch size | 4 sequences |
| Gradient accumulation | 8 micro-batches |
| Effective batch | 32 sequences |
| Tokens per optimizer step | 8,192 |
| Optimizer steps | 6,104 |
| Total token presentations | 50,003,968 |
| Approximate training passes | 5 |
| Seeds | 1337, 2027, 4099 |
| Numerical precision | float32 |
| Preferred device | Apple MPS |
| Fallback device | CPU |

The full configuration is in:

```text
level_0_baseline/configs/level0.yaml
```

The hyperparameter rationale and source trail are in:

```text
level_0_baseline/HYPERPARAMETERS.md
```

---

## Optimizer settings

| Optimizer | Peak learning rate | Final learning rate | Warm-up | Decay | Other settings |
|---|---:|---:|---:|---|---|
| SGD + Nesterov | 0.05 | 0.005 | 10% | Cosine | Momentum 0.90, weight decay 0.01 |
| AdamW | 6e-4 | 6e-5 | 1% | Cosine | Betas (0.90, 0.95), weight decay 0.10 |
| Muon hidden weights | 0.02 | 0.002 | 5% | Cosine | Momentum 0.95, Nesterov, 5 Newton-Schulz iterations |
| Muon auxiliary AdamW | 3e-4 | 3e-5 | 5% | Cosine | Betas (0.90, 0.95), weight decay 0.01 |

A single numerical learning rate is deliberately not forced across all three
optimizers. Their update geometries and natural scales are different.

---

# Exact run instructions

## 1. Clone or update the repository

For a new checkout:

```bash
git clone https://github.com/CalculatedContent/nanogpt-experiments.git
cd nanogpt-experiments
```

For an existing checkout:

```bash
cd /path/to/nanogpt-experiments
git pull --ff-only
```

Confirm that the folder exists:

```bash
ls level_0_baseline
```

You should see at least:

```text
README.md
COMMON_RESULTS_SCHEMA.md
HYPERPARAMETERS.md
configs/
notebooks/
scripts/
src/
tests/
```

## 2. Choose a persistent output directory

Recommended:

```bash
export NANOGPT_LEVEL0_ROOT="$HOME/nanogpt-level0-baselines"
mkdir -p "$NANOGPT_LEVEL0_ROOT"
```

This produces the following roots automatically:

```text
$NANOGPT_LEVEL0_ROOT/data
$NANOGPT_LEVEL0_ROOT/results
$NANOGPT_LEVEL0_ROOT/baseline_reference
```

Optional explicit overrides are:

```bash
export NANOGPT_LEVEL0_DATA_ROOT=/path/to/data
export NANOGPT_LEVEL0_RESULTS_ROOT=/path/to/results
export NANOGPT_LEVEL0_BASELINE_STORE=/path/to/baseline_reference
```

Use the same environment variables when launching Jupyter so the notebooks read
the same files as the command-line runs.

## 3. Install the Mac environment

```bash
bash level_0_baseline/scripts/setup_mac.sh
```

This command:

- creates `level_0_baseline/.venv-level0`;
- installs PyTorch, NumPy, pandas, matplotlib, WeightWatcher, dataset tooling,
  Jupyter, and test dependencies;
- installs the Level Zero package in editable mode;
- registers the Jupyter kernel `nanoGPT Level 0 Baselines (MPS)`;
- reports whether PyTorch was built with MPS and whether MPS is available.

The final setup output should include lines similar to:

```text
MPS built: True
MPS available: True
```

If `MPS available` is `False`, the experiment can still run on CPU, but it will
be substantially slower.

The run scripts automatically use:

```text
level_0_baseline/.venv-level0/bin/python
```

when that environment exists. Manual activation is not required.

## 4. Run the bounded validation suite

```bash
bash level_0_baseline/scripts/smoke_test.sh
```

This is not the scientific experiment. It is a bounded code-path test that
checks:

- all three optimizer update paths;
- checkpoint creation and restart behavior;
- epoch monitoring;
- the shared baseline-store exporter;
- Bollinger-summary calculations;
- notebook structure;
- protected test handling.

Do not start the long runs until this command exits successfully.

## 5. Prepare the pinned FineWeb-Edu data

```bash
bash level_0_baseline/scripts/prepare_data.sh
```

This step requires internet access the first time. It tokenizes the pinned
FineWeb-Edu stream with GPT-2 BPE and writes:

```text
$NANOGPT_LEVEL0_ROOT/data/meta.json
$NANOGPT_LEVEL0_ROOT/data/train.bin
$NANOGPT_LEVEL0_ROOT/data/val.bin
$NANOGPT_LEVEL0_ROOT/data/test.bin
```

The command prints progress every five seconds. A successful run ends by
printing the output directory. Running it again is safe: if all four files are
present, the script reports that the data already exist and exits.

Check the prepared data with:

```bash
ls -lh "$NANOGPT_LEVEL0_ROOT/data"
cat "$NANOGPT_LEVEL0_ROOT/data/meta.json"
```

The metadata should report GPT-2 tokenization and split sizes of 10M, 1M, and
1M tokens.

## 6. Run all nine baseline jobs

```bash
caffeinate -dimsu bash level_0_baseline/scripts/run_all_baselines.sh \
  2>&1 | tee "$NANOGPT_LEVEL0_ROOT/run_all_baselines.log"
```

This runs, sequentially:

```text
sgd_momentum: seeds 1337, 2027, 4099
adamw:        seeds 1337, 2027, 4099
muon:         seeds 1337, 2027, 4099
```

Sequential execution avoids simultaneous pressure on Apple unified memory.
After training, the same command exports the completed runs into the common
baseline store.

During a healthy run, the terminal periodically prints lines containing:

```text
optimizer=<name>
seed=<seed>
step=<completed>/<6104>
epoch=<approximately 0 through 5>
train_loss=<value>
val_loss=<value>
val_ppl=<value>
val_acc=<value>
```

WeightWatcher measurements also print aggregate alpha and ERG-gap information
at the configured spectral-analysis intervals.

## 7. Confirm that all runs completed

```bash
find "$NANOGPT_LEVEL0_ROOT/results" \
  -name run_complete.json -print | sort
```

There should be exactly nine completion files.

A compact count is:

```bash
find "$NANOGPT_LEVEL0_ROOT/results" \
  -name run_complete.json | wc -l
```

Expected result:

```text
9
```

## 8. Confirm that the shared baseline store exists

```bash
find "$NANOGPT_LEVEL0_ROOT/baseline_reference" \
  -maxdepth 3 -type f | sort
```

At minimum, it should contain:

```text
baseline_reference/store_manifest.json
baseline_reference/all_runs/trajectory_metrics.csv
baseline_reference/all_runs/epoch_checkpoint_metrics.csv
baseline_reference/all_runs/spectral_metrics.csv
baseline_reference/all_runs/terminal_test_metrics.csv
baseline_reference/summaries/trajectory_bollinger_summary.csv
baseline_reference/summaries/epoch_bollinger_summary.csv
baseline_reference/summaries/spectral_bollinger_summary.csv
baseline_reference/summaries/terminal_test_student_t_summary.csv
```

The exact schema is documented in:

```text
level_0_baseline/COMMON_RESULTS_SCHEMA.md
```

## 9. Open the notebooks

Launch Jupyter from the same terminal in which the result-root environment
variables were set:

```bash
level_0_baseline/.venv-level0/bin/jupyter lab \
  level_0_baseline/notebooks
```

Run the notebooks in this order when using the notebook-first workflow:

```text
01_sgd_momentum_baseline.ipynb
02_adamw_baseline.ipynb
03_muon_baseline.ipynb
04_compare_baselines.ipynb
```

The first three notebooks can run or resume their optimizer's three seeds,
produce optimizer-specific plots, export their core CSVs, and generate text
from each final checkpoint.

The fourth notebook does not reconstruct results from model checkpoints. It
reads the standardized CSV tables in `baseline_reference` and compares all
three optimizers on the same axes.

---

# Running smaller pieces

## Run one optimizer across all three seeds

```bash
bash level_0_baseline/scripts/run_multiseed.sh sgd_momentum
bash level_0_baseline/scripts/run_multiseed.sh adamw
bash level_0_baseline/scripts/run_multiseed.sh muon
```

Each multiseed command refreshes that optimizer's common-store export after its
three runs complete.

## Run one optimizer and one seed

```bash
bash level_0_baseline/scripts/run_one.sh adamw 1337
bash level_0_baseline/scripts/run_one.sh sgd_momentum 2027
bash level_0_baseline/scripts/run_one.sh muon 4099
```

## Force a single run to restart from scratch

Use this only when intentionally discarding the existing run directory:

```bash
NANOGPT_LEVEL0_OVERWRITE=1 \
  bash level_0_baseline/scripts/run_one.sh muon 1337
```

## Rebuild the shared result store without retraining

```bash
bash level_0_baseline/scripts/export_baseline_reference.sh
```

Equivalent direct command:

```bash
level_0_baseline/.venv-level0/bin/level0-export-baselines \
  --results-root "$NANOGPT_LEVEL0_ROOT/results" \
  --store-root "$NANOGPT_LEVEL0_ROOT/baseline_reference" \
  --optimizers sgd_momentum,adamw,muon \
  --seeds 1337,2027,4099
```

---

# Restart and interruption behavior

The experiment is designed to survive interruption.

- Completed runs are skipped.
- An incomplete run resumes from `checkpoint_latest.pt`.
- The data-preparation script skips already completed data.
- The common baseline store can be rebuilt at any time from completed runs.

After a terminal interruption, reboot, or MPS process failure, restore the same
root variable and rerun the original command:

```bash
export NANOGPT_LEVEL0_ROOT="$HOME/nanogpt-level0-baselines"

caffeinate -dimsu bash level_0_baseline/scripts/run_all_baselines.sh \
  2>&1 | tee -a "$NANOGPT_LEVEL0_ROOT/run_all_baselines.log"
```

Do not delete `checkpoint_latest.pt` from an incomplete run unless the intent is
to restart that run from step zero.

---

# Metrics and their meaning

## Accuracy

`train_accuracy`, `val_accuracy`, and `test_accuracy` are **exact next-token
top-1 accuracies** over a 50,257-way vocabulary. They are not image or tabular
classification accuracies.

Uniform-random next-token prediction would have approximately:

```text
accuracy   = 1 / 50,257 = 0.00199%
cross-entropy loss      = ln(50,257) ≈ 10.825
perplexity               = 50,257
```

Even a test accuracy in the teens is therefore far above random for this task.

## Loss and perplexity

The loss is next-token cross-entropy. Perplexity is:

```text
perplexity = exp(cross_entropy)
```

Lower loss and lower perplexity are better.

## Generalization gaps

The stored loss gaps are computed as held-out loss minus fixed-train-probe
loss. A growing positive gap indicates that the training probe is improving
faster than the held-out probe. A small gap is not automatically good if both
losses remain high because the optimizer may simply be underfitting.

## WeightWatcher metrics

The experiment calls WeightWatcher with ERG enabled and retains the raw result
tables. Important columns include:

- `alpha` and `alpha_weighted`;
- `ERG_gap`;
- power-law fit distance `D`;
- stable rank and MP soft rank;
- spectral and log norms;
- entropy;
- PL and ERG spike counts.

There is no fallback alpha and no fabricated ERG gap. A failed or unsupported
fit remains missing. Small transformer matrices may produce noisy or missing
fits, so the valid-layer counts must be examined alongside the aggregate
values.

---

# Test-by-epoch policy

The comparison requires genuine test curves rather than validation curves
renamed as test curves. Therefore every run evaluates one fixed test probe at
the preregistered integer-epoch grid:

```text
epoch 1: approximately step 1,221
epoch 2: approximately step 2,441
epoch 3: approximately step 3,662
epoch 4: approximately step 4,883
epoch 5: step 6,104
```

The results are written to:

```text
epoch_metrics.csv
```

with matching model-only checkpoints under:

```text
epoch_checkpoints/
```

These test measurements are monitoring-only. They are never used for:

- optimizer updates;
- learning-rate changes;
- early stopping;
- validation checkpoint selection;
- automatic hyperparameter tuning.

The validation-selected and final test summaries remain separately recorded in
`test_results.json`.

---

# Error bars and comparison statistics

The fourth notebook compares trajectories using an across-seed
**Bollinger-style variability envelope**:

```text
mean ± 2 × sample standard deviation
```

For every optimizer, epoch, and metric:

```text
lower = mean - 2 * sample_sd
upper = mean + 2 * sample_sd
```

These bands describe variation among the three seeds. They are not rolling
financial Bollinger bands and are not confidence intervals.

Final and validation-selected test summaries use two-sided 95% Student-t
confidence intervals across the three seeds.

The colorblind-safe optimizer colors are fixed throughout the notebooks:

- blue: SGD + momentum;
- orange: AdamW;
- green: Muon.

---

# What results should look like

The following are **pre-run sanity expectations**, not guaranteed outcomes and
not pass/fail thresholds. The actual measurements from the target Mac are the
result.

## Basic health checks

A healthy run should show:

- finite losses and gradient norms throughout training;
- a substantial reduction from the random-initialization loss near 10.825;
- increasing next-token accuracy;
- declining train and validation perplexity;
- no persistent NaNs or infinities;
- three broadly similar trajectories for each optimizer, with visible but not
  catastrophic seed variation;
- all five epoch-monitoring rows and all expected checkpoints;
- WeightWatcher tables with at least some valid fitted layers.

## Expected optimizer ordering

The working prior is:

```text
Muon ≈ AdamW faster in token efficiency than SGD + momentum
```

More specifically:

- **AdamW** should show reliable early descent because its warm-up is short.
- **Muon** may look slower during its longer warm-up, then show a steeper middle
  part of training and potentially match or improve on AdamW in validation
  loss.
- **SGD + momentum** is expected to converge more slowly and may finish at a
  higher validation and test loss within the same five-pass budget.

This is a hypothesis to test, not a result assumed by the analysis.

## Broad numerical sanity ranges

For this very small GPT and limited unique-data budget, a plausible final range
is approximately:

```text
held-out top-1 next-token accuracy: 10% to 25%
held-out cross-entropy loss:         5.8 to 7.8
held-out perplexity:                 roughly 330 to 2,400
```

Muon and AdamW are expected toward the stronger end of those broad ranges, with
SGD more likely toward the weaker end. Values outside these ranges are not
automatically wrong, but they should prompt inspection of the data manifest,
learning-rate curve, gradient clipping, and generated text.

## Train versus test behavior

Because ten million unique training tokens are reused for approximately five
passes, it is normal for:

- train accuracy to exceed test accuracy;
- train loss to fall below test loss;
- the generalization gap to widen late in training even while both losses are
  still improving.

A smaller SGD train/test gap does not prove better generalization if SGD has
simply learned less on both splits.

## Generated text

The final cell of each optimizer notebook generates text from every seed's
`checkpoint_final.pt`.

For a model of this size and training budget, expect:

- locally plausible English fragments;
- common phrase and punctuation structure;
- occasional short-range coherence;
- repetition, topic drift, malformed continuations, and factual unreliability.

Do not expect GPT-2-small-quality prose. The text samples are a qualitative
sanity check, while held-out loss and perplexity remain the primary language-
model metrics.

## WeightWatcher and ERG behavior

Do not require every layer to have `alpha ≈ 2`. The matrices are small, and
some fits may be unstable or absent. The useful comparison is the trajectory,
fit quality, valid-layer count, and optimizer-to-optimizer pattern.

Similarly, the ERG gap has no guaranteed sign or monotonic direction in this
baseline. The experiment measures whether its behavior differs systematically
across optimizers and whether it co-moves with loss, generalization gap, alpha,
and fit quality.

---

# Output layout

A completed run tree resembles:

```text
$NANOGPT_LEVEL0_ROOT/
├── data/
│   ├── meta.json
│   ├── train.bin
│   ├── val.bin
│   └── test.bin
├── results/
│   ├── sgd_momentum/
│   │   ├── seed_1337/
│   │   ├── seed_2027/
│   │   └── seed_4099/
│   ├── adamw/
│   │   ├── seed_1337/
│   │   ├── seed_2027/
│   │   └── seed_4099/
│   └── muon/
│       ├── seed_1337/
│       ├── seed_2027/
│       └── seed_4099/
└── baseline_reference/
    ├── store_manifest.json
    ├── all_runs/
    ├── summaries/
    └── per_optimizer/
```

Each seed directory contains approximately:

```text
manifest.json
metrics.csv
epoch_metrics.csv
epoch_checkpoints/
checkpoint_latest.pt
checkpoint_best.pt
checkpoint_final.pt
test_results.json
run_complete.json
generated_samples.json
generated_samples.md
spectral/summary.csv
spectral/layers.csv
spectral/raw/weightwatcher_step_*.csv
```

---

# Completion checklist

The experiment is operationally complete when all of the following are true:

- [ ] `smoke_test.sh` passes.
- [ ] `data/meta.json`, `train.bin`, `val.bin`, and `test.bin` exist.
- [ ] Nine `run_complete.json` files exist.
- [ ] Every run has `checkpoint_final.pt` and `checkpoint_best.pt`.
- [ ] Every run has five epoch-monitoring rows.
- [ ] Every run has WeightWatcher spectral output.
- [ ] Every run has final generated-text samples.
- [ ] `baseline_reference/store_manifest.json` exists.
- [ ] All four combined raw CSVs exist under `baseline_reference/all_runs`.
- [ ] All four summary CSVs exist under `baseline_reference/summaries`.
- [ ] The fourth notebook opens and renders all optimizer overlays.
- [ ] Mean curves and mean ± 2-SD bands contain all three seeds.

---

# Troubleshooting

## MPS is unavailable

Check directly:

```bash
level_0_baseline/.venv-level0/bin/python - <<'PY'
import torch
print(torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
PY
```

To force CPU execution:

```bash
export NANOGPT_LEVEL0_DEVICE=cpu
```

To return to automatic device selection:

```bash
unset NANOGPT_LEVEL0_DEVICE
```

## Jupyter cannot find the kernel

Rerun:

```bash
bash level_0_baseline/scripts/setup_mac.sh
```

Then select:

```text
nanoGPT Level 0 Baselines (MPS)
```

inside Jupyter.

## The comparison notebook reports missing optimizers

First inspect the store:

```bash
find "$NANOGPT_LEVEL0_ROOT/baseline_reference/per_optimizer" \
  -maxdepth 2 -type f | sort
```

Then rebuild it:

```bash
bash level_0_baseline/scripts/export_baseline_reference.sh
```

The full comparison requires completed exports for `sgd_momentum`, `adamw`, and
`muon`.

## A run was interrupted

Rerun the same command. The script resumes incomplete runs from
`checkpoint_latest.pt` and skips completed runs.

## A run repeatedly fails after resume

Restart only that run deliberately:

```bash
NANOGPT_LEVEL0_OVERWRITE=1 \
  bash level_0_baseline/scripts/run_one.sh <optimizer> <seed>
```

Example:

```bash
NANOGPT_LEVEL0_OVERWRITE=1 \
  bash level_0_baseline/scripts/run_one.sh adamw 2027
```

## WeightWatcher fails

The default experiment treats WeightWatcher failures as real failures rather
than inventing substitute alpha or ERG values. Confirm the environment:

```bash
level_0_baseline/.venv-level0/bin/python -m pip check
level_0_baseline/.venv-level0/bin/python - <<'PY'
import weightwatcher
print(weightwatcher.__version__)
PY
```

Fix the installation and rerun the incomplete seed.

## Apple unified-memory pressure

- Close large applications and browser tabs.
- Keep runs sequential.
- Run one optimizer or one seed at a time.
- Do not launch multiple training notebooks simultaneously.
- Reboot and resume if MPS enters a persistent failed state.

## The output root appears empty

Confirm the variable in the current shell:

```bash
printf '%s\n' "$NANOGPT_LEVEL0_ROOT"
```

Jupyter must be launched from a shell with the same environment variable, or
the notebook must be configured to use the same absolute path.

---

# Reproducibility rules

For a valid three-optimizer comparison, do not change any of the following
between runs:

- data files or data manifest;
- tokenizer;
- model architecture;
- seeds;
- effective batch size;
- total training-token budget;
- evaluation probes;
- epoch-monitoring grid;
- WeightWatcher settings.

Optimizer-specific learning rates, warm-ups, momentum/beta values, and decay
schedules are part of the predefined optimizer profiles and should not be
silently modified after inspecting test outcomes.

Any hyperparameter sweep should be run as a separate pilot experiment and
should not overwrite this baseline store.
