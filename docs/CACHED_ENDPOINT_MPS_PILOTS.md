# Cached-endpoint MPS pilot procedure

Run these pilots on a Mac, one level at a time. Data, logs, and results remain under
`/tmp`; periodic test-set feedback stays disabled by the experiment YAML.

```bash
export TMP=/tmp
export WWGPT_ROOT="$TMP/wwgpt_mac"
export DATA_ROOT="$WWGPT_ROOT/data"
export TOKENIZERS_PARALLELISM=false

for LEVEL in 0 1 2; do
  caffeinate -i wwgpt run-multiseed \
    --config "configs/level${LEVEL}_adaptive_alpha.yaml" --level "$LEVEL" \
    --data-root "$DATA_ROOT" --results-root "$WWGPT_ROOT/pilot_level${LEVEL}" \
    --token-multiplier 20 --seeds 1337 --optimizer adamw \
    --extensions none,wwpgd --device mps --precision fp32
  wwgpt audit-experiment --experiment-root "$WWGPT_ROOT/pilot_level${LEVEL}"
done
```

For each audit, require completion and readable endpoint measurement and relaxation
artifacts in the WW-PGD arm, no WW-PGD artifacts in the baseline, stock call and
candidate-generation counts to agree, projected-matrix count to equal changed fast
rows, protected embeddings/head to remain bitwise unchanged, and finite weights and
endpoints.

Only after all three pilots pass may the full three-seed run be launched:

```bash
for LEVEL in 0 1 2; do
  caffeinate -i wwgpt run-multiseed \
    --config "configs/level${LEVEL}_adaptive_alpha.yaml" --level "$LEVEL" \
    --data-root "$DATA_ROOT" --results-root "$WWGPT_ROOT/full_level${LEVEL}" \
    --token-multiplier 20 --seeds 1337,2027,4099 --optimizer adamw \
    --extensions none,wwpgd --device mps --precision fp32
done
```
