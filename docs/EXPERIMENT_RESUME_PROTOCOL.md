# Experiment Resume Protocol

Restartable checkpoints are written atomically through a temporary file followed by rename. `checkpoints/latest.json` points to the latest verified checkpoint. Resume validation checks configuration, data, tokenizer, initialization, and schema compatibility and reports every mismatch before refusing incompatible continuation.
