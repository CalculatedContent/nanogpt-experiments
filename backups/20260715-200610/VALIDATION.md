# Validation

- PASS: `ruff check .` completed with all checks passed.
- PASS: `pytest -q` completed with 13 passed and 2 warnings.
- PASS: `./scripts/run_smoke_test.sh /tmp` created `/tmp/wwgpt_smoke_invalid_20260715-050027_8c8fd514`.
- PASS: both AdamW and AdamW+WW-PGD wrote `run_complete.json` in the smoke output.
- PASS: two `metrics.csv` files and two `spectral.csv` files exist in the smoke output.
- PASS: one `wwpgd_projection.csv` file exists for the WW-PGD arm.
- PASS: twelve checkpoint files exist in the smoke output.
- PASS: scripts and Python source contain no standalone deletion command tokens from the required search pattern.
- PASS: all shell scripts under `scripts/` are executable.
- PASS: all notebooks parsed with nbformat; nbformat emitted a missing-id warning that it normalized in memory.
- PASS: README quick-start commands match the implemented script names and CLI entry points.
- PASS: `wwgpt analyze-results /tmp/wwgpt_smoke_invalid_20260715-050027_8c8fd514` created `/tmp/wwgpt_smoke_invalid_20260715-050027_8c8fd514/analysis/analysis_20260715-050044_98469403` with error-bar tables and a plot.
- PASS: `wwgpt plan-scaling --params 1000 --token-multiplier 20 --available-tokens 10` returned `scaling_valid=False` for an undersized corpus.
- PASS: a second smoke run created `/tmp/wwgpt_smoke_invalid_20260715-050058_8fc346ea`, distinct from the first output directory.

Known validation limitations: this is an implementation smoke validation only. It does not constitute scientific evidence about WW-PGD, FineWeb-EDU scaling, or five-seed statistical conclusions.
