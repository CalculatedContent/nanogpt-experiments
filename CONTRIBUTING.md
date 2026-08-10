# Contributing

Thanks for interest in improving this repository. This project prioritizes
**reproducible experiment infrastructure** and honest reporting over positive
efficacy narratives.

## Workflow

1. Fork `CalculatedContent/nanogpt-experiments` (or push to your existing fork).
2. Branch from current **`main`** (not from stale feature branches).
3. Open a pull request against `CalculatedContent/nanogpt-experiments` `main`.
4. Prefer **small, single-theme** PRs (docs, logging, or one behavior change).

## Before you open a PR

From a clean checkout of your branch:

```bash
# infrastructure smoke only — not a scientific result
./scripts/run_smoke_test.sh /tmp
```

Fix any smoke failures you introduced. Smoke success does **not** authorize
scaling-law, acceleration, or generalization claims.

If your change touches the WW-PGD adapter or projection CSV schema, also run the
repository CI suite (or the subset your PR description names) and note results
in the PR body.

## Naming: Level 0 is overloaded

Schema/runner Levels 0–2 (`wwgpt --level N`, `configs/levelN_*.yaml`) are **not**
the same as the `level_0_baseline/` / `level_0_wwpgd/` packages. Schema Level 0 is
typically **1 layer · 1 head · width 64**; the baseline packages use **4 layers ·
4 heads · width 128**. Prefer the package READMEs when working under those folders.

## PR expectations

| Do | Don't |
|---|---|
| State whether the PR is docs-only, logging, or behavior | Claim WW-PGD efficacy or statistical significance from this stack alone |
| Keep defaults unchanged unless the PR is explicitly about a default | Fabricate α / invent missing spectral fits |
| Name any new log field and its definition | Treat smoke or notebook demos as publication results |
| Link related docs (`docs/SCIENTIFIC_INTEGRITY_POLICY.md`, `docs/CONTROL_ARMS.md`) when relevant | Bundle unrelated refactors with a one-line fix |

Suggested PR body checklist:

```text
## Type
- [ ] docs only
- [ ] logging / schema field
- [ ] behavior (describe default impact)

## Summary
...

## Test plan
- [ ] smoke / CI
- [ ] no scientific efficacy claim

## Out of scope
...
```

## Where to look

| Topic | Path |
|---|---|
| Root overview + level naming | [`README.md`](README.md) |
| Scientific integrity | [`docs/SCIENTIFIC_INTEGRITY_POLICY.md`](docs/SCIENTIFIC_INTEGRITY_POLICY.md) |
| Control arms | [`docs/CONTROL_ARMS.md`](docs/CONTROL_ARMS.md) |
| Troubleshooting | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| Baseline package (4L/128) | [`level_0_baseline/README.md`](level_0_baseline/README.md) |
| WW-PGD package sibling | [`level_0_wwpgd/README.md`](level_0_wwpgd/README.md) |

## License

By contributing, you agree that your contributions are licensed under the same
terms as this repository (`LICENSE`).
