# Contributing

Thanks for interest in improving this repository. This project prioritizes
**reproducible experiment infrastructure** and honest reporting over positive
efficacy narratives.

This guide is for people sending **pull requests**. How you get a branch onto
GitHub depends on your access (see below). The **integrity / PR expectations**
apply to external and write-access contributors alike; maintainers may use a
shorter internal path for trivial fixes.

## Who you are (access paths)

| Role | Typical access | How you open a PR |
|---|---|---|
| **External contributor** | No push access to this repo | **Fork** the repo under your account, push a branch to *your* fork, open a PR into `CalculatedContent/nanogpt-experiments` `main` |
| **Collaborator / write access** | Can push branches to this repo | Create a branch on **this** repo from current `main`, push it here, open a PR into `main` (no personal fork required) |
| **Maintainer** | Admin / merge rights | Same as write access for reviewable changes; direct commits to `main` only for the project’s usual maintainer practice (prefer PRs for non-trivial work) |

**Fork is not a social rank.** It is only the usual GitHub path when you cannot
push to this repository. If you already have write access, do not fork just to
satisfy a ritual.

### External contributor (no write access)

1. Fork `CalculatedContent/nanogpt-experiments`.
2. Branch from current upstream **`main`** (not from stale feature branches).
3. Push the branch to **your fork**.
4. Open a PR against `CalculatedContent/nanogpt-experiments` `main`.

### Collaborator with write access

1. Clone this repository (or add it as `origin`).
2. Branch from current **`main`**.
3. Push the branch to **this** repo.
4. Open a PR against `main`.

Prefer **small, single-theme** PRs (docs, logging, or one behavior change) in
either path.

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

These expectations apply to **any** PR author (external or write-access), unless
a maintainer explicitly says otherwise for a given change.

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
