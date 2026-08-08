# Optional `WW_PGD` commit pin

**Default:** floating install from `pyproject.toml` (`ww-pgd @ git+…`), unchanged.

**Optional pin** for joint runs / freeze protocols:

```bash
export WWPGD_COMMIT_PIN=<full_or_prefix_sha>
```

When set, training with a WW-PGD extension **fails at run setup** if the installed package’s PEP 610 VCS `commit_id` does not match (prefix match allowed).

| Env | Effect |
|---|---|
| unset / empty | floating install (default) |
| set to SHA | require installed `ww_pgd` commit match |

Provenance fields in the run manifest:

- `wwpgd_resolved_commit`
- `wwpgd_commit_pin_requested`
- `wwpgd_dependency_pinned` (true only when env is set)

This does **not** change spectral targets, dose, or training defaults. It only makes actuator version drift fail loudly when you ask for a pin.
