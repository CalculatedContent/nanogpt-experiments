# Optional WW-PGD runtime commit verification

The repository continues to install WW-PGD from a floating Git dependency:

```text
ww-pgd @ git+https://github.com/CalculatedContent/WW_PGD.git
```

For a frozen or jointly reproduced experiment, require the installed package to resolve to a particular Git commit by setting:

```bash
export WWPGD_COMMIT_PIN=<full_sha_or_hex_prefix>
```

The value must be a hexadecimal Git SHA prefix containing at least 12 and at most 64 characters. A full SHA is accepted. Matching is one-directional: the installed PEP 610 `commit_id` must start with the requested value.

## Enforcement

Verification occurs at the shared pip adapter boundary used by:

- the main scientific Level 0–2 runner;
- the isolated Level Zero WW-PGD runner;
- Experiment 2 adaptive WW-PGD;
- direct callers of the common WW-PGD configuration or candidate functions.

A requested pin is checked before constructing a WW-PGD configuration and again before invoking the installed projector. A mismatch, malformed pin, or installation without a PEP 610 VCS commit fails loudly before a projection is accepted.

When the environment variable is unset or empty, behavior is unchanged and the floating installation is allowed.

## Manifest fields

WW-PGD manifests distinguish the installation specification from runtime verification:

- `wwpgd_resolved_commit`: installed PEP 610 VCS commit;
- `wwpgd_commit_pin_requested`: normalized requested SHA or `null`;
- `wwpgd_commit_pin_verified`: `true` only after a successful comparison;
- `wwpgd_commit_pin_status`: `floating` or `verified`;
- `wwpgd_dependency_pinned`: always `false` while `pyproject.toml` remains floating.

The provenance mapping evaluates the environment when the manifest is built rather than only when Python imports `wwgpt.ww`. This prevents notebooks and long-lived processes from verifying one pin state while recording stale import-time metadata.

This feature changes no spectral target, dose, trust-region rule, optimizer update, model weight, or default training behavior.
