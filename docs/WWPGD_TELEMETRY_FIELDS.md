# WW-PGD telemetry fields (how to read projection CSVs)

**Class B documentation.** Helps readers interpret WW-PGD projection / dose /
event fields without inventing missing quantities. It does not change training
code or defaults.

For density α vs derived rank exponent vocabulary, see
[`SPECTRAL_DUAL_LABEL.md`](SPECTRAL_DUAL_LABEL.md)
(μ_rank vs μ_entry vs MP aspect $q$ — never overload bare `q`).

---

## 1. Why this page exists

CSV rows look like science tables. Misreading them recreates confounds we already
know from experiment design:

| Misread | Better read |
|---|---|
| Projection **event index** = dose | Event index is a **schedule clock**; dose is a **strength** |
| First scheduled event = first real apply | Warmup, gates, trust region, or zero hardness can skip applies |
| Requested pin = verified pin | Runtime check and install pin are different facts |

---

## 2. Fields commonly present on `main` (scientific runner)

Exact headers evolve; always trust the CSV header of the run you analyze. Typical
projection / adapter-related columns include:

| Field (examples) | Role |
|---|---|
| `projection_event` | Schedule / event index for the projection cadence |
| `optimizer_step` / `actual_step` | Base optimizer step when the row was written |
| `layer_name` | Which matrix the row refers to |
| `target_alpha` | Public density spectral target |
| `derived_external_rank_exponent` | Derived rank-order exponent \(1/(α-1)\) at the adapter boundary |
| `relative_frobenius_change_applied` | Realized relative Frobenius movement of the applied update (when present) |
| `relative_frobenius_change_requested` | Requested movement before trust-region / scaling (when present) |
| `projected` / skip reasons | Whether an apply happened and why not |

**Rule:** if a column is missing, treat it as missing — never invent midpoint /
Cayley / TraceLog internals when the installed WW-PGD build does not expose them
(see root README adapter policy).

---

## 3. Conflict log and SoT (predicates)

| Intent | Earlier risk | Canonical direction |
|---|---|---|
| First-event / first-apply flags | Treating `event == 1` (or 1-based index) as first real apply | Split **first scheduled event** vs **first actual apply per layer** (positive realized dose that changes the layer). Upstream work: open PR discussion **#131** (resume-safe dose telemetry) — prefer that design over ad-hoc first-event booleans. |
| Optional WW_PGD commit pin | Check only in one runner; soft prefix match; stale import provenance | Verify at **every shared adapter boundary**; distinguish requested vs verified vs dependency-pinned. Upstream work: open PR **#132**. |
| Dual-label α / rank | Single column named “alpha” | Density target + derived rank exponent as separate fields (#128 logging + [`SPECTRAL_DUAL_LABEL.md`](SPECTRAL_DUAL_LABEL.md)). |

If #131 / #132 are not yet merged when you read this, treat their **predicate
design** as the intended SoT and mark any not-yet-on-main columns as pending.

---

## 4. Read rules (always)

1. **Scheduled ≠ applied.** A projection event can be scheduled and still apply
   zero dose (or skip a layer).  
2. **Dose is applied strength**, preferably relative Frobenius of the **applied**
   update after trust-region / scaling — not hardness alone.  
3. **Per-layer first apply** is not the same as global event zero.  
4. **Pin provenance** has three facts: what is installed, what was requested, whether
   runtime verification passed.  
5. **No efficacy claim** follows from a field being present.  
6. **Dual-label:** density `target_alpha` and `derived_external_rank_exponent`
   are different columns; do not rename the rank exponent to bare `q` (MP aspect).

---

## 5. Related

- Root README — WW-PGD installation policy; public `target_alpha` only.  
- `docs/SPECTRAL_DUAL_LABEL.md` — density α vs μ_rank.  
- `docs/SCIENTIFIC_INTEGRITY_POLICY.md` — no fabricated spectral fields.  
- `docs/CONTROL_ARMS.md` — measurement-only / sham arms when comparing dose.
