# Spectral dual-label vocabulary (density α vs rank exponent)

**Class B documentation.** Explains symbols already used at the WW-PGD adapter
boundary. It does not change defaults, mathematics, or public controls.

---

## 1. Why dual-label

Two different exponents appear when talking about power-law spectra:

1. A **density-law** exponent on the eigenvalue density (public HTSR-style α).  
2. A **rank-order** exponent on ordered eigenvalues \(λ_{(r)}\).

They are related under an **ideal continuous** power-law model, but they are not
the same control knob. Logging or plotting only one label under the name “alpha”
is a common source of cross-talk between notebooks and Discord tables.

This repository’s public interface exposes **only** the density target
`wwpgd.target_alpha`. The external WW-PGD package’s rank-law parameter is
**derived inside the adapter** and is not an independently scannable research
control (see also `SCIENTIFIC_INTEGRITY_POLICY.md`).

---

## 2. Symbols

| Symbol | Meaning | In this repo |
|---|---|---|
| **Density α** / `target_alpha` | Target for the density-law / WW-style α used by the adapter | **Public** config + manifests; default `2.0` |
| **Rank-order exponent μ_rank** | Ideal continuous map for ordered eigenvalues | **Derived** at the adapter as \(μ_{\mathrm{rank}} = 1/(α - 1)\); often logged as `derived_external_rank_exponent` |
| **MP q / Q** | Marchenko–Pastur **aspect ratio** | **Reserved** — do **not** use `q` for the rank-order exponent in joint tables |

Ideal correspondence (continuous pure power law, α > 1):

\[
μ_{\mathrm{rank}} = \frac{1}{α - 1},
\qquad
α = 1 + \frac{1}{μ_{\mathrm{rank}}}.
\]

Fixed point of that map: **α = 2 ⇔ μ_rank = 1**.

This is an **exponent correspondence under ideal assumptions**, not independent
evidence that a fitted spectrum is at an RG fixed point. Finite spectra, truncated
fit windows, and fit noise make the empirical relationship approximate.

---

## 3. Logging practice

When writing analysis code or extending projection CSVs:

| Do | Don’t |
|---|---|
| Keep density `target_alpha` and derived rank exponent as **separate columns** | Overwrite one field with the other |
| Name the rank column after the map (`derived_external_rank_exponent` / μ_rank) | Call the rank exponent `q` (conflicts with MP aspect) |
| Record which α convention a WeightWatcher fit used | Assume every package’s “alpha” column is the same estimator |

Projection / dual-label rows introduced for readability (e.g. density α next to
derived rank exponent on projection artifacts) are **logging clarity**, not a
second public training target.

---

## 4. Relation to α = 2

`target_alpha = 2.0` is the repository’s default **public spectral target** for
WW-PGD arms. In broader HTSR discussion, α = 2 is often used as a **boundary**
marker on plots. Those uses are related but not identical:

- **Control:** what the adapter aims at (`target_alpha`).  
- **Description:** where a fitted layer α sits relative to 2 on a baseline plot.

Neither use, by itself, implies a derived universal critical exponent from scale
counting alone, nor that every healthy run must sit exactly at 2.

---

## 5. Related

- Root README — public experiment interface (`wwpgd.target_alpha` only).  
- `docs/SCIENTIFIC_INTEGRITY_POLICY.md` — no fabricated fits; rank exponent not independently scannable.  
- `docs/WWPGD_TELEMETRY_FIELDS.md` — dose / first-apply / event-index read rules (when present).  
- Sibling `rg_optimizers` `OPTIMIZER_VARIANTS.md` — uses μ_rank language for the density↔rank map.
