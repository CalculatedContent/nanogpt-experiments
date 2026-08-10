# Spectral dual-label vocabulary (density α vs rank exponent)

**Class B documentation.** Explains symbols already used at the WW-PGD adapter
boundary. It does not change defaults, mathematics, or public controls.

---

## 1. Why dual-label

Two different exponents appear when talking about power-law spectra:

1. A **density-law** exponent on the eigenvalue density (public HTSR-style α).  
2. A **rank-order** exponent on ordered eigenvalues \(\lambda_{(r)}\).

They are related under an **ideal continuous** power-law model, but they are not
the same control knob. Logging or plotting only one label under the name “alpha”
is a common source of cross-talk between notebooks and Discord tables.

This repository’s public interface exposes **only** the density target
`wwpgd.target_alpha`. The external WW-PGD package’s rank-law parameter is
**derived inside the adapter** and is not an independently scannable research
control (see also `SCIENTIFIC_INTEGRITY_POLICY.md`).

---

## 2. Symbols (collision box)

| Symbol | Meaning | In this repo |
|---|---|---|
| **Density α** / `target_alpha` | Target for the density-law / WW-style α: \(\rho(\lambda)\propto\lambda^{-\alpha}\) on a fitted tail | **Public** config + manifests; default `2.0` |
| **Rank-order exponent \(\mu_{\mathrm{rank}}\)** | Ideal continuous map for ordered eigenvalues | **Derived** at the adapter as \(\mu_{\mathrm{rank}} = 1/(\alpha - 1)\); log as `derived_external_rank_exponent` |
| **MP \(q\) / \(Q\)** | Marchenko–Pastur **aspect ratio** \(N/M\) | **Reserved** — do **not** use bare `q` for the rank-order exponent |
| **\(\mu_{\mathrm{entry}}\)** (HTSR theory) | Heavy-tailed matrix-**element** tail index (Universality class parameter) | **Not** a public control here; **not** the same object as \(\mu_{\mathrm{rank}}\) |

### Ideal density ↔ rank map

Continuous pure power law, \(\alpha > 1\) (Newman / Zipf–Pareto algebra):

\[
\mu_{\mathrm{rank}} = \frac{1}{\alpha - 1},
\qquad
\alpha = 1 + \frac{1}{\mu_{\mathrm{rank}}}.
\]

Fixed point of that map: **α = 2 ⇔ \(\mu_{\mathrm{rank}} = 1\)**.

This is an **exponent correspondence under ideal assumptions**, not independent
evidence that a fitted spectrum is at an RG fixed point. Finite spectra,
\(x_{\min}\) selection, truncated tails, and fit noise make the empirical
relationship approximate (Clauset–Shalizi–Newman 2009 methodology).

### Separate: density α ↔ entry-tail \(\mu_{\mathrm{entry}}\) (HTSR)

In Heavy-Tailed Self-Regularization, the theoretical **matrix-element** tail
index \(\mu_{\mathrm{entry}}\) is related to the empirical ESD exponent in the
Very-Heavy-Tailed (Lévy) class by Martin & Mahoney (arXiv:1901.08278, Eq. A.4a):

\[
\text{VHT:}\quad \alpha = 1 + \frac{\mu_{\mathrm{entry}}}{2}
\quad\Rightarrow\quad
\mu_{\mathrm{entry}} = 2(\alpha - 1)
\quad\text{for }0 < \mu_{\mathrm{entry}} < 2.
\]

So **α = 2 corresponds to \(\mu_{\mathrm{entry}} = 2\)** — the VHT / MHT class
boundary — under that asymptotic VHT map. In the Moderately-Heavy-Tailed class
the relation is \(\alpha = a\mu + b\) with **strong finite-size** dependence on
\(M,N\) (Eq. A.4b); do not treat the linear VHT map as universal across classes.

**Collision warning:** \(\mu_{\mathrm{entry}}\cdot\mu_{\mathrm{rank}} = 2\) under the
two ideal maps above is an **algebraic coincidence**, not an identity of
objects. Always tag which μ you mean.

### Eigenvalue vs singular-value density

WeightWatcher-style ESDs are usually built from eigenvalues of \(X = W^\top W\)
(or a scaled form). If \(s = \sqrt{\lambda}\), density exponents convert as
\(\alpha_s = 2\alpha_\lambda - 1\). A “rank exponent” also needs an
eigenvalue-vs-singular-value label when comparing notebooks.

---

## 3. Logging practice

When writing analysis code or extending projection CSVs:

| Do | Don’t |
|---|---|
| Keep density `target_alpha` and derived rank exponent as **separate columns** | Overwrite one field with the other |
| Name the rank column after the map (`derived_external_rank_exponent` / \(\mu_{\mathrm{rank}}\)) | Call the rank exponent `q` (conflicts with MP aspect) |
| Record which α convention a WeightWatcher fit used | Assume every package’s “alpha” column is the same estimator |
| Tag \(\mu_{\mathrm{entry}}\) if you ever log HTSR class maps | Use bare `mu` for both entry-tail and rank-order |

Projection / dual-label rows introduced for readability (e.g. density α next to
derived rank exponent on projection artifacts) are **logging clarity**, not a
second public training target.

---

## 4. Relation to α = 2

`target_alpha = 2.0` is the repository’s default **public spectral target** for
WW-PGD arms. In broader HTSR discussion, α = 2 is often used as a **boundary**
marker on plots (VHT/MHT class boundary via \(\mu_{\mathrm{entry}}\); operational
“near 2” quality language). Those uses are related but not identical:

- **Control:** what the adapter aims at (`target_alpha`).  
- **Description:** where a fitted layer α sits relative to 2 on a baseline plot.  
- **Theory class boundary:** \(\mu_{\mathrm{entry}} = 2\) under the VHT map (A.4a).

None of these, by itself, implies a **derived universal critical exponent from
scale counting alone**, nor that every healthy run must sit exactly at 2, nor
that “α > 2 means random-like” (moderately heavy-tailed spectra commonly sit
above 2; random-like is an MP / phase statement).

---

## 5. Related

- Root README — public experiment interface (`wwpgd.target_alpha` only).  
- `docs/SCIENTIFIC_INTEGRITY_POLICY.md` — no fabricated fits; rank exponent not independently scannable.  
- `docs/WWPGD_TELEMETRY_FIELDS.md` — dose / first-apply / event-index read rules.  
- Sibling `rg_optimizers` `OPTIMIZER_VARIANTS.md` — density↔rank map; reserves `q` for MP aspect.  
- Primary anchors (external): Newman power laws; Clauset–Shalizi–Newman 2009; Martin & Mahoney arXiv:1901.08278 App. A.
