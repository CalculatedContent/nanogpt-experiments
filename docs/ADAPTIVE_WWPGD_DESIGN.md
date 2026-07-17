# Future Adaptive WW-PGD Design

This document is design-only; no adaptive policy is implemented.

Future policies could use proportional control based on absolute alpha error, fit-quality gating on WeightWatcher status/D/num_evals, trust-region limits on relative Frobenius change, smoothed alpha estimates across projection events, and backtracking when alpha error worsens. Raw per-layer alpha is too noisy for immediate unfiltered control, so adaptation should aggregate within events, cap displacement, and avoid treating failed or low-quality fits as direct control targets.
