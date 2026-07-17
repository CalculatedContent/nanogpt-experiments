# WW-PGD Strength Scan Protocol

Scientific question: for fixed WW-PGD strengths, how do alpha movement, projection displacement, validation loss, stability, runtime, and WeightWatcher overhead change relative to a paired AdamW control?

Default strengths are `0.02, 0.1, 0.25, 0.5, 1.0`. The design is paired by seed: one initialization, one token order, one AdamW control, identical probes, identical optimizer hyperparameters, identical projection schedule, and only the WW-PGD strength changes.

Strength multiplies scheduled hardness. The logged effective hardness is `schedule_hardness * scan_strength`; effective Cayley and blend etas multiply this value by their configured base etas.

An arm is unstable on non-finite train/validation loss, gradient norm, or parameters; loss above threshold; non-finite projection weights; exception; or incomplete token budget.

Analysis rules: compare optimizer effects across independent seeds only. Layers and projection events are aggregated within runs and are never confidence-interval replicates. One-seed scans report unavailable CIs. Limitations include noisy immediate WeightWatcher fits and small-scan runtime overhead.
