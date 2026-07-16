# Legacy Result Invalidation

Runs produced before the methodology repair commit `Repair WeightWatcher, WW-PGD, and validation methodology` are **not scientifically valid WW-PGD results**.

They must be treated only as infrastructure pilots / diagnostic legacy runs because they:

- used a local non-WeightWatcher SVD rank-regression alpha estimator instead of `WeightWatcher.analyze(detX=True, randomize=False, plot=False)`;
- projected the smallest singular values rather than the WeightWatcher-selected large-eigenvalue tail using `xmin` and `detX_num`;
- could contaminate validation estimates by constructing readers from `data.val + data.train`.

Legacy append-only result directories must not be modified or deleted. Analysis code and notebooks must label legacy spectral rows that lack `spectral_estimator == "weightwatcher"` as invalid for WeightWatcher alpha analysis.

Corrected manifests use `scientific_schema_version: 2` and include `spectral_estimator`, `spectral_estimator_version`, `wwpgd_implementation`, `wwpgd_commit`, `projection_schedule`, `validation_probe_hash`, and `training_probe_hash`.
