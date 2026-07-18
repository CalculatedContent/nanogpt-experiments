# Scientific schema v3

Schema v3 composes a base optimizer (`adamw`, `muon`, `stableadamw`) with an extension (`none`, `wwpgd`). Canonical arms are `adamw`, `adamw_wwpgd`, `muon`, `muon_wwpgd`, `stableadamw`, and `stableadamw_wwpgd`; paired effects compare the same base optimizer with and without WW-PGD.

The model ladder uses one-layer level 0 `(1 layer, 1 head, width 64, block 256)` and levels `L>=1` use `n_layer=2L`, `n_head=L+1`, `n_embd=64(L+1)`, preserving 64-dimensional heads. Attention uses separate bias-free `key`, `query`, `value`, and `proj` matrices. MLP linear layers and the LM head are bias-free by default, and the LM head is untied from token embeddings.

Training defaults are batch size 16, gradient accumulation 1, weight decay 0.01, dropout 0, and gradient clipping disabled when `grad_clip=0.0`. The default schedule is warmup-cosine with warmup ratio 0.05 and minimum LR ratio 0.10. Layer-wise LR decay derives gamma so the lowest semantic depth receives `llrd_min_multiplier` and the highest depth receives 1.0.

Token budgets use actual instantiated trainable parameter counts. With default `token_multiplier=20`, target tokens equal `20 * parameter_count_used`, steps are `ceil(target_tokens / (batch_size * block_size * gradient_accumulation))`, and realized tokens are the resulting full-step count times tokens per step. `max_steps` overrides `max_train_tokens`, which overrides token multiplier.

Evaluation defaults to `random_per_eval`: new deterministic random train and validation windows are sampled at each evaluation event from independent SHA-256-derived streams. Paired arms share seeds and therefore evaluation hashes. Evaluation does not advance the training reader and restores train/eval mode.

WW-PGD is a post-step extension run every `wwpgd_interval` optimizer steps (defaulting to `eval_interval`). It projects only raw eligible block matrices using WeightWatcher-selected large-eigenvalue tails from `xmin` and `detX_num`; embeddings, LayerNorms, biases, and the LM head are excluded by default.

Spectral diagnostics run independently of evaluation. Raw matrices include W_K, W_Q, W_V, W_O, W_MLP_IN, and W_MLP_OUT. Composite diagnostics include `KQ=W_K@W_Q`, `QK=W_Q@W_K`, `QK_effective=W_Q.T@W_K`, `KQ_effective=W_K.T@W_Q`, `OV=sum_h W_O,h@W_V,h`, `VO=W_V@W_O`, and `MLP_IO=W_MLP_OUT@W_MLP_IN`. Composite matrices are diagnostics only.

Schema-v2 and schema-v3 runs remain readable, but analysis must not pool them in statistical comparisons because optimizer arms, architecture, evaluation sampling, and projection schedules differ.
