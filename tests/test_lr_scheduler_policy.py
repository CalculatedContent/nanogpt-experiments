from pathlib import Path

import pytest

from wwgpt.config import TrainConfig, load_config
from wwgpt.model import GPT
from wwgpt.optim import (
    SCHEDULER_IMPLEMENTATION,
    apply_lr_schedule,
    build_optimizer_bundle,
    nanogpt_cosine_lr,
    optimizer_group_signature,
    resolve_lr_decay_steps,
    resolve_warmup_steps,
    schedule_factor,
)


def tiny_model():
    cfg = load_config(Path("configs/reproduction_tiny.yaml"), level=0).model
    return GPT(cfg)


def test_nanogpt_warmup_formula_and_first_cosine_step():
    assert nanogpt_cosine_lr(0, peak_lr=1.0, warmup_steps=2, lr_decay_steps=10, min_lr_ratio=0.1) == pytest.approx(1 / 3)
    assert nanogpt_cosine_lr(1, peak_lr=1.0, warmup_steps=2, lr_decay_steps=10, min_lr_ratio=0.1) == pytest.approx(2 / 3)
    assert nanogpt_cosine_lr(2, peak_lr=1.0, warmup_steps=2, lr_decay_steps=10, min_lr_ratio=0.1) == pytest.approx(1.0)


def test_cosine_endpoints_midpoint_and_after_horizon():
    assert nanogpt_cosine_lr(2, peak_lr=1.0, warmup_steps=2, lr_decay_steps=12, min_lr_ratio=0.25) == pytest.approx(1.0)
    assert nanogpt_cosine_lr(6, peak_lr=1.0, warmup_steps=2, lr_decay_steps=12, min_lr_ratio=0.25) == pytest.approx(0.6901180666)
    assert nanogpt_cosine_lr(11, peak_lr=1.0, warmup_steps=2, lr_decay_steps=12, min_lr_ratio=0.25) == pytest.approx(0.25)
    assert nanogpt_cosine_lr(12, peak_lr=1.0, warmup_steps=2, lr_decay_steps=12, min_lr_ratio=0.25) == pytest.approx(0.25)


def test_no_warmup_and_one_step_behavior():
    assert nanogpt_cosine_lr(0, peak_lr=2.0, warmup_steps=0, lr_decay_steps=5, min_lr_ratio=0.1) == pytest.approx(2.0)
    assert nanogpt_cosine_lr(4, peak_lr=2.0, warmup_steps=0, lr_decay_steps=5, min_lr_ratio=0.1) == pytest.approx(0.2)
    assert nanogpt_cosine_lr(0, peak_lr=2.0, warmup_steps=0, lr_decay_steps=1, min_lr_ratio=0.1) == pytest.approx(2.0)


def test_default_and_reproduction_profiles_resolve_flat_policy():
    for path in ["configs/default.yaml", "configs/reproduction_tiny.yaml", "configs/reproduction_fineweb.yaml"]:
        cfg = load_config(Path(path), level=0)
        assert cfg.train.lr_schedule == "warmup_cosine"
        assert cfg.train.warmup_steps is None
        assert cfg.train.warmup_ratio == pytest.approx(0.01)
        assert cfg.train.lr_decay_steps is None
        assert cfg.train.min_lr_ratio == pytest.approx(0.10)
        assert cfg.train.layer_lr == "flat"
        assert cfg.train.layer_lr not in {"manual", "llrd"}


def test_flat_mode_optimizer_peak_lrs_and_shared_factor():
    model = tiny_model()
    for base in ["adamw", "muon", "stableadamw"]:
        pytest.importorskip("optimi") if base == "stableadamw" else None
        cfg = TrainConfig(layer_lr="flat", lr_schedule="warmup_cosine", warmup_steps=1)
        bundle, _ = build_optimizer_bundle(model, cfg, base)
        rows = apply_lr_schedule(bundle, 1, 5, 1, cfg)
        assert all(r["normalized_time_factor"] == pytest.approx(1.0) for r in rows)
        by_opt = {}
        for row in rows:
            by_opt.setdefault(row["optimizer_name"], set()).add(row["peak_lr"])
        for peaks in by_opt.values():
            assert len(peaks) == 1


def test_paired_lr_rows_and_weight_decay_signatures_identical():
    model = tiny_model()
    for base in ["adamw", "muon", "stableadamw"]:
        pytest.importorskip("optimi") if base == "stableadamw" else None
        cfg = TrainConfig(layer_lr="flat", wwpgd_interval=2)
        b1, _ = build_optimizer_bundle(model, cfg, base)
        b2, _ = build_optimizer_bundle(tiny_model(), cfg, base)
        assert optimizer_group_signature(b1) == optimizer_group_signature(b2)
        rows1 = [r for step in [0, 1, 2, 4] for r in apply_lr_schedule(b1, step, 5, resolve_warmup_steps(5, cfg.warmup_ratio, cfg.warmup_steps), cfg)]
        rows2 = [r for step in [0, 1, 2, 4] for r in apply_lr_schedule(b2, step, 5, resolve_warmup_steps(5, cfg.warmup_ratio, cfg.warmup_steps), cfg)]
        comparable = ["optimizer_step", "optimizer_name", "parameter_name", "role", "peak_lr", "current_lr", "minimum_lr", "normalized_time_factor", "weight_decay"]
        assert [{k: r[k] for k in comparable} for r in rows1] == [{k: r[k] for k in comparable} for r in rows2]


def test_resume_sequence_is_stateless_and_cli_decay_horizon_changes_factor():
    cfg = TrainConfig(lr_decay_steps=4)
    warmup = resolve_warmup_steps(8, cfg.warmup_ratio, cfg.warmup_steps, cfg.lr_decay_steps)
    assert resolve_lr_decay_steps(8, cfg.lr_decay_steps) == 4
    uninterrupted = [schedule_factor(s, 4, warmup, cfg.lr_schedule, cfg.min_lr_ratio) for s in range(8)]
    resumed = [schedule_factor(s, 4, warmup, cfg.lr_schedule, cfg.min_lr_ratio) for s in range(3)] + [schedule_factor(s, 4, warmup, cfg.lr_schedule, cfg.min_lr_ratio) for s in range(3, 8)]
    assert uninterrupted == resumed
    assert uninterrupted[-1] == pytest.approx(cfg.min_lr_ratio)
    assert SCHEDULER_IMPLEMENTATION == "nanogpt_linear_warmup_cosine_v1"


def test_explicit_llrd_and_manual_work_but_default_is_flat():
    assert TrainConfig().layer_lr == "flat"
    llrd_peaks = {g["peak_lr"] for g in build_optimizer_bundle(tiny_model(), TrainConfig(layer_lr="llrd"), "adamw")[0].optimizers[0].param_groups}
    manual_peaks = {g["peak_lr"] for g in build_optimizer_bundle(tiny_model(), TrainConfig(layer_lr="manual"), "adamw")[0].optimizers[0].param_groups}
    assert len(llrd_peaks) > 1
    assert len(manual_peaks) > 1
