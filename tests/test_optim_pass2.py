import math
from pathlib import Path

import pytest
import torch

from wwgpt.config import TrainConfig, load_config
from wwgpt.model import GPT
from wwgpt.optim import (
    MANUAL_LAYER_LR_MULTIPLIERS,
    Muon,
    apply_lr_schedule,
    arm_name,
    build_optimizer_bundle,
    build_param_groups,
    muon_parameter_names,
    optimizer_group_signature,
    schedule_factor,
)


def tiny_model():
    cfg = load_config(Path("configs/reproduction_tiny.yaml"), level=0).model
    return GPT(cfg)


def test_exact_manual_multipliers():
    cfg = TrainConfig(layer_lr="manual", learning_rate=1.0)
    groups, _ = build_param_groups(tiny_model(), 1.0, 0.01, cfg)
    by_name = {g["parameter_name"]: g for g in groups}
    expected = {
        "wte.weight": 0.35,
        "wpe.weight": 0.35,
        "blocks.0.attn.key.weight": 0.70,
        "blocks.0.attn.query.weight": 0.70,
        "blocks.0.attn.value.weight": 0.70,
        "blocks.0.attn.proj.weight": 0.80,
        "blocks.0.mlp.0.weight": 1.00,
        "blocks.0.mlp.2.weight": 1.10,
        "blocks.0.ln_1.weight": 1.20,
        "blocks.0.ln_2.weight": 1.20,
        "ln_f.weight": 1.20,
        "lm_head.weight": 1.35,
    }
    assert MANUAL_LAYER_LR_MULTIPLIERS["other"] == 1.00
    for name, mult in expected.items():
        assert by_name[name]["layer_lr_multiplier"] == pytest.approx(mult)
        assert by_name[name]["peak_lr"] == pytest.approx(mult)


def test_warmup_peak_midpoint_and_final_cosine_lr():
    min_ratio = 0.25
    assert schedule_factor(0, 102, 3, "warmup_cosine", min_ratio) == pytest.approx(1 / 3)
    assert schedule_factor(2, 102, 3, "warmup_cosine", min_ratio) == pytest.approx(1.0)
    assert schedule_factor(52, 102, 3, "warmup_cosine", min_ratio) == pytest.approx(0.625)
    assert schedule_factor(101, 102, 3, "warmup_cosine", min_ratio) == pytest.approx(min_ratio)


def test_schedule_updates_all_muon_and_auxiliary_adamw_groups():
    cfg = TrainConfig(layer_lr="manual", lr_schedule="warmup_cosine", min_lr_ratio=0.25)
    bundle, _ = build_optimizer_bundle(tiny_model(), cfg, "muon")
    rows = apply_lr_schedule(bundle, 0, 10, 2, cfg)
    assert {r["optimizer_name"] for r in rows} == {"muon", "muon_aux_adamw"}
    assert len(rows) == sum(len(opt.param_groups) for _, opt in bundle.scheduled_optimizers)
    for _, opt in bundle.scheduled_optimizers:
        for group in opt.param_groups:
            assert group["lr"] == pytest.approx(group["peak_lr"] * 0.5)


def _update_norm_for_shape(shape):
    return float(Muon._orthogonalize(torch.ones(shape), 0).norm())


def test_muon_tall_square_and_wide_matrix_scaling():
    assert _update_norm_for_shape((8, 2)) == pytest.approx(2.0)
    assert _update_norm_for_shape((4, 4)) == pytest.approx(1.0)
    assert _update_norm_for_shape((2, 8)) == pytest.approx(1.0)


def test_complete_and_disjoint_muon_parameter_partitioning():
    model = tiny_model()
    mnames = muon_parameter_names(model)
    all_names = {n for n, p in model.named_parameters() if p.requires_grad}
    aux_names = all_names - mnames
    assert mnames.isdisjoint(aux_names)
    assert mnames | aux_names == all_names
    assert mnames == {
        "blocks.0.attn.key.weight",
        "blocks.0.attn.query.weight",
        "blocks.0.attn.value.weight",
        "blocks.0.attn.proj.weight",
        "blocks.0.mlp.0.weight",
        "blocks.0.mlp.2.weight",
    }
    assert {"wte.weight", "wpe.weight", "ln_f.weight", "lm_head.weight"}.issubset(aux_names)


def test_stableadamw_construction_without_skipping():
    optimi = pytest.importorskip("optimi")
    bundle, _ = build_optimizer_bundle(tiny_model(), TrainConfig(layer_lr="manual"), "stableadamw")
    assert isinstance(bundle.optimizers[0], optimi.StableAdamW)


def test_no_double_weight_decay_in_stableadamw():
    pytest.importorskip("optimi")
    cfg = TrainConfig(layer_lr="manual", weight_decay=0.005)
    bundle, _ = build_optimizer_bundle(tiny_model(), cfg, "stableadamw")
    opt = bundle.optimizers[0]
    decays = [g["weight_decay"] for g in opt.param_groups]
    assert opt.defaults["weight_decay"] == pytest.approx(0.0)
    assert max(decays) == pytest.approx(0.005)


def _signature_for_arm(base, extension):
    # Optimizer construction is keyed only by the paired base optimizer; the
    # extension is represented in the arm name and must not mutate groups.
    assert arm_name(base, extension) in {base, f"{base}_wwpgd"}
    return optimizer_group_signature(build_optimizer_bundle(tiny_model(), TrainConfig(layer_lr="manual"), base)[0])


@pytest.mark.parametrize("base", ["adamw", "muon", "stableadamw"])
def test_identical_optimizer_signatures_for_paired_base_and_wwpgd(base):
    pytest.importorskip("optimi") if base == "stableadamw" else None
    assert _signature_for_arm(base, "none") == _signature_for_arm(base, "wwpgd")


@pytest.mark.parametrize("path,expected", [("configs/reproduction_tiny.yaml", 0.005), ("configs/reproduction_fineweb.yaml", 0.005), ("configs/default.yaml", 0.01)])
def test_profile_weight_decay_in_both_paired_arms(path, expected):
    cfg = load_config(Path(path), level=0).train
    for base in ("adamw", "muon"):
        sig = optimizer_group_signature(build_optimizer_bundle(tiny_model(), cfg, base)[0])
        assert max(row["weight_decay"] for row in sig) == pytest.approx(expected)
