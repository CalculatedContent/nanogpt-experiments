from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from wwgpt.config import TrainConfig

MANUAL_LAYER_LR_MULTIPLIERS = {
    "token_embedding": 0.35,
    "position_embedding": 0.35,
    "attention_key": 0.70,
    "attention_query": 0.70,
    "attention_value": 0.70,
    "attention_projection": 0.80,
    "mlp_input": 1.00,
    "mlp_output": 1.10,
    "block_layernorm": 1.20,
    "final_layernorm": 1.20,
    "lm_head": 1.35,
    "other": 1.00,
    "block_other": 1.00,
}

BASE_OPTIMIZERS = {"adamw", "muon", "stableadamw", "stable_adamw"}
EXTENSIONS = {"none", "wwpgd"}
ARM_DISPLAY = {
    "adamw": "AdamW",
    "adamw_wwpgd": "AdamW+WW-PGD",
    "muon": "Muon",
    "muon_wwpgd": "Muon+WW-PGD",
    "stableadamw": "StableAdamW",
    "stableadamw_wwpgd": "StableAdamW+WW-PGD",
    "stable_adamw": "StableAdamW",
    "stable_adamw_wwpgd": "StableAdamW+WW-PGD",
}


def arm_name(base_optimizer: str, extension: str) -> str:
    if base_optimizer not in BASE_OPTIMIZERS or extension not in EXTENSIONS:
        raise ValueError(f"invalid optimizer/extension: {base_optimizer}/{extension}")
    return base_optimizer if extension == "none" else f"{base_optimizer}_{extension}"


@dataclass
class OptimizerBundle:
    name: str
    optimizers: list[torch.optim.Optimizer]
    scheduled_optimizers: list[tuple[str, torch.optim.Optimizer]]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for opt, opt_state in zip(self.optimizers, state["optimizers"], strict=True):
            opt.load_state_dict(opt_state)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-2, momentum=0.95, weight_decay=0.01, newton_schulz_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, newton_schulz_steps=newton_schulz_steps))

    @staticmethod
    def _orthogonalize(g: torch.Tensor, steps: int) -> torch.Tensor:
        a, b, c = 3.4445, -4.7750, 2.0315
        x = g.float()
        if x.ndim != 2:
            return x.to(g.dtype)
        transposed = x.size(0) > x.size(1)
        if transposed:
            x = x.T
        x = x / (x.norm() + 1e-7)
        for _ in range(steps):
            xx = x @ x.T
            x = (a * x) + (b * xx + c * (xx @ xx)) @ x
        if transposed:
            x = x.T
        rows, columns = g.size(0), max(1, g.size(1))
        scale = max(1.0, rows / columns) ** 0.5
        return (x * scale).to(g.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                update = self._orthogonalize(buf, group["newton_schulz_steps"])
                p.add_(update, alpha=-lr)
        return loss


def parameter_role_depth(name: str, model: nn.Module) -> tuple[str, int]:
    if name.startswith("wte."):
        return "token_embedding", 0
    if name.startswith("wpe."):
        return "position_embedding", 0
    if name.startswith("blocks."):
        parts = name.split("."); bi = int(parts[1]); base = 1 + bi * 6
        if ".attn.key." in name: return "attention_key", base
        if ".attn.query." in name: return "attention_query", base
        if ".attn.value." in name: return "attention_value", base
        if ".attn.proj." in name: return "attention_projection", base + 1
        if ".mlp.0." in name: return "mlp_input", base + 2
        if ".mlp.2." in name: return "mlp_output", base + 3
        if ".ln_" in name: return "block_layernorm", base + 4
        return "block_other", base + 5
    max_depth = 1 + len(getattr(model, "blocks", [])) * 6
    if name.startswith("ln_f."):
        return "final_layernorm", max_depth
    if name.startswith("lm_head."):
        return "lm_head", max_depth + 1
    return "other", max_depth + 1


def _decay_for(name: str, p: nn.Parameter) -> bool:
    return p.requires_grad and p.ndim >= 2 and not name.endswith(".bias") and "ln_" not in name and not name.startswith("ln_f.")


def build_param_groups(model: nn.Module, base_lr: float, weight_decay: float, cfg: TrainConfig, *, include_names: set[str] | None = None) -> tuple[list[dict[str, Any]], float]:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and (include_names is None or n in include_names)]
    depths = {n: parameter_role_depth(n, model)[1] for n, _ in named}
    max_depth = max(depths.values(), default=1)
    gamma = cfg.llrd_gamma if cfg.llrd_gamma is not None else cfg.llrd_min_multiplier ** (1.0 / max(max_depth, 1))
    groups = []
    multipliers = cfg.matrix_lr_multipliers or {}
    for n, p in named:
        role, depth = parameter_role_depth(n, model)
        if cfg.layer_lr == "flat":
            layer_mult = 1.0
        elif cfg.layer_lr == "llrd":
            layer_mult = gamma ** (max_depth - depth)
        elif cfg.layer_lr == "manual":
            layer_mult = MANUAL_LAYER_LR_MULTIPLIERS.get(role, MANUAL_LAYER_LR_MULTIPLIERS["other"])
        else:
            raise ValueError(f"unknown layer_lr {cfg.layer_lr}")
        matrix_mult = float(multipliers.get(role, 1.0))
        peak_lr = base_lr * layer_mult * matrix_mult
        groups.append({"params": [p], "lr": peak_lr, "initial_lr": peak_lr, "peak_lr": peak_lr, "minimum_lr": peak_lr * cfg.min_lr_ratio, "weight_decay": weight_decay if _decay_for(n, p) else 0.0, "group_name": n, "parameter_name": n, "role": role, "depth": depth, "layer_lr_multiplier": layer_mult, "matrix_specific_multiplier": matrix_mult, "parameter_count": p.numel()})
    return groups, gamma


def muon_parameter_names(model: nn.Module) -> set[str]:
    hidden_matrix_roles = {
        "attention_key",
        "attention_query",
        "attention_value",
        "attention_projection",
        "mlp_input",
        "mlp_output",
    }
    out = set()
    for n, p in model.named_parameters():
        if not p.requires_grad or p.ndim != 2:
            continue
        role, _ = parameter_role_depth(n, model)
        if role in hidden_matrix_roles:
            out.add(n)
    return out


def build_optimizer_bundle(model: nn.Module, cfg: TrainConfig, base_optimizer: str) -> tuple[OptimizerBundle, float]:
    base_optimizer = "stableadamw" if base_optimizer == "stable_adamw" else base_optimizer
    if base_optimizer == "adamw":
        groups, gamma = build_param_groups(model, cfg.learning_rate, cfg.weight_decay, cfg)
        opt = torch.optim.AdamW(groups, betas=cfg.betas, eps=cfg.epsilon)
        return OptimizerBundle("adamw", [opt], [("adamw", opt)]), gamma
    if base_optimizer == "stableadamw":
        try:
            from optimi import StableAdamW
        except Exception as e:
            raise RuntimeError("StableAdamW requires installing the 'optimi' package") from e
        groups, gamma = build_param_groups(model, cfg.stable_learning_rate, cfg.weight_decay, cfg)
        opt = StableAdamW(groups, lr=cfg.stable_learning_rate, betas=cfg.stable_betas, eps=cfg.stable_epsilon, weight_decay=0.0, triton=cfg.stable_triton)
        return OptimizerBundle("stableadamw", [opt], [("stableadamw", opt)]), gamma
    if base_optimizer == "muon":
        mnames = muon_parameter_names(model); anames = {n for n, p in model.named_parameters() if p.requires_grad} - mnames
        mg, gamma = build_param_groups(model, cfg.muon_learning_rate, cfg.weight_decay, cfg, include_names=mnames)
        ag, gamma2 = build_param_groups(model, cfg.learning_rate, cfg.weight_decay, cfg, include_names=anames)
        mu = Muon(mg, lr=cfg.muon_learning_rate, momentum=cfg.muon_momentum, newton_schulz_steps=cfg.newton_schulz_steps, weight_decay=0.0)
        aux = torch.optim.AdamW(ag, betas=cfg.betas, eps=cfg.epsilon)
        return OptimizerBundle("muon", [mu, aux], [("muon", mu), ("muon_aux_adamw", aux)]), gamma or gamma2
    raise ValueError(f"unknown optimizer {base_optimizer}")


SCHEDULER_IMPLEMENTATION = "nanogpt_linear_warmup_cosine_v1"


def resolve_lr_decay_steps(total_optimizer_steps: int, lr_decay_steps: int | None) -> int:
    return int(lr_decay_steps) if lr_decay_steps is not None else int(total_optimizer_steps)


def resolve_warmup_steps(total: int, warmup_ratio: float, warmup_steps: int | None, lr_decay_steps: int | None = None) -> int:
    decay = resolve_lr_decay_steps(total, lr_decay_steps)
    warmup = int(warmup_steps) if warmup_steps is not None else round(warmup_ratio * decay)
    warmup = 0 if decay <= 1 else min(decay - 1, max(0, warmup))
    if not warmup < decay:
        raise ValueError("resolved_warmup_steps must be less than resolved_lr_decay_steps")
    return warmup


def nanogpt_cosine_lr(step0: int, *, peak_lr: float, warmup_steps: int, lr_decay_steps: int, min_lr_ratio: float) -> float:
    min_lr = peak_lr * min_lr_ratio
    if lr_decay_steps <= 1:
        return peak_lr
    if warmup_steps > 0 and step0 < warmup_steps:
        return peak_lr * (step0 + 1) / (warmup_steps + 1)
    decay_end_step0 = lr_decay_steps - 1
    if step0 >= decay_end_step0:
        return min_lr
    decay_ratio = (step0 - warmup_steps) / (decay_end_step0 - warmup_steps)
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coefficient * (peak_lr - min_lr)


def schedule_factor(step0: int, total: int, warmup: int, schedule: str, min_lr_ratio: float) -> float:
    if schedule == "constant":
        return 1.0
    if schedule == "warmup_linear":
        if total <= 1:
            return 1.0
        if warmup > 0 and step0 < warmup:
            return (step0 + 1) / (warmup + 1)
        end = total - 1
        if step0 >= end:
            return min_lr_ratio
        progress = max(0.0, min(1.0, (step0 - warmup) / max(1, end - warmup)))
        return min_lr_ratio + (1 - progress) * (1 - min_lr_ratio)
    if schedule == "warmup_cosine":
        return nanogpt_cosine_lr(step0, peak_lr=1.0, warmup_steps=warmup, lr_decay_steps=total, min_lr_ratio=min_lr_ratio)
    raise ValueError(f"unknown lr_schedule {schedule}")


def optimizer_group_signature(bundle: OptimizerBundle) -> tuple[dict[str, Any], ...]:
    signature = []
    for opt_name, opt in bundle.scheduled_optimizers:
        for group in opt.param_groups:
            betas = group.get("betas")
            if betas is None and "beta1" in group and "beta2" in group:
                betas = (group["beta1"], group["beta2"])
            if betas is not None:
                betas = tuple(float(x) for x in betas)
            epsilon = group.get("eps", group.get("epsilon"))
            signature.append({
                "parameter_name": group.get("parameter_name", group.get("group_name", "")),
                "optimizer_name": opt_name,
                "role": group.get("role", ""),
                "peak_lr": float(group.get("peak_lr", group.get("initial_lr", group.get("lr", 0.0)))),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "betas": betas,
                "epsilon": None if epsilon is None else float(epsilon),
            })
    return tuple(signature)


def apply_lr_schedule(bundle: OptimizerBundle, step0: int, total: int, warmup: int, cfg: TrainConfig) -> list[dict[str, Any]]:
    resolved_decay = resolve_lr_decay_steps(total, cfg.lr_decay_steps)
    factor = schedule_factor(step0, resolved_decay, warmup, cfg.lr_schedule, cfg.min_lr_ratio)
    rows=[]
    for opt_name,opt in bundle.scheduled_optimizers:
        for i,g in enumerate(opt.param_groups):
            peak=float(g.get("peak_lr", g.get("initial_lr", g["lr"])))
            lr=peak*factor; g["lr"]=lr
            rows.append({"optimizer_step": step0+1, "optimizer_name": opt_name, "group_index": i, "group_name": g.get("group_name", str(i)), "parameter_name": g.get("parameter_name", ""), "role": g.get("role", ""), "depth": g.get("depth", -1), "current_lr": lr, "peak_lr": peak, "minimum_lr": peak*cfg.min_lr_ratio, "layer_lr_multiplier": g.get("layer_lr_multiplier", 1.0), "matrix_specific_multiplier": g.get("matrix_specific_multiplier", 1.0), "normalized_time_factor": factor, "time_schedule_factor": factor, "weight_decay": g.get("weight_decay", 0.0), "parameter_count": g.get("parameter_count", 0), "resolved_warmup_steps": warmup, "resolved_lr_decay_steps": resolved_decay, "min_lr_ratio": cfg.min_lr_ratio, "scheduler_implementation": SCHEDULER_IMPLEMENTATION})
    return rows
