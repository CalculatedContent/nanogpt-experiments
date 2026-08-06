from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch


@dataclass
class OptimizerHandle:
    role: str
    optimizer: torch.optim.Optimizer
    peak_lr: float
    min_lr: float

    def set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


@torch.no_grad()
def zeropower_via_newtonschulz5(
    gradient: torch.Tensor, steps: int = 5
) -> torch.Tensor:
    """Muon quintic Newton-Schulz orthogonalization, evaluated in float32.

    The float32 path is deliberate for Apple MPS reliability. The returned update
    is converted back to the gradient dtype.
    """
    if gradient.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization requires a matrix")
    x = gradient.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(gradient.dtype)


class Muon(torch.optim.Optimizer):
    """Single-device Muon for hidden 2D weight matrices."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        newton_schulz_steps: int = 5,
    ) -> None:
        params = list(params)
        if not params:
            raise ValueError("Muon requires at least one parameter")
        if any(parameter.ndim != 2 for parameter in params):
            raise ValueError("Muon accepts only 2D parameters")
        super().__init__(
            params,
            {
                "lr": float(lr),
                "momentum": float(momentum),
                "nesterov": bool(nesterov),
                "weight_decay": float(weight_decay),
                "newton_schulz_steps": int(newton_schulz_steps),
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta = float(group["momentum"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("Muon received a non-matrix parameter")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                momentum = state["momentum_buffer"]
                momentum.lerp_(parameter.grad, 1.0 - beta)
                update_source = (
                    torch.lerp(parameter.grad, momentum, beta)
                    if group["nesterov"]
                    else momentum
                )
                update = zeropower_via_newtonschulz5(
                    update_source, int(group["newton_schulz_steps"])
                )
                update.mul_(
                    math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
                )
                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


def cosine_learning_rate(
    update_index: int,
    *,
    max_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if warmup_steps < 0 or warmup_steps >= max_steps:
        raise ValueError("warmup_steps must be in [0, max_steps)")
    if update_index < 0:
        raise ValueError("update_index must be nonnegative")
    if warmup_steps and update_index < warmup_steps:
        return peak_lr * (update_index + 1) / warmup_steps
    progress = (update_index - warmup_steps) / max(
        1, max_steps - warmup_steps - 1
    )
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (peak_lr - min_lr)


def set_learning_rates(
    handles: list[OptimizerHandle],
    *,
    update_index: int,
    max_steps: int,
    warmup_steps: int,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for handle in handles:
        value = cosine_learning_rate(
            update_index,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            peak_lr=handle.peak_lr,
            min_lr=handle.min_lr,
        )
        handle.set_lr(value)
        values[handle.role] = value
    return values


def _named_trainable_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def _decay_groups(
    named_parameters: list[tuple[str, torch.nn.Parameter]], weight_decay: float
):
    decay = [parameter for _, parameter in named_parameters if parameter.ndim >= 2]
    no_decay = [parameter for _, parameter in named_parameters if parameter.ndim < 2]
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer_handles(model, profile: dict) -> list[OptimizerHandle]:
    named_parameters = _named_trainable_parameters(model)
    family = str(profile["family"])

    if family == "sgd":
        optimizer = torch.optim.SGD(
            _decay_groups(named_parameters, float(profile["weight_decay"])),
            lr=float(profile["learning_rate"]),
            momentum=float(profile["momentum"]),
            dampening=float(profile["dampening"]),
            nesterov=bool(profile["nesterov"]),
        )
        return [
            OptimizerHandle(
                "primary",
                optimizer,
                float(profile["learning_rate"]),
                float(profile["min_lr"]),
            )
        ]

    if family == "adamw":
        optimizer = torch.optim.AdamW(
            _decay_groups(named_parameters, float(profile["weight_decay"])),
            lr=float(profile["learning_rate"]),
            betas=(float(profile["beta1"]), float(profile["beta2"])),
            eps=float(profile["epsilon"]),
        )
        return [
            OptimizerHandle(
                "primary",
                optimizer,
                float(profile["learning_rate"]),
                float(profile["min_lr"]),
            )
        ]

    if family != "muon":
        raise ValueError(f"unsupported optimizer family: {family}")

    hidden = [
        parameter
        for name, parameter in named_parameters
        if parameter.ndim == 2 and name.startswith("blocks.")
    ]
    hidden_ids = {id(parameter) for parameter in hidden}
    auxiliary = [
        parameter for _, parameter in named_parameters if id(parameter) not in hidden_ids
    ]
    if not hidden or not auxiliary:
        raise ValueError("Muon partition must contain hidden and auxiliary parameters")

    muon = Muon(
        hidden,
        lr=float(profile["hidden_learning_rate"]),
        momentum=float(profile["momentum"]),
        nesterov=bool(profile["nesterov"]),
        weight_decay=float(profile["hidden_weight_decay"]),
        newton_schulz_steps=int(profile["newton_schulz_steps"]),
    )
    auxiliary_adamw = torch.optim.AdamW(
        auxiliary,
        lr=float(profile["auxiliary_learning_rate"]),
        betas=(float(profile["beta1"]), float(profile["beta2"])),
        eps=float(profile["epsilon"]),
        weight_decay=float(profile["auxiliary_weight_decay"]),
    )
    return [
        OptimizerHandle(
            "primary",
            muon,
            float(profile["hidden_learning_rate"]),
            float(profile["hidden_min_lr"]),
        ),
        OptimizerHandle(
            "auxiliary",
            auxiliary_adamw,
            float(profile["auxiliary_learning_rate"]),
            float(profile["auxiliary_min_lr"]),
        ),
    ]


def zero_grad(handles: list[OptimizerHandle]) -> None:
    for handle in handles:
        handle.optimizer.zero_grad(set_to_none=True)


def optimizer_step(handles: list[OptimizerHandle]) -> None:
    for handle in handles:
        handle.optimizer.step()
