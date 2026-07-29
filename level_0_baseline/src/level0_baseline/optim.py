from __future__ import annotations

import inspect
from typing import Iterable

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(
    gradient: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    if gradient.ndim != 2:
        raise ValueError("Newton-Schulz zero-power update requires a matrix")
    x = gradient.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(gradient.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("Muon received a non-matrix parameter")
                buffer = self.state[parameter].setdefault(
                    "momentum_buffer",
                    torch.zeros_like(parameter),
                )
                buffer.mul_(group["momentum"]).add_(parameter.grad)
                update_source = (
                    parameter.grad.add(buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                update = zeropower_via_newtonschulz5(update_source)
                update.mul_(max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5)
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


def _adamw(
    groups: list[dict],
    *,
    learning_rate: float,
    betas: tuple[float, float],
    epsilon: float,
    device_type: str,
) -> torch.optim.AdamW:
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra = {"fused": True} if use_fused else {}
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=betas,
        eps=epsilon,
        **extra,
    )


def make_optimizers(
    model: torch.nn.Module,
    cfg: dict,
    *,
    device_type: str = "cpu",
) -> list[torch.optim.Optimizer]:
    training = cfg["training"]
    optimizer_name = training["optimizer"].lower()
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    decay = [parameter for parameter in parameters.values() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in parameters.values() if parameter.ndim < 2]
    betas = (training["beta1"], training["beta2"])
    epsilon = float(training.get("epsilon", 1e-8))

    if optimizer_name == "adamw":
        groups = [
            {"params": decay, "weight_decay": training["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return [
            _adamw(
                groups,
                learning_rate=training["learning_rate"],
                betas=betas,
                epsilon=epsilon,
                device_type=device_type,
            )
        ]

    if optimizer_name != "muon":
        raise ValueError(f"unsupported optimizer: {optimizer_name}")

    muon_parameters: list[torch.nn.Parameter] = []
    auxiliary_parameters: list[torch.nn.Parameter] = []
    for name, parameter in parameters.items():
        if (
            parameter.ndim == 2
            and "token_embedding" not in name
            and "position_embedding" not in name
            and "lm_head" not in name
        ):
            muon_parameters.append(parameter)
        else:
            auxiliary_parameters.append(parameter)

    return [
        Muon(
            muon_parameters,
            lr=training["muon_learning_rate"],
            momentum=training["muon_momentum"],
            nesterov=training["muon_nesterov"],
            weight_decay=training["weight_decay"],
        ),
        _adamw(
            [{"params": auxiliary_parameters, "weight_decay": 0.0}],
            learning_rate=training["muon_aux_adamw_learning_rate"],
            betas=betas,
            epsilon=epsilon,
            device_type=device_type,
        ),
    ]
