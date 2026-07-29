from __future__ import annotations

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(
    gradient: torch.Tensor, steps: int = 5
) -> torch.Tensor:
    if gradient.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization requires a matrix")
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
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "nesterov": nesterov,
                "weight_decay": weight_decay,
            },
        )

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
                    "momentum_buffer", torch.zeros_like(parameter)
                )
                buffer.mul_(group["momentum"]).add_(parameter.grad)
                update_source = (
                    parameter.grad.add(buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                update = zeropower_via_newtonschulz5(update_source)
                update.mul_(max(1, parameter.shape[0] / parameter.shape[1]) ** 0.5)
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


def make_optimizers(model, cfg):
    training = cfg["training"]
    name = training["optimizer"].lower()
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    decay = [parameter for _, parameter in named_parameters if parameter.ndim >= 2]
    no_decay = [parameter for _, parameter in named_parameters if parameter.ndim < 2]

    if name == "adamw":
        return [
            torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": training["weight_decay"]},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=training["learning_rate"],
                betas=(training["beta1"], training["beta2"]),
                eps=training["epsilon"],
            )
        ]
    if name != "muon":
        raise ValueError(f"unsupported optimizer: {name}")

    muon_parameters = []
    adam_parameters = []
    for parameter_name, parameter in named_parameters:
        if parameter.ndim == 2 and "embedding" not in parameter_name:
            muon_parameters.append(parameter)
        else:
            adam_parameters.append(parameter)
    return [
        Muon(
            muon_parameters,
            lr=training["muon_learning_rate"],
            momentum=training["muon_momentum"],
            nesterov=training["muon_nesterov"],
            weight_decay=training["weight_decay"],
        ),
        torch.optim.AdamW(
            adam_parameters,
            lr=training["muon_aux_adamw_learning_rate"],
            betas=(training["beta1"], training["beta2"]),
            eps=training["epsilon"],
            weight_decay=0.0,
        ),
    ]
