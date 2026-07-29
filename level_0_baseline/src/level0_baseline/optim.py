from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

import torch

_BLOCK_PATTERN = re.compile(r"^blocks\.(\d+)\.")


@torch.no_grad()
def zeropower_via_newtonschulz5(
    gradient: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    if gradient.ndim != 2:
        raise ValueError("Muon zero-power update requires a matrix")
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
        params: Iterable[torch.nn.Parameter] | list[dict[str, Any]],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
    ) -> None:
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
                momentum_buffer = self.state[parameter].setdefault(
                    "momentum_buffer",
                    torch.zeros_like(parameter),
                )
                momentum_buffer.mul_(group["momentum"]).add_(parameter.grad)
                update_input = (
                    parameter.grad.add(
                        momentum_buffer,
                        alpha=group["momentum"],
                    )
                    if group["nesterov"]
                    else momentum_buffer
                )
                update = zeropower_via_newtonschulz5(update_input)
                update.mul_(
                    max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
                )
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


def _layer_multiplier(name: str, n_layer: int, decay: float) -> float:
    if decay == 1.0:
        return 1.0
    match = _BLOCK_PATTERN.match(name)
    if match:
        block_index = int(match.group(1))
        return decay ** max(n_layer - 1 - block_index, 0)
    if name.startswith(("token_embedding", "position_embedding")):
        return decay**n_layer
    return 1.0


def _parameter_records(model, training: dict[str, Any]):
    n_layer = int(model.cfg.n_layer)
    layer_decay = float(training.get("layer_lr_decay", 1.0))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        yield {
            "name": name,
            "parameter": parameter,
            "decay": parameter.ndim >= 2,
            "lr_multiplier": _layer_multiplier(name, n_layer, layer_decay),
        }


def _adamw_groups(model, training: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[bool, float], list[torch.nn.Parameter]] = defaultdict(list)
    names: dict[tuple[bool, float], list[str]] = defaultdict(list)
    for record in _parameter_records(model, training):
        key = (bool(record["decay"]), float(record["lr_multiplier"]))
        grouped[key].append(record["parameter"])
        names[key].append(str(record["name"]))

    base_lr = float(training["learning_rate"])
    groups: list[dict[str, Any]] = []
    for (use_decay, multiplier), parameters in sorted(
        grouped.items(),
        key=lambda item: (item[0][1], item[0][0]),
    ):
        groups.append(
            {
                "params": parameters,
                "lr": base_lr * multiplier,
                "initial_lr": base_lr * multiplier,
                "lr_multiplier": multiplier,
                "weight_decay": (
                    float(training["weight_decay"]) if use_decay else 0.0
                ),
                "group_name": (
                    f"{'decay' if use_decay else 'no_decay'}_lr_{multiplier:.6f}"
                ),
                "parameter_names": names[(use_decay, multiplier)],
            }
        )
    return groups


def make_optimizers(model, config: dict[str, Any]) -> list[torch.optim.Optimizer]:
    training = config["training"]
    optimizer_name = str(training["optimizer"]).lower()
    if optimizer_name == "adamw":
        return [
            torch.optim.AdamW(
                _adamw_groups(model, training),
                lr=float(training["learning_rate"]),
                betas=(float(training["beta1"]), float(training["beta2"])),
                eps=float(training["epsilon"]),
            )
        ]
    if optimizer_name != "muon":
        raise ValueError(f"unsupported optimizer: {optimizer_name}")

    muon_grouped: dict[float, list[torch.nn.Parameter]] = defaultdict(list)
    auxiliary_grouped: dict[
        tuple[bool, float], list[torch.nn.Parameter]
    ] = defaultdict(list)
    for record in _parameter_records(model, training):
        name = str(record["name"])
        parameter = record["parameter"]
        multiplier = float(record["lr_multiplier"])
        use_muon = parameter.ndim == 2 and not name.startswith(
            ("token_embedding", "position_embedding", "lm_head")
        )
        if use_muon:
            muon_grouped[multiplier].append(parameter)
        else:
            auxiliary_grouped[(bool(record["decay"]), multiplier)].append(parameter)

    muon_base_lr = float(training["muon_learning_rate"])
    muon_groups = [
        {
            "params": parameters,
            "lr": muon_base_lr * multiplier,
            "initial_lr": muon_base_lr * multiplier,
            "lr_multiplier": multiplier,
            "weight_decay": float(training["weight_decay"]),
            "group_name": f"muon_lr_{multiplier:.6f}",
        }
        for multiplier, parameters in sorted(muon_grouped.items())
    ]
    auxiliary_base_lr = float(training["muon_aux_adamw_learning_rate"])
    auxiliary_groups = [
        {
            "params": parameters,
            "lr": auxiliary_base_lr * multiplier,
            "initial_lr": auxiliary_base_lr * multiplier,
            "lr_multiplier": multiplier,
            "weight_decay": (
                float(training["weight_decay"]) if use_decay else 0.0
            ),
            "group_name": (
                f"aux_{'decay' if use_decay else 'no_decay'}_lr_{multiplier:.6f}"
            ),
        }
        for (use_decay, multiplier), parameters in sorted(
            auxiliary_grouped.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )
    ]
    return [
        Muon(
            muon_groups,
            lr=muon_base_lr,
            momentum=float(training["muon_momentum"]),
            nesterov=bool(training["muon_nesterov"]),
            weight_decay=float(training["weight_decay"]),
        ),
        torch.optim.AdamW(
            auxiliary_groups,
            lr=auxiliary_base_lr,
            betas=(float(training["beta1"]), float(training["beta2"])),
            eps=float(training["epsilon"]),
        ),
    ]
