from __future__ import annotations

import torch


def make_optimizer(model, cfg):
    training = cfg["training"]
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    decay = [parameter for _, parameter in named_parameters if parameter.ndim >= 2]
    no_decay = [parameter for _, parameter in named_parameters if parameter.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": training["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=training["learning_rate"],
        betas=(training["beta1"], training["beta2"]),
        eps=training["epsilon"],
    )
