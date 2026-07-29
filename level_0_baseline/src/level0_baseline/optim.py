from __future__ import annotations
import torch

@torch.no_grad()
def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert g.ndim == 2
    x = g.float()
    if x.shape[0] > x.shape[1]:
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = x @ x.T
        x = a * x + (b * A + c * (A @ A)) @ x
    if g.shape[0] > g.shape[1]:
        x = x.T
    return x.to(g.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon received a non-matrix parameter")
                buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(g)
                update.mul_(max(1, p.shape[0] / p.shape[1]) ** 0.5)
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss

def make_optimizers(model, cfg):
    t = cfg["training"]
    name = t["optimizer"].lower()
    decay = [p for _, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for _, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    if name == "adamw":
        return [torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]}, {"params": nodecay, "weight_decay": 0.0}], lr=t["learning_rate"], betas=(t["beta1"], t["beta2"]))]
    if name != "muon":
        raise ValueError(f"unsupported optimizer: {name}")
    muon, adam = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embedding" not in n and "lm_head" not in n:
            muon.append(p)
        else:
            adam.append(p)
    return [Muon(muon, lr=t["muon_learning_rate"], momentum=t["muon_momentum"], nesterov=t["muon_nesterov"], weight_decay=t["weight_decay"]), torch.optim.AdamW(adam, lr=t["muon_aux_adamw_learning_rate"], betas=(t["beta1"], t["beta2"]), weight_decay=0.0)]
