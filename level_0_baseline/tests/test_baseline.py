import torch
from level0_baseline.model import GPT, GPTConfig
from level0_baseline.optim import make_optimizers

def cfg(opt):
    return {"training": {"optimizer": opt, "learning_rate": 0.001, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "muon_momentum": 0.95, "muon_nesterov": True, "muon_learning_rate": 0.02, "muon_aux_adamw_learning_rate": 0.001}}

def test_forward_and_accuracy_shape():
    model = GPT(GPTConfig(block_size=8, n_embd=16, n_head=1, n_layer=1))
    x = torch.randint(0, 256, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 256)
    assert torch.isfinite(loss)

def test_adamw_step():
    model = GPT(GPTConfig(block_size=8, n_embd=16))
    optimizers = make_optimizers(model, cfg("adamw"))
    _, loss = model(torch.randint(0, 256, (2, 8)), torch.randint(0, 256, (2, 8)))
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()

def test_muon_partition_and_step():
    model = GPT(GPTConfig(block_size=8, n_embd=16))
    optimizers = make_optimizers(model, cfg("muon"))
    assert len(optimizers) == 2
    _, loss = model(torch.randint(0, 256, (2, 8)), torch.randint(0, 256, (2, 8)))
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
