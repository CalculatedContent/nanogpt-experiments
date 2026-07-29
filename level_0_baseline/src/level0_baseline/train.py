from __future__ import annotations
import argparse, csv, json, math, platform, random, time
from pathlib import Path
import numpy as np
import torch
from .config import load_config, roots
from .model import GPT, GPTConfig
from .optim import make_optimizers

def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def lr_at(step, t):
    if step < t["warmup_steps"]:
        return t["learning_rate"] * (step + 1) / max(1, t["warmup_steps"])
    if step >= t["max_steps"]:
        return t["min_lr"]
    ratio = (step - t["warmup_steps"]) / max(1, t["max_steps"] - t["warmup_steps"])
    return t["min_lr"] + 0.5 * (1 + math.cos(math.pi * ratio)) * (t["learning_rate"] - t["min_lr"])

def batch(data, batch_size, block_size, device, generator):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(np.array(data[i:i+block_size], dtype=np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.array(data[i+1:i+1+block_size], dtype=np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate(model, data, batch_size, block_size, n_batches, device, generator):
    model.eval()
    losses, correct, total = [], 0, 0
    for _ in range(n_batches):
        x, y = batch(data, batch_size, block_size, device, generator)
        logits, loss = model(x, y)
        losses.append(loss.item())
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    model.train()
    loss = float(np.mean(losses))
    return loss, math.exp(min(20, loss)), correct / total

def weightwatch(model, out, step, randomize):
    try:
        import weightwatcher as ww
    except ImportError:
        return
    df = ww.WeightWatcher(model=model).analyze(randomize=randomize)
    df.insert(0, "step", step)
    df.to_csv(out / f"weightwatcher_step_{step:07d}.csv", index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/level0.yaml")
    p.add_argument("--data-root")
    p.add_argument("--results-root")
    p.add_argument("--optimizer", choices=["adamw", "muon"])
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    cfg = load_config(a.config)
    t = cfg["training"]
    if a.optimizer:
        t["optimizer"] = a.optimizer
    if a.seed is not None:
        t["seed"] = a.seed
    resolved = roots()
    data_root = Path(a.data_root or resolved["data"])
    base = Path(a.results_root or resolved["results"])
    run = base / f"{t['optimizer']}_seed_{t['seed']}"
    run.mkdir(parents=True, exist_ok=True)
    device = device_auto() if a.device == "auto" else torch.device(a.device)
    seed_all(t["seed"])
    generator = torch.Generator().manual_seed(t["seed"])
    arrays = {s: np.memmap(data_root / f"{s}.bin", dtype=np.uint8, mode="r") for s in ("train", "val", "test")}
    model = GPT(GPTConfig(**cfg["model"])).to(device)
    optimizers = make_optimizers(model, cfg)
    base_lrs = [group["lr"] for opt in optimizers for group in opt.param_groups]
    manifest = {"config": cfg, "device": str(device), "torch": torch.__version__, "platform": platform.platform(), "parameter_count": sum(p.numel() for p in model.parameters()), "data_root": str(data_root.resolve())}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    fields = ["step", "tokens_seen", "elapsed_sec", "learning_rate", "train_loss", "train_perplexity", "train_accuracy", "val_loss", "val_perplexity", "val_accuracy", "test_loss", "test_perplexity", "test_accuracy", "val_generalization_gap", "test_generalization_gap", "grad_norm", "weight_norm"]
    with open(run / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        start = time.time()
        last_grad_norm = float("nan")
        for step in range(t["max_steps"] + 1):
            lr = lr_at(step, t)
            scale = lr / t["learning_rate"]
            j = 0
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = base_lrs[j] * scale
                    j += 1
            if step % t["eval_interval"] == 0 or step == t["max_steps"]:
                values = {s: evaluate(model, arrays[s], t["batch_size"], cfg["model"]["block_size"], t["eval_batches"], device, generator) for s in ("train", "val", "test")}
                weight_norm = math.sqrt(sum(float((p.detach().float() ** 2).sum()) for p in model.parameters()))
                writer.writerow({"step": step, "tokens_seen": step * t["batch_size"] * cfg["model"]["block_size"] * t["grad_accum_steps"], "elapsed_sec": time.time() - start, "learning_rate": lr, "train_loss": values["train"][0], "train_perplexity": values["train"][1], "train_accuracy": values["train"][2], "val_loss": values["val"][0], "val_perplexity": values["val"][1], "val_accuracy": values["val"][2], "test_loss": values["test"][0], "test_perplexity": values["test"][1], "test_accuracy": values["test"][2], "val_generalization_gap": values["val"][0] - values["train"][0], "test_generalization_gap": values["test"][0] - values["train"][0], "grad_norm": last_grad_norm, "weight_norm": weight_norm})
                f.flush()
                print(step, {k: round(v[0], 4) for k, v in values.items()})
                if cfg["analysis"]["weightwatcher"] and step % cfg["analysis"]["weightwatcher_interval"] == 0:
                    weightwatch(model, run, step, cfg["analysis"]["randomize"])
            if step == t["max_steps"]:
                break
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            for _ in range(t["grad_accum_steps"]):
                x, y = batch(arrays["train"], t["batch_size"], cfg["model"]["block_size"], device, generator)
                _, loss = model(x, y)
                (loss / t["grad_accum_steps"]).backward()
            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]))
            for opt in optimizers:
                opt.step()
            if (step + 1) % t["checkpoint_interval"] == 0:
                torch.save({"model": model.state_dict(), "step": step + 1, "config": cfg}, run / f"checkpoint_{step+1:07d}.pt")
    torch.save({"model": model.state_dict(), "step": t["max_steps"], "config": cfg}, run / "checkpoint_final.pt")

if __name__ == "__main__":
    main()
