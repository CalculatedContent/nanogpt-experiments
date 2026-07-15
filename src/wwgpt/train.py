from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from wwgpt.config import ExperimentConfig, ModelConfig, TrainConfig, WWPGDConfig
from wwgpt.data import NonRepeatingTokenReader, prepare_local_text
from wwgpt.model import GPT
from wwgpt.utils import environment, sha256_bytes, unique_dir, write_json
from wwgpt.ww import apply_wwpgd, matrix_modules, spectral_summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _metrics(loss: float, logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
    top1 = float((pred[..., 0] == y).float().mean())
    top5 = float((pred == y.unsqueeze(-1)).any(dim=-1).float().mean())
    return {"loss": loss, "perplexity": float(math.exp(min(loss, 20))), "bits_per_token": loss / math.log(2), "top1_accuracy": top1, "top5_accuracy": top5, "token_error": 1 - top1}


def run_single(run_parent: Path, optimizer_name: str, seed: int, cfg: ExperimentConfig, train_tokens: list[int], val_tokens: list[int], pair_id: str, max_steps: int | None = None, init_state: dict[str, torch.Tensor] | None = None) -> Path:
    torch.manual_seed(seed)
    run_dir = unique_dir(run_parent / optimizer_name, "run")
    ckpt = run_dir / "checkpoints"; ckpt.mkdir()
    model_cfg = ModelConfig(**{**asdict(cfg.model), "vocab_size": max(train_tokens + val_tokens) + 1})
    model = GPT(model_cfg)
    if init_state is not None:
        model.load_state_dict(init_state)
    init_hash = sha256_bytes(b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values()))
    (run_dir / "initialization_hash.txt").write_text(init_hash)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, betas=cfg.train.betas, eps=cfg.train.epsilon, weight_decay=cfg.train.weight_decay)
    steps = max_steps or cfg.train.max_steps or 3
    reader = NonRepeatingTokenReader(train_tokens, model_cfg.block_size)
    val_reader = NonRepeatingTokenReader(val_tokens + train_tokens, model_cfg.block_size)
    metric_rows=[]; spectral_rows=[]; proj_rows=[]
    write_json(run_dir / "environment.json", environment())
    write_json(run_dir / "manifest.json", {"optimizer": optimizer_name, "seed": seed, "pair_id": pair_id, "smoke_test": True, "valid_for_science": False, "parameter_report": model.report_dict()})
    write_json(run_dir / "data_manifest.json", {"dataset": "local_text", "corpus_hash": sha256_bytes(bytes([x % 256 for x in train_tokens]))})
    write_json(run_dir / "tokenizer_manifest.json", {"tokenizer": "char-smoke", "vocab_size": model_cfg.vocab_size})
    (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(asdict(cfg)))))
    write_json(run_dir / "config.json", json.loads(json.dumps(asdict(cfg))))
    torch.save(model.state_dict(), ckpt / f"initial_step_000000_{seed}.pt")
    start=time.perf_counter(); last_loss=0.0
    for step in range(1, steps + 1):
        xb, yb = reader.next_batch(cfg.train.batch_size)
        x = torch.tensor(xb); y = torch.tensor(yb)
        _, loss = model(x, y); assert loss is not None
        opt.zero_grad(); loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip); opt.step()
        proj_time=0.0
        if optimizer_name == "adamw_wwpgd":
            pstart=time.perf_counter(); proj_rows.extend(apply_wwpgd(model, cfg.wwpgd.target_alpha, cfg.wwpgd.strength, step)); proj_time=time.perf_counter()-pstart
        with torch.no_grad():
            vx, vy = val_reader.next_batch(cfg.train.batch_size)
            vlogits, vloss = model(torch.tensor(vx), torch.tensor(vy)); assert vloss is not None
            tlogits, tloss = model(x, y); assert tloss is not None
        tm=_metrics(float(tloss), tlogits, y); vm=_metrics(float(vloss), vlogits, torch.tensor(vy))
        elapsed=time.perf_counter()-start; last_loss=float(vloss)
        metric_rows.append({"step": step, "tokens_processed": step*cfg.train.batch_size*model_cfg.block_size, "elapsed_time": elapsed, "learning_rate": cfg.train.learning_rate, "gradient_norm": float(grad), "train_minibatch_loss": float(loss), "train_loss": tm["loss"], "val_loss": vm["loss"], "train_perplexity": tm["perplexity"], "val_perplexity": vm["perplexity"], "train_bits_per_token": tm["bits_per_token"], "val_bits_per_token": vm["bits_per_token"], "train_top1_accuracy": tm["top1_accuracy"], "val_top1_accuracy": vm["top1_accuracy"], "train_top5_accuracy": tm["top5_accuracy"], "val_top5_accuracy": vm["top5_accuracy"], "train_token_error": tm["token_error"], "val_token_error": vm["token_error"], "generalization_gap": vm["loss"]-tm["loss"], "tokens_per_second": (step*cfg.train.batch_size*model_cfg.block_size)/max(elapsed,1e-9), "examples_per_second": (step*cfg.train.batch_size)/max(elapsed,1e-9), "weightwatcher_overhead": 0.0, "projection_overhead": proj_time, "peak_memory": 0.0})
        for lid,(name,w) in enumerate(matrix_modules(model)):
            rec=asdict(spectral_summary(name,w)); rec.update({"layer_id": lid, "step": step, "optimizer": optimizer_name, "seed": seed}); spectral_rows.append(rec)
        torch.save({"model": model.state_dict(), "step": step}, ckpt / f"latest_step_{step:06d}_{seed}.pt")
    torch.save(model.state_dict(), ckpt / f"final_step_{steps:06d}_{seed}.pt")
    torch.save(model.state_dict(), ckpt / f"best_val_step_{steps:06d}_{seed}.pt")
    _write_csv(run_dir / "metrics.csv", metric_rows); _write_csv(run_dir / "spectral.csv", spectral_rows)
    if optimizer_name == "adamw_wwpgd": _write_csv(run_dir / "wwpgd_projection.csv", proj_rows)
    (run_dir / "events.jsonl").write_text(json.dumps({"event":"complete"})+"\n")
    write_json(run_dir / "run_complete.json", {"step": steps, "final_val_loss": last_loss})
    return run_dir


def smoke(root: Path, steps: int = 3, seeds: list[int] | None = None) -> Path:
    smoke_dir=unique_dir(root, "wwgpt_smoke_invalid")
    text=("WeightWatcher PGD smoke corpus. This is not Tiny Shakespeare and is invalid for science. "*400).split(".")
    cfg=ExperimentConfig(model=ModelConfig(n_layer=1,n_head=1,n_embd=32,block_size=16,vocab_size=128), train=TrainConfig(batch_size=2, max_steps=steps, eval_interval=1), wwpgd=WWPGDConfig(enabled=True, strength=0.01))
    data=prepare_local_text(smoke_dir / "data", [t+"." for t in text], min_train_tokens=steps*cfg.train.batch_size*cfg.model.block_size*2+1)
    pair_parent=smoke_dir / "level_00" / "pair_smoke"
    torch.manual_seed(seeds[0] if seeds else 1337); init=GPT(ModelConfig(**{**asdict(cfg.model), "vocab_size": data.vocab_size})).state_dict()
    for opt in ["adamw", "adamw_wwpgd"]:
        run_single(pair_parent, opt, seeds[0] if seeds else 1337, cfg, data.train, data.val, "pair_smoke", steps, init)
    return smoke_dir
