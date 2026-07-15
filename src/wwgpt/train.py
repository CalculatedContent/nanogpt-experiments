from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from wwgpt.config import DEFAULT_SEEDS, ExperimentConfig, ModelConfig, TrainConfig, WWPGDConfig, load_config
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
        tm=_metrics(float(tloss.detach()), tlogits, y); vm=_metrics(float(vloss.detach()), vlogits, torch.tensor(vy))
        elapsed=time.perf_counter()-start; last_loss=float(vloss.detach())
        metric_rows.append({"step": step, "tokens_processed": step*cfg.train.batch_size*model_cfg.block_size, "elapsed_time": elapsed, "learning_rate": cfg.train.learning_rate, "gradient_norm": float(grad.detach()), "train_minibatch_loss": float(loss.detach()), "train_loss": tm["loss"], "val_loss": vm["loss"], "train_perplexity": tm["perplexity"], "val_perplexity": vm["perplexity"], "train_bits_per_token": tm["bits_per_token"], "val_bits_per_token": vm["bits_per_token"], "train_top1_accuracy": tm["top1_accuracy"], "val_top1_accuracy": vm["top1_accuracy"], "train_top5_accuracy": tm["top5_accuracy"], "val_top5_accuracy": vm["top5_accuracy"], "train_token_error": tm["token_error"], "val_token_error": vm["token_error"], "generalization_gap": vm["loss"]-tm["loss"], "tokens_per_second": (step*cfg.train.batch_size*model_cfg.block_size)/max(elapsed,1e-9), "examples_per_second": (step*cfg.train.batch_size)/max(elapsed,1e-9), "weightwatcher_overhead": 0.0, "projection_overhead": proj_time, "peak_memory": 0.0})
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
    run_seeds = seeds or [1337]
    smoke_dir=unique_dir(root, "wwgpt_invalid_smoke")
    text=("WeightWatcher PGD smoke corpus. This is not Tiny Shakespeare and is invalid for science. "*400).split(".")
    cfg=ExperimentConfig(model=ModelConfig(n_layer=1,n_head=1,n_embd=32,block_size=16,vocab_size=128), train=TrainConfig(batch_size=2, max_steps=steps, eval_interval=1), wwpgd=WWPGDConfig(enabled=True, strength=0.01))
    data=prepare_local_text(smoke_dir / "data", [t+"." for t in text], min_train_tokens=steps*cfg.train.batch_size*cfg.model.block_size*2+1)
    pair_parent=smoke_dir / "level_00" / "pair_invalid"
    for seed in run_seeds:
        torch.manual_seed(seed); init=GPT(ModelConfig(**{**asdict(cfg.model), "vocab_size": data.vocab_size})).state_dict()
        for opt in ["adamw", "adamw_wwpgd"]:
            run_single(pair_parent, opt, seed, cfg, data.train, data.val, f"pair_invalid_seed_{seed}", steps, init)
    return smoke_dir


def select_device(override: str | None = None) -> torch.device:
    if override: return torch.device(override)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401
        return torch.device("xla")
    except Exception:
        return torch.device("cpu")


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    return sha256_bytes(b"".join(state[k].detach().cpu().numpy().tobytes() for k in sorted(state)))


def run_scientific_single(run_parent: Path, optimizer_name: str, seed: int, cfg: ExperimentConfig, data, pair_id: str, init_state: dict[str, torch.Tensor], init_hash: str, level: int, token_multiplier: int, device: str | None = None, ww_interval: int | None = None, eval_interval: int | None = None, checkpoint_interval: int | None = None) -> Path:
    torch.manual_seed(seed); run_dir=unique_dir(run_parent / optimizer_name, "run"); ckpt=run_dir/"checkpoints"; ckpt.mkdir()
    model=GPT(cfg.model).to(select_device(device)); model.load_state_dict(init_state)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, betas=cfg.train.betas, eps=cfg.train.epsilon, weight_decay=cfg.train.weight_decay)
    steps=int(data.data_manifest["optimizer_steps"]); tokens_per_step=int(data.data_manifest["tokens_per_optimizer_step"])
    reader=NonRepeatingTokenReader(data.train, cfg.model.block_size); val_reader=NonRepeatingTokenReader(data.val + data.train, cfg.model.block_size)
    write_json(run_dir/"environment.json", environment()); (run_dir/"initialization_hash.txt").write_text(init_hash)
    man={"smoke_test": False, "valid_for_science": True, "level": level, "token_multiplier": token_multiplier, "seed": seed, "pair_id": pair_id, "optimizer": optimizer_name, "requested_tokens": data.data_manifest["requested_tokens"], "realized_tokens": data.data_manifest["realized_tokens"], "optimizer_steps": steps, "dataset_name": data.data_manifest["dataset_name"], "dataset_config": data.data_manifest["dataset_config"], "dataset_revision": data.data_manifest["dataset_revision"], "tokenizer_hash": data.tokenizer_manifest["tokenizer_hash"], "data_hash": data.corpus_hash, "corpus_hash": data.corpus_hash, "initialization_hash": init_hash, "parameter_report": GPT(cfg.model).report_dict(), "estimated_flops": 6 * GPT(cfg.model).parameter_report().total_parameters * int(data.data_manifest["realized_tokens"])}
    write_json(run_dir/"manifest.json", man); write_json(run_dir/"data_manifest.json", data.data_manifest); write_json(run_dir/"tokenizer_manifest.json", data.tokenizer_manifest)
    cfgd=json.loads(json.dumps(asdict(cfg))); (run_dir/"config.yaml").write_text(yaml.safe_dump(cfgd)); write_json(run_dir/"config.json", cfgd)
    metric_rows=[]; spectral_rows=[]; proj_rows=[]; start=time.perf_counter(); last_loss=0.0; ww_over=0.0
    for step in range(1, steps+1):
        xb,yb=reader.next_batch(cfg.train.batch_size); x=torch.tensor(xb,device=select_device(device)); y=torch.tensor(yb,device=select_device(device))
        _,loss=model(x,y); opt.zero_grad(); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip); opt.step(); proj_time=0.0
        if optimizer_name=="adamw_wwpgd" and step % (ww_interval or cfg.wwpgd.interval)==0:
            ps=time.perf_counter(); proj_rows.extend(apply_wwpgd(model,cfg.wwpgd.target_alpha,cfg.wwpgd.strength,step,cfg.wwpgd.warmup_steps,cfg.wwpgd.ramp_steps)); proj_time=time.perf_counter()-ps
        if step % (eval_interval or cfg.train.eval_interval)==0 or step==steps:
            with torch.no_grad():
                vx,vy=val_reader.next_batch(cfg.train.batch_size); vt=torch.tensor(vx,device=select_device(device)); vyt=torch.tensor(vy,device=select_device(device)); vlogits,vloss=model(vt,vyt); tlogits,tloss=model(x,y)
            tm=_metrics(float(tloss.detach().cpu()), tlogits.detach().cpu(), y.detach().cpu()); vm=_metrics(float(vloss.detach().cpu()), vlogits.detach().cpu(), torch.tensor(vy))
            elapsed=time.perf_counter()-start; last_loss=float(vloss.detach().cpu())
            metric_rows.append({"step":step,"tokens_processed":step*tokens_per_step,"elapsed_time":elapsed,"learning_rate":cfg.train.learning_rate,"gradient_norm":float(grad.detach().cpu()),"train_minibatch_loss":float(loss.detach().cpu()),"train_loss":tm["loss"],"val_loss":vm["loss"],"train_perplexity":tm["perplexity"],"val_perplexity":vm["perplexity"],"train_bits_per_token":tm["bits_per_token"],"val_bits_per_token":vm["bits_per_token"],"train_top1_accuracy":tm["top1_accuracy"],"val_top1_accuracy":vm["top1_accuracy"],"train_top5_accuracy":tm["top5_accuracy"],"val_top5_accuracy":vm["top5_accuracy"],"train_token_error":tm["token_error"],"val_token_error":vm["token_error"],"generalization_gap":vm["loss"]-tm["loss"],"tokens_per_second":(step*tokens_per_step)/max(elapsed,1e-9),"examples_per_second":(step*cfg.train.batch_size)/max(elapsed,1e-9),"weightwatcher_overhead":ww_over,"projection_overhead":proj_time,"peak_memory":float(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0.0)})
            ws=time.perf_counter()
            for lid,(name,w) in enumerate(matrix_modules(model)):
                rec=asdict(spectral_summary(name,w,require_weightwatcher=True)); rec.update({"layer_id":lid,"step":step,"optimizer":optimizer_name,"seed":seed}); spectral_rows.append(rec)
            ww_over += time.perf_counter()-ws
        if step % (checkpoint_interval or cfg.train.checkpoint_interval)==0: torch.save({"model":model.state_dict(),"step":step}, ckpt/f"latest_step_{step:06d}_{seed}.pt")
    torch.save(model.state_dict(), ckpt/f"final_step_{steps:06d}_{seed}.pt")
    _write_csv(run_dir/"metrics.csv", metric_rows); _write_csv(run_dir/"spectral.csv", spectral_rows)
    if optimizer_name=="adamw_wwpgd": _write_csv(run_dir/"wwpgd_projection.csv", proj_rows)
    (run_dir/"events.jsonl").write_text(json.dumps({"event":"complete"})+"\n"); write_json(run_dir/"run_complete.json", {"step":steps,"final_val_loss":last_loss})
    return run_dir


def run_multiseed_scientific(level:int, data_root:Path, results_root:Path, token_multiplier:int, seeds:list[int]|None=None, config_path:Path|None=None, device:str|None=None, ww_interval:int|None=None, eval_interval:int|None=None, checkpoint_interval:int|None=None) -> Path:
    from wwgpt.data import load_prepared_scientific_data
    cfg=load_config(config_path, level); data=load_prepared_scientific_data(data_root, level, token_multiplier)
    exp_root=results_root/"experiments"/f"level_{level:02d}"/f"multiplier_{token_multiplier}"; exp_root.mkdir(parents=True,exist_ok=True)
    for seed in (seeds or DEFAULT_SEEDS):
        pair=unique_dir(exp_root, f"pair_{seed}"); pair_id=pair.name; torch.manual_seed(seed); init_model=GPT(cfg.model); init_state={k:v.detach().clone() for k,v in init_model.state_dict().items()}; init_hash=_state_hash(init_state)
        init_dir=pair/"initial_state"; init_dir.mkdir(); torch.save(init_state, init_dir/"model.pt"); (init_dir/"initialization_hash.txt").write_text(init_hash)
        write_json(pair/"pair_manifest.json", {"pair_id":pair_id,"seed":seed,"level":level,"token_multiplier":token_multiplier,"initialization_hash":init_hash,"arms":["adamw","adamw_wwpgd"]})
        run_scientific_single(pair,"adamw",seed,cfg,data,pair_id,init_state,init_hash,level,token_multiplier,device,ww_interval,eval_interval,checkpoint_interval)
        run_scientific_single(pair,"adamw_wwpgd",seed,cfg,data,pair_id,init_state,init_hash,level,token_multiplier,device,ww_interval,eval_interval,checkpoint_interval)
    return exp_root
