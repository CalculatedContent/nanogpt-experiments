from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml
import numpy as np

from wwgpt.config import (
    DEFAULT_SEEDS,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    WWPGDConfig,
    load_config,
)
from wwgpt.optim import ARM_DISPLAY, arm_name as make_arm_name, build_optimizer_bundle, apply_lr_schedule, resolve_warmup_steps
from wwgpt.data import NonRepeatingTokenReader, RandomWindowTokenReader, prepare_local_text, fixed_probe, random_probe, stable_seed
from wwgpt.model import GPT
from wwgpt.utils import environment, sha256_bytes, unique_dir, write_json
from wwgpt.checkpointing import assert_checkpoint_compatible, load_latest_checkpoint, rng_state, restore_rng_state, save_checkpoint, stable_hash
from wwgpt.ww import (
    apply_wwpgd,
    apply_external_wwpgd,
    fallback_spectral_summary,
    spectral_summary,
    composite_spectral_summary,
    weightwatcher_details,
    measured_projection_spectral_rows,
    weightwatcher_details,
    _ww_version,
    WWPGD_COMMIT,
    SCIENTIFIC_SCHEMA_VERSION,
    resolved_external_wwpgd_config,
    external_wwpgd_manifest_fields,
)



def _select_resume_run(arm_dir: Path, expected: dict[str, object]) -> Path:
    if not arm_dir.exists():
        raise FileNotFoundError(f"no runs exist for resume arm directory: {arm_dir}")
    candidates=[]; incompatible=[]
    for run in sorted([p for p in arm_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        if not (run / "checkpoints" / "latest.json").exists():
            continue
        try:
            manifest=json.loads((run / "manifest.json").read_text())
            data_manifest=json.loads((run / "data_manifest.json").read_text())
            tok_manifest=json.loads((run / "tokenizer_manifest.json").read_text())
            init=(run / "initialization_hash.txt").read_text().strip()
        except Exception as exc:
            incompatible.append({"run": str(run), "error": str(exc)})
            continue
        observed={
            "pair_id": manifest.get("pair_id"),
            "arm_name": manifest.get("arm_name", manifest.get("optimizer")),
            "seed": manifest.get("seed"),
            "configuration_hash": manifest.get("configuration_hash"),
            "data_hash": manifest.get("data_hash", data_manifest.get("corpus_hash")),
            "tokenizer_hash": manifest.get("tokenizer_hash", tok_manifest.get("tokenizer_hash")),
            "initialization_hash": manifest.get("initialization_hash", init),
            "immediate_projection_spectral": manifest.get("immediate_projection_spectral"),
        }
        mm={k:{"expected": v, "found": observed.get(k)} for k,v in expected.items() if observed.get(k) != v}
        if mm:
            incompatible.append({"run": str(run), "mismatches": mm})
        else:
            candidates.append(run)
    if len(candidates)==1:
        return candidates[0]
    if not candidates:
        raise RuntimeError("no compatible resume run found; mismatches=" + json.dumps(incompatible, sort_keys=True, default=str))
    raise RuntimeError("multiple compatible resume runs found; refusing ambiguous resume: " + json.dumps([str(p) for p in candidates]))


class TrainingExtension:
    name = "base"
    def after_optimizer_step(self, *, model, optimizer_step: int, total_optimizer_steps: int, tokens_seen: int):
        return []


class NoExtension(TrainingExtension):
    name = "none"


class WWPGDExtension(TrainingExtension):
    name = "wwpgd"
    def __init__(self, cfg: WWPGDConfig, interval: int):
        self.cfg = cfg; self.interval = interval
    def after_optimizer_step(self, *, model, optimizer_step: int, total_optimizer_steps: int, tokens_seen: int) -> list[dict[str, object]]:
        if optimizer_step % self.interval != 0:
            return []
        event = optimizer_step // self.interval - 1
        details = weightwatcher_details(model)
        frac = max(0.0, min(1.0, optimizer_step / max(1, total_optimizer_steps)))
        rows = apply_external_wwpgd(model, event_index=event, scheduled_token_fraction=frac, actual_step=optimizer_step, actual_tokens_seen=tokens_seen, cfg=resolved_external_wwpgd_config())
        return details, rows

def _log_train_progress(message: str) -> None:
    print(f"[wwgpt run-multiseed] {message}", file=sys.stderr, flush=True)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _metrics(loss: float, logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
    top1 = float((pred[..., 0] == y).float().mean())
    top5 = float((pred == y.unsqueeze(-1)).any(dim=-1).float().mean())
    return {
        "loss": loss,
        "perplexity": float(math.exp(min(loss, 20))),
        "bits_per_token": loss / math.log(2),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "token_error": 1 - top1,
    }


def _evaluate_probe_batches(
    model: GPT, probe_x: np.ndarray, probe_y: np.ndarray, device: torch.device
) -> tuple[dict[str, float], float]:
    loss_sum = 0.0
    token_count = 0
    top1_correct = 0
    top5_correct = 0
    for batch_x, batch_y in zip(probe_x, probe_y, strict=True):
        x = torch.tensor(batch_x, device=device)
        y = torch.tensor(batch_y, device=device)
        logits, loss = model(x, y)
        assert loss is not None
        tokens = int(y.numel())
        loss_sum += float(loss.detach().cpu()) * tokens
        token_count += tokens
        pred = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
        top1_correct += int((pred[..., 0] == y).sum().detach().cpu())
        top5_correct += int((pred == y.unsqueeze(-1)).any(dim=-1).sum().detach().cpu())
    mean_loss = loss_sum / max(token_count, 1)
    top1 = top1_correct / max(token_count, 1)
    top5 = top5_correct / max(token_count, 1)
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 20))),
        "bits_per_token": mean_loss / math.log(2),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "token_error": 1 - top1,
    }, mean_loss


def run_single(
    run_parent: Path,
    optimizer_name: str,
    seed: int,
    cfg: ExperimentConfig,
    train_tokens: list[int],
    val_tokens: list[int],
    pair_id: str,
    max_steps: int | None = None,
    init_state: dict[str, torch.Tensor] | None = None,
) -> Path:
    torch.manual_seed(seed)
    if optimizer_name in {"adamw_wwpgd_reference", "adamw_wwpgd"}:
        base_optimizer, extension_name = "adamw", "wwpgd"
    elif optimizer_name in {"adamw", "muon", "stableadamw"}:
        base_optimizer, extension_name = optimizer_name, getattr(cfg.wwpgd, "extension", "none")
    else:
        base_optimizer, extension_name = optimizer_name, getattr(cfg.wwpgd, "extension", "none")
    optimizer_name = make_arm_name(base_optimizer, extension_name)
    run_dir = unique_dir(run_parent / optimizer_name, "run")
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir()
    model_cfg = ModelConfig(
        **{**asdict(cfg.model), "vocab_size": max(train_tokens + val_tokens) + 1}
    )
    model = GPT(model_cfg)
    if init_state is not None:
        model.load_state_dict(init_state)
    init_hash = sha256_bytes(
        b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())
    )
    (run_dir / "initialization_hash.txt").write_text(init_hash)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=cfg.train.betas,
        eps=cfg.train.epsilon,
        weight_decay=cfg.train.weight_decay,
    )
    steps = max_steps or cfg.train.max_steps or 3
    reader = NonRepeatingTokenReader(train_tokens, model_cfg.block_size)
    val_reader = NonRepeatingTokenReader(val_tokens + train_tokens, model_cfg.block_size)
    metric_rows = []
    spectral_rows = []
    proj_rows = []
    immediate_spectral_rows = []
    write_json(run_dir / "environment.json", environment())
    write_json(
        run_dir / "manifest.json",
        {
            "optimizer": optimizer_name,
        "base_optimizer": base_optimizer,
        "extension": extension_name,
        "arm_name": optimizer_name,
        "arm_display_name": ARM_DISPLAY[optimizer_name],
            "seed": seed,
            "pair_id": pair_id,
            "smoke_test": True,
            "valid_for_science": False,
            "parameter_report": model.report_dict(),
        },
    )
    write_json(
        run_dir / "data_manifest.json",
        {
            "dataset": "local_text",
            "corpus_hash": sha256_bytes(bytes([x % 256 for x in train_tokens])),
        },
    )
    write_json(
        run_dir / "tokenizer_manifest.json",
        {"tokenizer": "char-smoke", "vocab_size": model_cfg.vocab_size},
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(asdict(cfg)))))
    write_json(run_dir / "config.json", json.loads(json.dumps(asdict(cfg))))
    torch.save(model.state_dict(), ckpt / f"initial_step_000000_{seed}.pt")
    _log_train_progress(
        f"starting smoke run optimizer={optimizer_name} seed={seed} pair={pair_id} steps={steps} output={run_dir}"
    )
    start = time.perf_counter()
    last_loss = 0.0
    for step in range(1, steps + 1):
        xb, yb = reader.next_batch(cfg.train.batch_size)
        x = torch.tensor(xb)
        y = torch.tensor(yb)
        _, loss = model(x, y)
        assert loss is not None
        opt.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        proj_time = 0.0
        if optimizer_name in {"adamw_wwpgd", "adamw_wwpgd_reference"}:
            pstart = time.perf_counter()
            proj_rows.extend(apply_wwpgd(model, cfg.wwpgd.target_alpha, cfg.wwpgd.strength, step))
            proj_time = time.perf_counter() - pstart
        with torch.no_grad():
            vx, vy = val_reader.next_batch(cfg.train.batch_size)
            vlogits, vloss = model(torch.tensor(vx), torch.tensor(vy))
            assert vloss is not None
            tlogits, tloss = model(x, y)
            assert tloss is not None
        tm = _metrics(float(tloss.detach()), tlogits, y)
        vm = _metrics(float(vloss.detach()), vlogits, torch.tensor(vy))
        elapsed = time.perf_counter() - start
        last_loss = float(vloss.detach())
        metric_rows.append(
            {
                "step": step,
                "tokens_processed": step * cfg.train.batch_size * model_cfg.block_size,
                "elapsed_time": elapsed,
                "learning_rate": cfg.train.learning_rate,
                "gradient_norm": float(grad.detach()),
                "train_minibatch_loss": float(loss.detach()),
                "train_loss": tm["loss"],
                "val_loss": vm["loss"],
                "train_perplexity": tm["perplexity"],
                "val_perplexity": vm["perplexity"],
                "train_bits_per_token": tm["bits_per_token"],
                "val_bits_per_token": vm["bits_per_token"],
                "train_top1_accuracy": tm["top1_accuracy"],
                "val_top1_accuracy": vm["top1_accuracy"],
                "train_top5_accuracy": tm["top5_accuracy"],
                "val_top5_accuracy": vm["top5_accuracy"],
                "train_token_error": tm["token_error"],
                "val_token_error": vm["token_error"],
                "generalization_gap": vm["loss"] - tm["loss"],
                "tokens_per_second": (step * cfg.train.batch_size * model_cfg.block_size)
                / max(elapsed, 1e-9),
                "examples_per_second": (step * cfg.train.batch_size) / max(elapsed, 1e-9),
                "weightwatcher_overhead": 0.0,
                "projection_overhead": proj_time,
                "peak_memory": 0.0,
            }
        )
        spectral_rows.extend(
            fallback_spectral_summary(
                model,
                step=step,
                tokens_seen=step * cfg.train.batch_size * model_cfg.block_size,
                optimizer=optimizer_name,
                seed=seed,
                pair_id=pair_id,
            )
        )
        _log_train_progress(
            f"smoke progress optimizer={optimizer_name} seed={seed} step={step}/{steps} val_loss={last_loss:.4f} elapsed_s={elapsed:.1f}"
        )
        torch.save(
            {"model": model.state_dict(), "step": step}, ckpt / f"latest_step_{step:06d}_{seed}.pt"
        )
    torch.save(model.state_dict(), ckpt / f"final_step_{steps:06d}_{seed}.pt")
    torch.save(model.state_dict(), ckpt / f"best_val_step_{steps:06d}_{seed}.pt")
    _write_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "spectral.csv", spectral_rows)
    if optimizer_name in {"adamw_wwpgd", "adamw_wwpgd_reference"}:
        _write_csv(run_dir / "wwpgd_projection.csv", proj_rows)
    (run_dir / "events.jsonl").write_text(json.dumps({"event": "complete"}) + "\n")
    write_json(run_dir / "run_complete.json", {"step": steps, "final_val_loss": last_loss})
    _log_train_progress(
        f"completed smoke run optimizer={optimizer_name} seed={seed} steps={steps} final_val_loss={last_loss:.4f} output={run_dir}"
    )
    return run_dir


def smoke(root: Path, steps: int = 3, seeds: list[int] | None = None) -> Path:
    run_seeds = seeds or [1337]
    smoke_dir = unique_dir(root, "wwgpt_invalid_smoke")
    text = (
        "WeightWatcher PGD smoke corpus. This is not Tiny Shakespeare and is invalid for science. "
        * 400
    ).split(".")
    cfg = ExperimentConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=32, block_size=16, vocab_size=128),
        train=TrainConfig(batch_size=2, max_steps=steps, eval_interval=1),
        wwpgd=WWPGDConfig(enabled=True, strength=0.01),
    )
    data = prepare_local_text(
        smoke_dir / "data",
        [t + "." for t in text],
        min_train_tokens=steps * cfg.train.batch_size * cfg.model.block_size * 2 + 1,
    )
    pair_parent = smoke_dir / "level_00" / "pair_invalid"
    for seed in run_seeds:
        torch.manual_seed(seed)
        init = GPT(ModelConfig(**{**asdict(cfg.model), "vocab_size": data.vocab_size})).state_dict()
        for opt in ["adamw", "adamw_wwpgd_reference"]:
            run_single(
                pair_parent,
                opt,
                seed,
                cfg,
                data.train,
                data.val,
                f"pair_invalid_seed_{seed}",
                steps,
                init,
            )
    return smoke_dir


def select_device(override: str | None = None):
    from wwgpt.device import resolve_device
    return resolve_device(override or "auto")


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    return sha256_bytes(b"".join(state[k].detach().cpu().numpy().tobytes() for k in sorted(state)))


def _compatibility(cfg: ExperimentConfig, data, init_hash: str, validation_probe_hash: str, training_probe_hash: str) -> dict[str, object]:
    cfgd = json.loads(json.dumps(asdict(cfg)))
    return {
        "configuration_hash": stable_hash(cfgd),
        "model_configuration_hash": stable_hash(cfgd.get("model", {})),
        "training_configuration_hash": stable_hash(cfgd.get("train", {})),
        "wwpgd_configuration_hash": stable_hash(cfgd.get("wwpgd", {})),
        "data_hash": data.corpus_hash,
        "tokenizer_hash": data.tokenizer_manifest.get("tokenizer_hash") or data.tokenizer_manifest.get("hash"),
        "initialization_hash": init_hash,
        "validation_probe_hash": validation_probe_hash,
        "training_probe_hash": training_probe_hash,
        "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
    }


def run_scientific_single(
    run_parent: Path,
    optimizer_name: str,
    seed: int,
    cfg: ExperimentConfig,
    data,
    pair_id: str,
    init_state: dict[str, torch.Tensor],
    init_hash: str,
    level: int,
    token_multiplier: int,
    device: str | None = None,
    ww_interval: int | None = None,
    eval_interval: int | None = None,
    checkpoint_interval: int | None = None,
    spectral_interval: int | None = None,
    precision: str | None = None,
    resume: bool = False,
    immediate_projection_spectral: bool = False,
    allow_code_version_mismatch: bool = False,
) -> Path:
    torch.manual_seed(seed)
    if optimizer_name in {"adamw_wwpgd_reference", "adamw_wwpgd"}:
        base_optimizer, extension_name = "adamw", "wwpgd"
    elif optimizer_name in {"adamw", "muon", "stableadamw"}:
        base_optimizer, extension_name = optimizer_name, getattr(cfg.wwpgd, "extension", "none")
    else:
        base_optimizer, extension_name = optimizer_name, getattr(cfg.wwpgd, "extension", "none")
    optimizer_name = make_arm_name(base_optimizer, extension_name)
    run_dir = None
    ckpt = None
    selected_device = select_device(device)
    model = GPT(cfg.model).to(selected_device)
    model.load_state_dict(init_state)
    bundle, resolved_llrd_gamma = build_optimizer_bundle(model, cfg.train, base_optimizer)
    report = model.parameter_report()
    parameter_count_used = report.total_parameters if cfg.parameter_count_convention == "total" else report.non_embedding_parameters
    tokens_per_step = cfg.train.batch_size * cfg.model.block_size * cfg.train.gradient_accumulation
    if cfg.train.max_steps is not None:
        steps = cfg.train.max_steps; target_tokens = steps * tokens_per_step; budget_source = "max_steps"
    elif cfg.train.max_train_tokens is not None:
        target_tokens = cfg.train.max_train_tokens; steps = max(1, math.ceil(target_tokens / tokens_per_step)); budget_source = "max_train_tokens"
    else:
        target_tokens = parameter_count_used * token_multiplier; steps = max(1, math.ceil(target_tokens / tokens_per_step)); budget_source = "token_multiplier"
    realized_tokens = steps * tokens_per_step
    resolved_warmup_steps = resolve_warmup_steps(steps, cfg.train.warmup_ratio, cfg.train.warmup_steps)
    wwpgd_interval = int(ww_interval or cfg.train.wwpgd_interval or eval_interval or cfg.train.eval_interval)
    extension = WWPGDExtension(cfg.wwpgd, wwpgd_interval) if extension_name == "wwpgd" else NoExtension()
    reader = (RandomWindowTokenReader(data.train, cfg.model.block_size, stable_seed(seed, pair_id, "train_reader_v1")) if cfg.train.training_sampling == "random_window" else NonRepeatingTokenReader(data.train, cfg.model.block_size))
    validation_probe_hash = ""
    training_probe_hash = ""
    assert not np.shares_memory(np.array(data.val), np.array(data.train))
    man = {
        "smoke_test": False,
        "valid_for_science": True,
        "level": level,
        "token_multiplier": token_multiplier,
        "seed": seed,
        "pair_id": pair_id,
        "optimizer": optimizer_name,
        "base_optimizer": base_optimizer,
        "extension": extension_name,
        "arm_name": optimizer_name,
        "arm_display_name": ARM_DISPLAY[optimizer_name],
        "requested_tokens": target_tokens,
        "target_train_tokens": target_tokens,
        "realized_tokens": realized_tokens,
        "realized_train_tokens": realized_tokens,
        "optimizer_steps": steps,
        "total_optimizer_steps": steps,
        "tokens_per_optimizer_step": tokens_per_step,
        "budget_source": budget_source,
        "parameter_count_convention": cfg.parameter_count_convention,
        "parameter_count_used": parameter_count_used,
        "dataset_name": data.data_manifest["dataset_name"],
        "dataset_config": data.data_manifest["dataset_config"],
        "dataset_revision": data.data_manifest["dataset_revision"],
        "tokenizer_hash": data.tokenizer_manifest["tokenizer_hash"],
        "data_hash": data.corpus_hash,
        "corpus_hash": data.corpus_hash,
        "initialization_hash": init_hash,
        "parameter_report": model.report_dict(),
        "model_config": asdict(cfg.model),
        "model_architecture_version": cfg.model.model_architecture_version,
        "model_config_hash": sha256_bytes(json.dumps(asdict(cfg.model), sort_keys=True).encode()),
        "optimizer_hyperparameters": asdict(cfg.train),
        "extension_hyperparameters": asdict(cfg.wwpgd),
        "training_schedule_hash": sha256_bytes(json.dumps({"seed": seed, "steps": steps, "batch": cfg.train.batch_size, "training_sampling": cfg.train.training_sampling}, sort_keys=True).encode()),
        "training_sampling": cfg.train.training_sampling,
        "evaluation_sampling": cfg.train.evaluation_sampling,
        "evaluation_schedule_version": "random_per_eval_v1",
        "lr_schedule": cfg.train.lr_schedule,
        "resolved_warmup_steps": resolved_warmup_steps,
        "layer_lr": cfg.train.layer_lr,
        "resolved_llrd_gamma": resolved_llrd_gamma,
        "llrd_min_multiplier": cfg.train.llrd_min_multiplier,
        "weight_decay": cfg.train.weight_decay,
        "grad_clip": cfg.train.grad_clip,
        "batch_size": cfg.train.batch_size,
        "gradient_accumulation": cfg.train.gradient_accumulation,
        "wwpgd_interval": wwpgd_interval,
        "projection_schedule_type": "optimizer_step_interval",
        "total_projection_events": steps // wwpgd_interval,
        "WeightWatcher version": "",
        "spectral estimator": "weightwatcher",
        "composite specification version": "raw_and_composite_v1",
        "estimated_flops": 6
        * GPT(cfg.model).parameter_report().total_parameters
        * int(data.data_manifest["realized_tokens"]),
        "spectral_estimator": "weightwatcher",
        "spectral_estimator_version": "",
        "wwpgd_implementation": "ww_pgd" if extension_name == "wwpgd" else "none",
        "wwpgd_commit": WWPGD_COMMIT if extension_name == "wwpgd" else "",
        "projection_schedule": cfg.wwpgd.projection_schedule,
        "validation_probe_hash": validation_probe_hash,
        "training_probe_hash": training_probe_hash,
        "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "checkpoint_schema_version": 2,
        "immediate_projection_spectral": immediate_projection_spectral,
        "immediate_spectral_source": "weightwatcher_measured" if immediate_projection_spectral else "disabled",
        "weightwatcher_version": _ww_version(),
        "weightwatcher_configuration": {"detX": True, "randomize": False, "plot": False},
    }
    man.update(external_wwpgd_manifest_fields(extension_name == "wwpgd"))
    cfgd_for_hash = json.loads(json.dumps(asdict(cfg)))
    man.update({
        "configuration_hash": stable_hash(cfgd_for_hash),
        "model_configuration_hash": stable_hash(cfgd_for_hash.get("model", {})),
        "training_configuration_hash": stable_hash(cfgd_for_hash.get("train", {})),
        "wwpgd_configuration_hash": stable_hash(cfgd_for_hash.get("wwpgd", {})),
    })
    expected_identity = {
        "pair_id": pair_id,
        "arm_name": optimizer_name,
        "seed": seed,
        "configuration_hash": man["configuration_hash"],
        "data_hash": data.corpus_hash,
        "tokenizer_hash": data.tokenizer_manifest["tokenizer_hash"],
        "initialization_hash": init_hash,
        "immediate_projection_spectral": immediate_projection_spectral,
    }
    if resume:
        run_dir = _select_resume_run(run_parent / optimizer_name, expected_identity)
        ckpt = run_dir / "checkpoints"
    else:
        run_dir = unique_dir(run_parent / optimizer_name, "run")
        ckpt = run_dir / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "environment.json", environment())
    (run_dir / "initialization_hash.txt").write_text(init_hash)
    write_json(run_dir / "manifest.json", man)
    write_json(run_dir / "data_manifest.json", data.data_manifest)
    write_json(run_dir / "tokenizer_manifest.json", data.tokenizer_manifest)
    cfgd = json.loads(json.dumps(asdict(cfg)))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfgd))
    write_json(run_dir / "config.json", cfgd)
    metric_rows = []
    spectral_rows = []
    composite_rows = []
    proj_rows = []
    immediate_spectral_rows = []
    lr_rows = []
    best_validation_loss = float("inf")
    best_validation_step = 0
    latest_validation_loss = float("nan")
    completed_projection_event_indexes = []
    next_projection_event_index = 0
    elapsed_prior = 0.0
    compatibility = _compatibility(cfg, data, init_hash, validation_probe_hash, training_probe_hash)
    compatibility.update({"optimizer_name": optimizer_name, "optimizer_class": type(bundle.optimizers[0]).__name__, "immediate_projection_spectral": immediate_projection_spectral, "weightwatcher_version": _ww_version(), "weightwatcher_configuration": {"detX": True, "randomize": False, "plot": False}, "wwpgd_commit": WWPGD_COMMIT if extension_name == "wwpgd" else "", "git_commit": man.get("git_commit", "unknown"), "seed": seed, "level": level, "token_multiplier": token_multiplier, "requested_tokens": target_tokens, "realized_tokens": realized_tokens})
    _log_train_progress(
        f"starting run level={level} token_multiplier={token_multiplier} pair={pair_id} optimizer={optimizer_name} seed={seed} steps={steps} device={selected_device} output={run_dir}"
    )
    start_step = 1
    if resume:
        loaded = load_latest_checkpoint(run_dir)
        assert_checkpoint_compatible(loaded, compatibility)
        model.load_state_dict(loaded["model_state_dict"])
        bundle.load_state_dict(loaded["optimizer_state_dict"])
        if "training_reader_state" in loaded and hasattr(reader, "load_state_dict"):
            reader.load_state_dict(loaded["training_reader_state"])
        else:
            reader.pos = int(loaded["training_reader_position"])
        metric_rows = list(loaded.get("metrics_rows", []))
        spectral_rows = list(loaded.get("periodic_weightwatcher_rows", []))
        proj_rows = list(loaded.get("wwpgd_projection_rows", []))
        immediate_spectral_rows = list(loaded.get("immediate_projection_weightwatcher_rows", []))
        best_validation_loss = float(loaded.get("best_validation_loss", best_validation_loss))
        best_validation_step = int(loaded.get("best_validation_step", best_validation_step))
        latest_validation_loss = float(loaded.get("latest_validation_loss", latest_validation_loss))
        completed_projection_event_indexes = list(loaded.get("completed_projection_event_indexes", []))
        next_projection_event_index = int(loaded.get("next_projection_event_index", len(completed_projection_event_indexes)))
        elapsed_prior = float(loaded.get("elapsed_training_time", 0.0))
        restore_rng_state(loaded)
        start_step = int(loaded.get("next_step", int(loaded.get("current_step", 0)) + 1))
        _log_train_progress(f"resuming run pair={pair_id} optimizer={optimizer_name} seed={seed} from step={start_step} checkpoint={run_dir}")
    start = time.perf_counter()
    last_loss = latest_validation_loss if math.isfinite(latest_validation_loss) else 0.0
    ww_over = 0.0
    for step in range(start_step, steps + 1):
        lr_rows.extend(apply_lr_schedule(bundle, step - 1, steps, resolved_warmup_steps, cfg.train))
        bundle.zero_grad()
        train_loss_value = 0.0
        for _ in range(cfg.train.gradient_accumulation):
            xb, yb = reader.next_batch(cfg.train.batch_size)
            x = torch.tensor(xb, device=selected_device)
            y = torch.tensor(yb, device=selected_device)
            _, loss = model(x, y)
            assert loss is not None
            (loss / cfg.train.gradient_accumulation).backward()
            train_loss_value += float(loss.detach().cpu())
        grad = torch.tensor(0.0)
        if cfg.train.grad_clip > 0.0:
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        bundle.step()
        loss = torch.tensor(train_loss_value / cfg.train.gradient_accumulation)
        ps = time.perf_counter()
        ext_result = extension.after_optimizer_step(model=model, optimizer_step=step, total_optimizer_steps=steps, tokens_seen=step * tokens_per_step)
        if isinstance(ext_result, tuple):
            pre_details, new_proj = ext_result
        else:
            pre_details, new_proj = None, ext_result
        proj_rows.extend(new_proj)
        if new_proj:
            event_idx = int(new_proj[0].get("projection_event", next_projection_event_index))
            if event_idx not in completed_projection_event_indexes:
                completed_projection_event_indexes.append(event_idx)
            next_projection_event_index = max(next_projection_event_index, event_idx + 1)
            if immediate_projection_spectral:
                post = measured_projection_spectral_rows(pre_details, model, step=step, tokens_seen=step * tokens_per_step, optimizer=optimizer_name, seed=seed, pair_id=pair_id, projection_event=event_idx, projection_rows=new_proj, target_alpha=cfg.wwpgd.target_alpha, phase="post")
                immediate_spectral_rows.extend(post)
        proj_time = time.perf_counter() - ps if new_proj else 0.0
        if step % (eval_interval or cfg.train.eval_interval) == 0 or step == steps:
            eval_index = len(metric_rows)
            was_training = model.training
            model.eval()
            if cfg.train.evaluation_sampling == "fixed_probe":
                train_x, train_y, training_probe_hash = fixed_probe(data.train[cfg.train.batch_size * cfg.model.block_size * 2:], cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
                val_x, val_y, validation_probe_hash = fixed_probe(data.val, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
            else:
                train_x, train_y, training_probe_hash = random_probe(data.train, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches, stable_seed(seed, "train", eval_index, "random_per_eval_v1"))
                val_x, val_y, validation_probe_hash = random_probe(data.val, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches, stable_seed(seed, "val", eval_index, "random_per_eval_v1"))
            with torch.no_grad():
                tm, _ = _evaluate_probe_batches(model, train_x, train_y, selected_device)
                vm, validation_probe_loss = _evaluate_probe_batches(model, val_x, val_y, selected_device)
            model.train(was_training)
            elapsed = time.perf_counter() - start
            last_loss = validation_probe_loss
            latest_validation_loss = validation_probe_loss
            if validation_probe_loss < best_validation_loss:
                best_validation_loss = validation_probe_loss
                best_validation_step = step
                torch.save(model.state_dict(), ckpt / f"best_val_step_{step:06d}_{seed}.pt")
            metric_rows.append(
                {
                    "step": step,
                    "tokens_processed": step * tokens_per_step,
                    "elapsed_time": elapsed,
                    "learning_rate": cfg.train.learning_rate,
                    "gradient_norm": float(grad.detach().cpu()),
                    "train_minibatch_loss": float(loss.detach().cpu()),
                    "train_loss": tm["loss"],
                    "val_loss": vm["loss"],
                    "train_perplexity": tm["perplexity"],
                    "val_perplexity": vm["perplexity"],
                    "train_bits_per_token": tm["bits_per_token"],
                    "val_bits_per_token": vm["bits_per_token"],
                    "train_top1_accuracy": tm["top1_accuracy"],
                    "val_top1_accuracy": vm["top1_accuracy"],
                    "train_top5_accuracy": tm["top5_accuracy"],
                    "val_top5_accuracy": vm["top5_accuracy"],
                    "train_token_error": tm["token_error"],
                    "val_token_error": vm["token_error"],
                    "generalization_gap": vm["loss"] - tm["loss"],
                    "evaluation_index": eval_index,
                    "evaluation_sampling": cfg.train.evaluation_sampling,
                    "train_eval_batch_hash": training_probe_hash,
                    "val_eval_batch_hash": validation_probe_hash,
                    "evaluation_token_count": int(
                        cfg.train.eval_batches * cfg.train.batch_size * cfg.model.block_size
                    ),
                    "validation_probe_hash": validation_probe_hash,
                    "training_probe_hash": training_probe_hash,
                    "evaluation_batches": cfg.train.eval_batches,
                    "validation_document_count": data.data_manifest.get(
                        "validation_document_count", 0
                    ),
                    "tokens_per_second": (step * tokens_per_step) / max(elapsed, 1e-9),
                    "examples_per_second": (step * cfg.train.batch_size) / max(elapsed, 1e-9),
                    "weightwatcher_overhead": ww_over,
                    "projection_overhead": proj_time,
                    "peak_memory": float(
                        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0.0
                    ),
                }
            )
            ws = time.perf_counter()
            if step % (spectral_interval or cfg.train.spectral_interval) == 0 or step == steps:
                spectral_rows.extend(
                    spectral_summary(
                        model,
                        step=step,
                        tokens_seen=step * tokens_per_step,
                        optimizer=optimizer_name,
                        seed=seed,
                        pair_id=pair_id,
                    )
                )
            ww_over += time.perf_counter() - ws
            _log_train_progress(
                f"progress pair={pair_id} optimizer={optimizer_name} seed={seed} step={step}/{steps} tokens={step * tokens_per_step}/{int(data.data_manifest['realized_tokens'])} train_loss={tm['loss']:.4f} val_loss={vm['loss']:.4f} elapsed_s={elapsed:.1f} tokens_per_s={(step * tokens_per_step) / max(elapsed, 1e-9):.1f}"
            )
        if False and cfg.composite_spectral_analysis_enabled and (step % (spectral_interval or cfg.train.spectral_interval) == 0 or step == steps):
            composite_rows.extend(composite_spectral_summary(model, step=step, tokens_seen=step * tokens_per_step, base_optimizer=base_optimizer, extension=extension_name, arm_name=optimizer_name, seed=seed, pair_id=pair_id))
        if step % (checkpoint_interval or cfg.train.checkpoint_interval) == 0:
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": bundle.state_dict(),
                "scheduler_state_dict": None,
                "gradient_scaler_state_dict": None,
                "current_step": step,
                "next_step": step + 1,
                "tokens_processed": step * tokens_per_step,
                "training_reader_position": reader.pos,
                "reader_position": reader.pos,
                "training_reader_state": reader.state_dict() if hasattr(reader, "state_dict") else {"pos": reader.pos},
                "seed": seed,
                **rng_state(),
                "device_type": selected_device.type,
                "precision_policy": precision or "torch_default",
                "gradient_accumulation_position": 0,
                "best_validation_loss": best_validation_loss,
                "best_validation_step": best_validation_step,
                "latest_validation_loss": latest_validation_loss,
                "completed_projection_event_indexes": completed_projection_event_indexes,
                "next_projection_event_index": next_projection_event_index,
                "projection_schedule": cfg.wwpgd.projection_schedule,
                "metrics_rows": metric_rows,
                "periodic_weightwatcher_rows": spectral_rows,
                "wwpgd_projection_rows": proj_rows,
                "immediate_projection_weightwatcher_rows": immediate_spectral_rows,
                "elapsed_training_time": elapsed_prior + time.perf_counter() - start,
                "initialization_hash": init_hash,
                "compatibility": compatibility,
                "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
                "weightwatcher_version": _ww_version(), "weightwatcher_configuration": {"detX": True, "randomize": False, "plot": False}, "wwpgd_commit": WWPGD_COMMIT if extension_name == "wwpgd" else "", "git_commit": man.get("git_commit", "unknown"), "optimizer_name": optimizer_name, "pair_id": pair_id, "level": level, "token_multiplier": token_multiplier, "realized_tokens": realized_tokens, "requested_tokens": target_tokens, "immediate_projection_spectral": immediate_projection_spectral, "run_directory": str(run_dir),
            }
            save_checkpoint(run_dir, state)
            _log_train_progress(
                f"checkpoint saved pair={pair_id} optimizer={optimizer_name} seed={seed} step={step}/{steps} dir={ckpt}"
            )
    final_elapsed = elapsed_prior + time.perf_counter() - start
    save_checkpoint(run_dir, {"model_state_dict": model.state_dict(), "optimizer_state_dict": bundle.state_dict(), "scheduler_state_dict": None, "gradient_scaler_state_dict": None, "current_step": steps, "next_step": steps + 1, "tokens_processed": steps * tokens_per_step, "training_reader_position": reader.pos, "reader_position": reader.pos, "training_reader_state": reader.state_dict() if hasattr(reader, "state_dict") else {"pos": reader.pos}, "seed": seed, **rng_state(), "device_type": selected_device.type, "precision_policy": precision or "torch_default", "gradient_accumulation_position": 0, "best_validation_loss": best_validation_loss, "best_validation_step": best_validation_step, "latest_validation_loss": latest_validation_loss, "completed_projection_event_indexes": completed_projection_event_indexes, "next_projection_event_index": next_projection_event_index, "projection_schedule": cfg.wwpgd.projection_schedule, "metrics_rows": metric_rows, "periodic_weightwatcher_rows": spectral_rows, "wwpgd_projection_rows": proj_rows, "immediate_projection_weightwatcher_rows": immediate_spectral_rows, "elapsed_training_time": final_elapsed, "initialization_hash": init_hash, "compatibility": compatibility, "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION, "weightwatcher_version": _ww_version(), "weightwatcher_configuration": {"detX": True, "randomize": False, "plot": False}, "wwpgd_commit": WWPGD_COMMIT if extension_name == "wwpgd" else "", "git_commit": man.get("git_commit", "unknown"), "optimizer_name": optimizer_name, "pair_id": pair_id, "level": level, "token_multiplier": token_multiplier, "realized_tokens": realized_tokens, "requested_tokens": target_tokens, "immediate_projection_spectral": immediate_projection_spectral, "run_directory": str(run_dir)})
    torch.save(model.state_dict(), ckpt / f"final_step_{steps:06d}_{seed}.pt")
    _write_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "spectral.csv", spectral_rows)
    _write_csv(run_dir / "composite_spectral.csv", composite_rows)
    _write_csv(run_dir / "lrs.csv", lr_rows)
    if extension_name == "wwpgd":
        _write_csv(run_dir / "wwpgd_projection.csv", proj_rows)
        if immediate_projection_spectral:
            _write_csv(run_dir / "wwpgd_projection_spectral.csv", immediate_spectral_rows)
    (run_dir / "events.jsonl").write_text(json.dumps({"event": "complete"}) + "\n")
    write_json(run_dir / "run_complete.json", {"step": steps, "final_val_loss": last_loss})
    _log_train_progress(
        f"completed run pair={pair_id} optimizer={optimizer_name} seed={seed} steps={steps} final_val_loss={last_loss:.4f} output={run_dir}"
    )
    return run_dir


def run_multiseed_scientific(
    level: int,
    data_root: Path,
    results_root: Path,
    token_multiplier: int,
    seeds: list[int] | None = None,
    config_path: Path | None = None,
    device: str | None = None,
    ww_interval: int | None = None,
    eval_interval: int | None = None,
    checkpoint_interval: int | None = None,
    spectral_interval: int | None = None,
    precision: str | None = None,
    resume: bool = False,
    optimizer: str = "adamw",
    extensions: list[str] | None = None,
    immediate_projection_spectral: bool = False,
    allow_code_version_mismatch: bool = False,
) -> Path:
    from wwgpt.data import load_prepared_scientific_data

    cfg = load_config(config_path, level)
    data = load_prepared_scientific_data(data_root, level, token_multiplier)
    exp_root = (
        results_root / "experiments" / f"level_{level:02d}" / f"multiplier_{token_multiplier}"
    )
    exp_root.mkdir(parents=True, exist_ok=True)
    run_seeds = seeds or DEFAULT_SEEDS
    _log_train_progress(
        f"starting multiseed level={level} token_multiplier={token_multiplier} seeds={','.join(str(s) for s in run_seeds)} results={exp_root}"
    )
    for seed_index, seed in enumerate(run_seeds, start=1):
        if resume:
            existing_pairs = sorted(
                [p for p in exp_root.glob(f"pair_{seed}*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            pair = existing_pairs[0] if existing_pairs else unique_dir(exp_root, f"pair_{seed}")
        else:
            pair = unique_dir(exp_root, f"pair_{seed}")
        pair_id = pair.name
        torch.manual_seed(seed)
        init_model = GPT(cfg.model)
        init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
        init_hash = _state_hash(init_state)
        _log_train_progress(
            f"starting seed {seed_index}/{len(run_seeds)} seed={seed} pair={pair_id}"
        )
        init_dir = pair / "initial_state"
        init_dir.mkdir(exist_ok=True)
        if not (init_dir / "model.pt").exists():
            torch.save(init_state, init_dir / "model.pt")
        (init_dir / "initialization_hash.txt").write_text(init_hash)
        write_json(
            pair / "pair_manifest.json",
            {
                "pair_id": pair_id,
                "seed": seed,
                "level": level,
                "token_multiplier": token_multiplier,
                "initialization_hash": init_hash,
                "base_optimizer": optimizer,
                "extensions": extensions or ["none", "wwpgd"],
                "arms": [make_arm_name(optimizer, e) for e in (extensions or ["none", "wwpgd"])],
            },
        )
        for ext in (extensions or ["none", "wwpgd"]):
            arm_cfg = replace(cfg, wwpgd=replace(cfg.wwpgd, extension=ext, enabled=(ext == "wwpgd")))
            run_scientific_single(
                pair,
                optimizer,
                seed,
                arm_cfg,
                data,
                pair_id,
                init_state,
                init_hash,
                level,
                token_multiplier,
                device,
                ww_interval,
                eval_interval,
                checkpoint_interval,
                spectral_interval,
                precision,
                resume,
                immediate_projection_spectral,
                allow_code_version_mismatch,
            )
        _log_train_progress(
            f"completed seed {seed_index}/{len(run_seeds)} seed={seed} pair={pair_id}"
        )
    _log_train_progress(
        f"completed multiseed level={level} token_multiplier={token_multiplier} output={exp_root}"
    )
    return exp_root
