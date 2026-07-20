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
from wwgpt.optim import ARM_DISPLAY, SCHEDULER_IMPLEMENTATION, arm_name as make_arm_name, build_optimizer_bundle, apply_lr_schedule, optimizer_fingerprint, resolve_lr_decay_steps, resolve_warmup_steps
from wwgpt.data import NonRepeatingTokenReader, RandomWindowTokenReader, prepare_local_text, fixed_probe, random_probe, stable_seed
from wwgpt.model import GPT
from wwgpt.utils import environment, sha256_bytes, unique_dir, write_json
from wwgpt.device import autocast_context, device_summary, memory_stats, optimizer_step, precision_policy, synchronize_device
from wwgpt.checkpointing import assert_checkpoint_compatible, load_latest_checkpoint, rng_state, restore_rng_state, save_checkpoint, stable_hash
from wwgpt.ww import (
    apply_external_wwpgd,
    external_wwpgd_config_from_experiment,
    fallback_spectral_summary,
    spectral_summary,
    composite_spectral_summary,
    weightwatcher_details,
    measured_projection_spectral_rows,
    weightwatcher_details,
    weightwatcher_run_aggregates,
    _ww_version,
    WWPGD_COMMIT,
    SCIENTIFIC_SCHEMA_VERSION,
    external_wwpgd_config_from_experiment,
    external_wwpgd_manifest_fields,
)




def resolved_stochastic_seeds(user_seed: int, level: int, token_multiplier: int, *, split: str = "train", optimizer_identity: str | None = None) -> dict[str, int]:
    """Resolve stochastic seeds from stable scientific identity only.

    Storage identifiers such as run IDs, pair IDs, paths, timestamps, and UUIDs must
    never enter this derivation. Optimizer identity is included only for the explicit
    optimizer-scoped stream.
    """
    base = ("wwgpt_scientific_seed_v1", int(user_seed), int(level), int(token_multiplier))
    seeds = {
        "model_init_seed": stable_seed(*base, "model_init"),
        "dropout_seed": stable_seed(*base, split, "dropout"),
        "train_reader_seed": stable_seed(*base, "train", "reader"),
        "train_eval_probe_seed_base": stable_seed(*base, "train", "eval_probe"),
        "val_eval_probe_seed_base": stable_seed(*base, "val", "eval_probe"),
    }
    if optimizer_identity is not None:
        seeds["optimizer_seed"] = stable_seed(*base, "optimizer", optimizer_identity)
    return seeds


def _initial_minibatch_indices(tokens, block_size: int, batch_size: int, sampling: str, reader_seed: int) -> list[int]:
    if sampling == "random_window":
        rng = np.random.default_rng(reader_seed)
        return [int(x) for x in rng.integers(0, len(tokens) - block_size, size=batch_size)]
    return list(range(0, batch_size * block_size, block_size))

def _select_resume_run(arm_dir: Path, expected: dict[str, object]) -> Path:
    if not arm_dir.exists():
        raise FileNotFoundError(f"no runs exist for resume arm directory: {arm_dir}")
    candidates=[]; incompatible=[]
    for run in sorted([p for p in arm_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        if not (run / "checkpoints" / "latest.json").exists() or (run / "run_complete.json").exists():
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
            "optimizer_fingerprint": manifest.get("optimizer_fingerprint"),
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

    def after_optimizer_step(self, *, model, optimizer_step: int, total_optimizer_steps: int, tokens_seen: int, collect_pre_details: bool = False):
        return None, []


class NoExtension(TrainingExtension):
    name = "none"


class WWPGDExtension(TrainingExtension):
    name = "wwpgd"

    def __init__(self, cfg: WWPGDConfig, interval: int = 1):
        if interval != 1:
            raise ValueError("standard WW-PGD interval must be 1 optimizer step")
        self.cfg = cfg
        self.interval = 1

    def after_optimizer_step(
        self,
        *,
        model,
        optimizer_step: int,
        total_optimizer_steps: int,
        tokens_seen: int,
        collect_pre_details: bool = False,
    ) -> tuple[object | None, list[dict[str, object]]]:
        event = optimizer_step - 1
        details = weightwatcher_details(model) if collect_pre_details else None
        frac = max(0.0, min(1.0, optimizer_step / max(1, total_optimizer_steps)))
        rows = apply_external_wwpgd(model, event_index=event, scheduled_token_fraction=frac, actual_step=optimizer_step, actual_tokens_seen=tokens_seen, cfg=external_wwpgd_config_from_experiment(self.cfg))
        return details, rows


def resolved_baseline_hyperparameters(cfg: ExperimentConfig, *, resolved_warmup_steps: int | None = None, resolved_lr_decay_steps: int | None = None, resolved_llrd_gamma: float | None = None) -> dict[str, object]:
    """Return the fully resolved nanoGPT baseline settings recorded in run metadata."""
    model = cfg.model
    train = cfg.train
    out: dict[str, object] = {
        "learning_rate": train.learning_rate,
        "weight_decay": train.weight_decay,
        "grad_clip": train.grad_clip,
        "adamw_betas": tuple(train.betas),
        "adamw_epsilon": train.epsilon,
        "lr_schedule": train.lr_schedule,
        "warmup_steps_requested": train.warmup_steps,
        "warmup_ratio": train.warmup_ratio,
        "lr_decay_steps_requested": train.lr_decay_steps,
        "min_lr_ratio": train.min_lr_ratio,
        "layer_lr": train.layer_lr,
        "llrd_gamma": resolved_llrd_gamma,
        "matrix_lr_multipliers": dict(train.matrix_lr_multipliers),
        "tie_weights": model.tie_weights,
        "init_mode": model.init_mode,
        "residual_projection_init_std": 0.02 / (2 * model.n_layer) ** 0.5,
        "causal_attention": True,
        "attention_implementation": "torch_scaled_dot_product_attention_is_causal",
        "attention_dropout": model.dropout,
        "residual_dropout": model.dropout,
        "embedding_dropout": model.dropout,
        "separate_qkv_projections": True,
        "linear_bias": model.linear_bias,
        "layernorm_bias": model.layernorm_bias,
        "model_architecture_version": model.model_architecture_version,
        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,
    }
    if resolved_warmup_steps is not None:
        out["resolved_warmup_steps"] = resolved_warmup_steps
    if resolved_lr_decay_steps is not None:
        out["resolved_lr_decay_steps"] = resolved_lr_decay_steps
    return out


def _gradient_norm(parameters) -> torch.Tensor:
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms), ord=2)

def _log_train_progress(message: str) -> None:
    print(f"[wwgpt run-multiseed] {message}", file=sys.stderr, flush=True)


def _write_csv(path: Path, rows: list[dict[str, object]], *, overwrite: bool = False) -> None:
    if not rows:
        return
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _append_only_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Append missing rows without rewriting existing raw metric logs."""
    if not rows:
        return
    existing = 0
    existing_fields: list[str] | None = None
    if path.exists():
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing = sum(1 for _ in reader)
    pending = rows[existing:]
    if not pending:
        return
    if existing_fields:
        fieldnames = existing_fields
    else:
        fieldnames = list(rows[0])
        extras = [k for r in pending for k in r if k not in fieldnames]
        fieldnames = fieldnames + list(dict.fromkeys(extras))
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if existing == 0 and not existing_fields:
            w.writeheader()
        w.writerows(pending)


def _perplexity_from_cross_entropy(loss: float) -> float:
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


def _metrics(loss: float, logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
    top1 = float((pred[..., 0] == y).float().mean())
    top5 = float((pred == y.unsqueeze(-1)).any(dim=-1).float().mean())
    return {
        "loss": loss,
        "perplexity": _perplexity_from_cross_entropy(loss),
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
        "perplexity": _perplexity_from_cross_entropy(mean_loss),
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
    elif optimizer_name in {"muon_wwpgd", "stableadamw_wwpgd", "stable_adamw_wwpgd"}:
        base_optimizer, extension_name = optimizer_name.removesuffix("_wwpgd"), "wwpgd"
    elif optimizer_name in {"adamw", "muon", "stableadamw", "stable_adamw"}:
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
            "resolved_baseline_hyperparameters": resolved_baseline_hyperparameters(cfg),
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
            proj_rows.extend(apply_external_wwpgd(model, event_index=step, actual_step=step, cfg=external_wwpgd_config_from_experiment(cfg.wwpgd)))
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
        wwpgd=WWPGDConfig(enabled=True, extension="wwpgd"),
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
    if optimizer_name in {"adamw_wwpgd_reference", "adamw_wwpgd"}:
        base_optimizer, extension_name = "adamw", "wwpgd"
    elif optimizer_name in {"muon_wwpgd", "stableadamw_wwpgd", "stable_adamw_wwpgd"}:
        base_optimizer, extension_name = optimizer_name.removesuffix("_wwpgd"), "wwpgd"
    elif optimizer_name in {"adamw", "muon", "stableadamw", "stable_adamw"}:
        base_optimizer, extension_name = optimizer_name, getattr(cfg.wwpgd, "extension", "none")
    else:
        base_optimizer, extension_name = optimizer_name, getattr(cfg.wwpgd, "extension", "none")
    optimizer_name = make_arm_name(base_optimizer, extension_name)
    run_dir = None
    ckpt = None
    resolved_seeds = resolved_stochastic_seeds(seed, level, token_multiplier, split="train", optimizer_identity=base_optimizer)
    selected_device = select_device(device)
    selected_device_summary = device_summary(device or "auto")
    _log_train_progress(f"device selection: {selected_device_summary['selection_reason']}; single_device_only={selected_device_summary['single_device_only']}")
    precision_info = precision_policy(selected_device, precision)
    model = GPT(cfg.model).to(selected_device)
    model.load_state_dict(init_state)
    torch.manual_seed(resolved_seeds["dropout_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seeds["dropout_seed"])
    bundle, resolved_llrd_gamma = build_optimizer_bundle(model, cfg.train, base_optimizer)
    report = model.parameter_report()
    from wwgpt.scaling import selected_parameter_count
    parameter_count_used = selected_parameter_count(report, cfg.parameter_count_convention)
    tokens_per_step = cfg.train.batch_size * cfg.model.block_size * cfg.train.gradient_accumulation
    if cfg.train.max_steps is not None:
        steps = cfg.train.max_steps; target_tokens = steps * tokens_per_step; budget_source = "max_steps"
    elif cfg.train.max_train_tokens is not None:
        target_tokens = cfg.train.max_train_tokens; steps = max(1, math.ceil(target_tokens / tokens_per_step)); budget_source = "max_train_tokens"
    else:
        target_tokens = parameter_count_used * token_multiplier; steps = max(1, math.ceil(target_tokens / tokens_per_step)); budget_source = "token_multiplier"
    realized_tokens = steps * tokens_per_step
    resolved_lr_decay_steps = resolve_lr_decay_steps(steps, cfg.train.lr_decay_steps)
    resolved_warmup_steps = resolve_warmup_steps(steps, cfg.train.warmup_ratio, cfg.train.warmup_steps, cfg.train.lr_decay_steps)
    wwpgd_interval = 1 if extension_name == "wwpgd" else int(cfg.train.wwpgd_interval or 1)
    extension = WWPGDExtension(cfg.wwpgd, wwpgd_interval) if extension_name == "wwpgd" else NoExtension()
    reader = (RandomWindowTokenReader(data.train, cfg.model.block_size, resolved_seeds["train_reader_seed"]) if cfg.train.training_sampling == "random_window" else NonRepeatingTokenReader(data.train, cfg.model.block_size))
    initial_minibatch_indices = _initial_minibatch_indices(data.train, cfg.model.block_size, cfg.train.batch_size, cfg.train.training_sampling, resolved_seeds["train_reader_seed"])
    validation_probe_hash = ""
    training_probe_hash = ""
    if data.data_manifest and data.data_manifest.get("storage_format") not in (None, "raw_memmap_v1"):
        raise RuntimeError("obsolete prepared-data format: rebuild with `wwgpt prepare-data` to create memmap token files")
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
        "selected_parameter_count": parameter_count_used,
        "realized_tokens_per_selected_parameter": realized_tokens / max(parameter_count_used, 1),
        "sequence_count": realized_tokens // cfg.model.block_size,
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
        "optimizer_fingerprint": json.loads(json.dumps(optimizer_fingerprint(bundle), default=str)),
        "extension_hyperparameters": asdict(cfg.wwpgd),
        "resolved_baseline_hyperparameters": resolved_baseline_hyperparameters(
            cfg,
            resolved_warmup_steps=resolved_warmup_steps,
            resolved_lr_decay_steps=resolved_lr_decay_steps,
            resolved_llrd_gamma=resolved_llrd_gamma,
        ),
        "training_schedule_hash": sha256_bytes(json.dumps({"seed": seed, "level": level, "token_multiplier": token_multiplier, "steps": steps, "batch": cfg.train.batch_size, "training_sampling": cfg.train.training_sampling}, sort_keys=True).encode()),
        "resolved_stochastic_seeds": resolved_seeds,
        "initial_minibatch_indices": initial_minibatch_indices,
        "training_sampling": cfg.train.training_sampling,
        "evaluation_sampling": cfg.train.evaluation_sampling,
        "evaluation_schedule_version": "random_per_eval_v1",
        "lr_schedule": cfg.train.lr_schedule,
        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,
        "layer_lr": cfg.train.layer_lr,
        "warmup_steps_requested": cfg.train.warmup_steps,
        "warmup_ratio": cfg.train.warmup_ratio,
        "resolved_warmup_steps": resolved_warmup_steps,
        "lr_decay_steps_requested": cfg.train.lr_decay_steps,
        "resolved_lr_decay_steps": resolved_lr_decay_steps,
        "min_lr_ratio": cfg.train.min_lr_ratio,
        "resolved_llrd_gamma": resolved_llrd_gamma,
        "llrd_min_multiplier": cfg.train.llrd_min_multiplier,
        "weight_decay": cfg.train.weight_decay,
        "grad_clip": cfg.train.grad_clip,
        "batch_size": cfg.train.batch_size,
        "gradient_accumulation": cfg.train.gradient_accumulation,
        "wwpgd_interval": wwpgd_interval,
        "projection_schedule_type": "optimizer_step_interval",
        "total_projection_events": steps if extension_name == "wwpgd" else 0,
        "optimizer_step_count": 0,
        "device": selected_device_summary,
        "device_support": {"single_device_only": True, "distributed_training": False, "multi_gpu_or_tpu": "not claimed; no executable distributed smoke path is implemented"},
        "precision_policy": {k: v for k, v in precision_info.items() if k != "torch_dtype"},
        "wwpgd_call_count": 0,
        "projected_matrix_count": 0,
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
        "weightwatcher_diagnostic_configuration": {"detX": True, "randomize": True, "plot": False},
        "weightwatcher_diagnostic_outputs": {"per_layer_long_form": "spectral.csv", "run_level_aggregates": "weightwatcher_aggregates.csv"},
    }
    man.update(external_wwpgd_manifest_fields(extension_name == "wwpgd", cfg.wwpgd if extension_name == "wwpgd" else None))
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
        "optimizer_fingerprint": man["optimizer_fingerprint"],
        "immediate_projection_spectral": immediate_projection_spectral,
    }
    if resume:
        run_dir = _select_resume_run(run_parent / optimizer_name, expected_identity)
        ckpt = run_dir / "checkpoints"
    else:
        run_dir = unique_dir(run_parent / optimizer_name, "run")
        ckpt = run_dir / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
    cfgd = json.loads(json.dumps(asdict(cfg)))
    if not resume:
        write_json(run_dir / "environment.json", environment())
        (run_dir / "initialization_hash.txt").write_text(init_hash)
        write_json(run_dir / "manifest.json", man)
        write_json(run_dir / "data_manifest.json", data.data_manifest)
        write_json(run_dir / "tokenizer_manifest.json", data.tokenizer_manifest)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfgd))
        write_json(run_dir / "config.json", cfgd)
    metric_rows = []
    spectral_rows = []
    spectral_aggregate_rows = []
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
    optimizer_step_count = 0
    wwpgd_call_count = 0
    projected_matrix_count = 0
    compatibility = _compatibility(cfg, data, init_hash, validation_probe_hash, training_probe_hash)
    compatibility.update({"optimizer_name": optimizer_name, "optimizer_class": type(bundle.optimizers[0]).__name__, "immediate_projection_spectral": immediate_projection_spectral, "weightwatcher_version": _ww_version(), "weightwatcher_configuration": {"detX": True, "randomize": False, "plot": False}, "wwpgd_commit": WWPGD_COMMIT if extension_name == "wwpgd" else "", "git_commit": man.get("git_commit", "unknown"), "seed": seed, "level": level, "token_multiplier": token_multiplier, "requested_tokens": target_tokens, "realized_tokens": realized_tokens, "optimizer_fingerprint": man["optimizer_fingerprint"]})
    _log_train_progress(
        f"starting run level={level} token_multiplier={token_multiplier} pair={pair_id} optimizer={optimizer_name} seed={seed} steps={steps} device={selected_device} output={run_dir}"
    )
    start_step = 1
    if resume:
        loaded = load_latest_checkpoint(run_dir)
        assert_checkpoint_compatible(loaded, compatibility)
        model.load_state_dict(loaded["model_state_dict"])
        bundle.load_state_dict(loaded.get("optimizer_state_dict", loaded.get("base_optimizer_state_dict")))
        if "training_reader_state" in loaded and hasattr(reader, "load_state_dict"):
            reader.load_state_dict(loaded["training_reader_state"])
        else:
            reader.pos = int(loaded["training_reader_position"])
        metric_rows = list(loaded.get("metrics_rows", []))
        spectral_rows = list(loaded.get("periodic_weightwatcher_rows", []))
        spectral_aggregate_rows = list(loaded.get("periodic_weightwatcher_aggregate_rows", []))
        proj_rows = list(loaded.get("wwpgd_projection_rows", []))
        immediate_spectral_rows = list(loaded.get("immediate_projection_weightwatcher_rows", []))
        lr_rows = list(loaded.get("lr_rows", []))
        composite_rows = list(loaded.get("composite_spectral_rows", []))
        best_validation_loss = float(loaded.get("best_validation_loss", best_validation_loss))
        best_validation_step = int(loaded.get("best_validation_step", best_validation_step))
        latest_validation_loss = float(loaded.get("latest_validation_loss", latest_validation_loss))
        completed_projection_event_indexes = list(loaded.get("completed_projection_event_indexes", []))
        next_projection_event_index = int(loaded.get("next_projection_event_index", len(completed_projection_event_indexes)))
        elapsed_prior = float(loaded.get("elapsed_training_time", 0.0))
        optimizer_step_count = int(loaded.get("optimizer_step_count", len(metric_rows)))
        wwpgd_call_count = int(loaded.get("wwpgd_call_count", len(completed_projection_event_indexes)))
        projected_matrix_count = int(loaded.get("projected_matrix_count", len(proj_rows)))
        restore_rng_state(loaded)
        start_step = int(loaded.get("next_step", int(loaded.get("current_step", 0)) + 1))
        _log_train_progress(f"resuming run pair={pair_id} optimizer={optimizer_name} seed={seed} from step={start_step} checkpoint={run_dir}")
    start = time.perf_counter()
    last_loss = latest_validation_loss if math.isfinite(latest_validation_loss) else 0.0
    ww_over = 0.0
    if not resume:
        optimizer_step_count = start_step - 1
        wwpgd_call_count = len(completed_projection_event_indexes)
        projected_matrix_count = len(proj_rows)
    for step in range(start_step, steps + 1):
        train_loss_value = 0.0
        for _ in range(cfg.train.gradient_accumulation):
            xb, yb = reader.next_batch(cfg.train.batch_size)
            x = torch.tensor(xb, device=selected_device)
            y = torch.tensor(yb, device=selected_device)
            with autocast_context(selected_device, precision):
                _, loss = model(x, y)
            assert loss is not None
            (loss / cfg.train.gradient_accumulation).backward()
            train_loss_value += float(loss.detach().cpu())
        grad_before_clip = _gradient_norm(model.parameters())
        if cfg.train.grad_clip > 0.0:
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            grad_after_clip = _gradient_norm(model.parameters())
        else:
            grad = grad_before_clip
            grad_after_clip = grad_before_clip
        lr_update_rows = apply_lr_schedule(bundle, step - 1, steps, resolved_warmup_steps, cfg.train)
        lr_rows.extend(lr_update_rows)
        logged_lr = float(lr_update_rows[0]["current_lr"]) if lr_update_rows else cfg.train.learning_rate
        for _opt in bundle.optimizers:
            optimizer_step(_opt, selected_device)
        synchronize_device(selected_device)
        optimizer_step_count = step
        loss = torch.tensor(train_loss_value / cfg.train.gradient_accumulation)
        ps = time.perf_counter()
        pre_details, new_proj = extension.after_optimizer_step(model=model, optimizer_step=step, total_optimizer_steps=steps, tokens_seen=step * tokens_per_step, collect_pre_details=immediate_projection_spectral)
        if extension_name == "wwpgd":
            wwpgd_call_count += 1
        projected_matrix_count += len(new_proj)
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
        bundle.zero_grad()
        if step % (eval_interval or cfg.train.eval_interval) == 0 or step == steps:
            eval_index = len(metric_rows)
            was_training = model.training
            model.eval()
            if cfg.train.evaluation_sampling == "fixed_probe":
                train_x, train_y, training_probe_hash = fixed_probe(data.train[cfg.train.batch_size * cfg.model.block_size * 2:], cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
                val_x, val_y, validation_probe_hash = fixed_probe(data.val, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
            else:
                train_x, train_y, training_probe_hash = random_probe(data.train, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches, stable_seed(resolved_seeds["train_eval_probe_seed_base"], eval_index, "random_per_eval_v1"))
                val_x, val_y, validation_probe_hash = random_probe(data.val, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches, stable_seed(resolved_seeds["val_eval_probe_seed_base"], eval_index, "random_per_eval_v1"))
            diagnostic_test_metrics = None
            diagnostic_test_loss = float("nan")
            diagnostic_test_probe_hash = ""
            with torch.no_grad():
                tm, _ = _evaluate_probe_batches(model, train_x, train_y, selected_device)
                vm, validation_probe_loss = _evaluate_probe_batches(model, val_x, val_y, selected_device)
                if cfg.train.test_evaluation_mode == "diagnostic_periodic" and data.test is not None:
                    test_x, test_y, diagnostic_test_probe_hash = random_probe(
                        data.test,
                        cfg.model.block_size,
                        cfg.train.batch_size,
                        cfg.train.eval_batches,
                        stable_seed(resolved_seeds["val_eval_probe_seed_base"], eval_index, "diagnostic_test_random_per_eval_v1"),
                    )
                    diagnostic_test_metrics, diagnostic_test_loss = _evaluate_probe_batches(model, test_x, test_y, selected_device)
            model.train(was_training)
            elapsed = elapsed_prior + time.perf_counter() - start
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
                    "optimizer_steps": optimizer_step_count,
                    "wall_clock_time": elapsed,
                    "learning_rate": logged_lr,
                    "gradient_norm": float(grad_before_clip.detach().cpu()),
                    "gradient_norm_before_clip": float(grad_before_clip.detach().cpu()),
                    "gradient_norm_after_clip": float(grad_after_clip.detach().cpu()),
                    "train_minibatch_loss": float(loss.detach().cpu()),
                    "train_loss": tm["loss"],
                    "train_cross_entropy": tm["loss"],
                    "validation_loss": vm["loss"],
                    "validation_cross_entropy": vm["loss"],
                    "val_loss": vm["loss"],
                    "test_loss": diagnostic_test_loss,
                    "test_cross_entropy": diagnostic_test_loss,
                    "train_perplexity": tm["perplexity"],
                    "validation_perplexity": vm["perplexity"],
                    "val_perplexity": vm["perplexity"],
                    "test_perplexity": diagnostic_test_metrics["perplexity"] if diagnostic_test_metrics else float("nan"),
                    "train_bits_per_token": tm["bits_per_token"],
                    "val_bits_per_token": vm["bits_per_token"],
                    "train_top1_accuracy": tm["top1_accuracy"],
                    "val_top1_accuracy": vm["top1_accuracy"],
                    "train_top5_accuracy": tm["top5_accuracy"],
                    "val_top5_accuracy": vm["top5_accuracy"],
                    "train_token_error": tm["token_error"],
                    "val_token_error": vm["token_error"],
                    "train_validation_gap": vm["loss"] - tm["loss"],
                    "train_test_gap": diagnostic_test_loss - tm["loss"] if diagnostic_test_metrics else float("nan"),
                    "generalization_gap": vm["loss"] - tm["loss"],
                    "evaluation_index": eval_index,
                    "evaluation_sampling": cfg.train.evaluation_sampling,
                    "train_eval_batch_hash": training_probe_hash,
                    "val_eval_batch_hash": validation_probe_hash,
                    "test_eval_batch_hash": diagnostic_test_probe_hash,
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
                        memory_stats(selected_device).get("max_allocated", 0.0)
                    ),
                }
            )
            ws = time.perf_counter()
            if step % (spectral_interval or cfg.train.spectral_interval) == 0 or step == steps:
                new_spectral_rows = spectral_summary(
                    model,
                    step=step,
                    tokens_seen=step * tokens_per_step,
                    optimizer=optimizer_name,
                    seed=seed,
                    pair_id=pair_id,
                )
                spectral_rows.extend(new_spectral_rows)
                spectral_aggregate_rows.extend(weightwatcher_run_aggregates(new_spectral_rows))
            ww_over += time.perf_counter() - ws
            _log_train_progress(
                f"progress pair={pair_id} optimizer={optimizer_name} seed={seed} step={step}/{steps} tokens={step * tokens_per_step}/{int(data.data_manifest['realized_tokens'])} train_loss={tm['loss']:.4f} val_loss={vm['loss']:.4f} elapsed_s={elapsed:.1f} tokens_per_s={(step * tokens_per_step) / max(elapsed, 1e-9):.1f}"
            )
        if cfg.composite_spectral_analysis_enabled and (step % (spectral_interval or cfg.train.spectral_interval) == 0 or step == steps):
            composite_rows.extend(composite_spectral_summary(model, step=step, tokens_seen=step * tokens_per_step, base_optimizer=base_optimizer, extension=extension_name, arm_name=optimizer_name, seed=seed, pair_id=pair_id))
        if step % (checkpoint_interval or cfg.train.checkpoint_interval) == 0:
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": bundle.state_dict(),
                "base_optimizer_state_dict": bundle.state_dict(),
                "scheduler_state_dict": None,
                "gradient_scaler_state_dict": None,
                "current_step": step,
                "next_step": step + 1,
                "optimizer_step_count": optimizer_step_count,
                "wwpgd_call_count": wwpgd_call_count,
                "projected_matrix_count": projected_matrix_count,
                "wwpgd_state": {"extension": extension_name, "call_count": wwpgd_call_count, "projected_matrix_count": projected_matrix_count, "completed_projection_event_indexes": list(completed_projection_event_indexes), "next_projection_event_index": next_projection_event_index, "projection_schedule": cfg.wwpgd.projection_schedule},
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
                "periodic_weightwatcher_aggregate_rows": spectral_aggregate_rows,
                "wwpgd_projection_rows": proj_rows,
                "immediate_projection_weightwatcher_rows": immediate_spectral_rows,
                "lr_rows": lr_rows,
                "composite_spectral_rows": composite_rows,
                "elapsed_training_time": elapsed_prior + time.perf_counter() - start,
                "initialization_hash": init_hash,
                "resolved_stochastic_seeds": resolved_seeds,
                "compatibility": compatibility,
                "resolved_config": cfgd,
                "optimizer_fingerprint": man["optimizer_fingerprint"],
                "data_hash": data.corpus_hash,
                "tokenizer_hash": data.tokenizer_manifest["tokenizer_hash"],
                "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
                "lr_schedule": cfg.train.lr_schedule, "scheduler_implementation": SCHEDULER_IMPLEMENTATION, "layer_lr": cfg.train.layer_lr, "warmup_steps_requested": cfg.train.warmup_steps, "warmup_ratio": cfg.train.warmup_ratio, "resolved_warmup_steps": resolved_warmup_steps, "lr_decay_steps_requested": cfg.train.lr_decay_steps, "resolved_lr_decay_steps": resolved_lr_decay_steps, "min_lr_ratio": cfg.train.min_lr_ratio,
                "weightwatcher_version": _ww_version(), "weightwatcher_configuration": {"detX": True, "randomize": False, "plot": False}, "wwpgd_commit": WWPGD_COMMIT if extension_name == "wwpgd" else "", "git_commit": man.get("git_commit", "unknown"), "optimizer_name": optimizer_name, "pair_id": pair_id, "level": level, "token_multiplier": token_multiplier, "realized_tokens": realized_tokens, "requested_tokens": target_tokens, "immediate_projection_spectral": immediate_projection_spectral, "run_directory": str(run_dir),
            }
            save_checkpoint(run_dir, state)
            _log_train_progress(
                f"checkpoint saved pair={pair_id} optimizer={optimizer_name} seed={seed} step={step}/{steps} dir={ckpt}"
            )
    if data.test is not None and metric_rows:
        final_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_paths = sorted(ckpt.glob(f"best_val_step_*_{seed}.pt"))
        selected_path = best_paths[-1] if best_paths else None
        if selected_path is not None:
            selected_state = torch.load(selected_path, map_location=selected_device, weights_only=False)
            model.load_state_dict(selected_state)
        was_training = model.training
        model.eval()
        test_x, test_y, test_probe_hash = fixed_probe(data.test, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
        with torch.no_grad():
            test_metrics, test_loss = _evaluate_probe_batches(model, test_x, test_y, selected_device)
        model.train(was_training)
        metric_rows[-1].update({
            "selected_checkpoint_step": best_validation_step or steps,
            "selected_checkpoint_metric": "validation_loss",
            "test_evaluation_mode": cfg.train.test_evaluation_mode,
            "test_eval_batch_hash": test_probe_hash,
            "test_loss": test_loss,
            "test_cross_entropy": test_loss,
            "test_perplexity": test_metrics["perplexity"],
            "test_top1_accuracy": test_metrics["top1_accuracy"],
            "test_top5_accuracy": test_metrics["top5_accuracy"],
            "test_token_error": test_metrics["token_error"],
            "train_test_gap": test_loss - metric_rows[-1]["train_loss"],
        })
        model.load_state_dict(final_model_state)
    final_elapsed = elapsed_prior + time.perf_counter() - start
    save_checkpoint(run_dir, {"model_state_dict": model.state_dict(), "optimizer_state_dict": bundle.state_dict(), "base_optimizer_state_dict": bundle.state_dict(), "scheduler_state_dict": None, "gradient_scaler_state_dict": None, "current_step": steps, "next_step": steps + 1, "optimizer_step_count": optimizer_step_count, "wwpgd_call_count": wwpgd_call_count, "projected_matrix_count": projected_matrix_count, "wwpgd_state": {"extension": extension_name, "call_count": wwpgd_call_count, "projected_matrix_count": projected_matrix_count, "completed_projection_event_indexes": list(completed_projection_event_indexes), "next_projection_event_index": next_projection_event_index, "projection_schedule": cfg.wwpgd.projection_schedule}, "tokens_processed": steps * tokens_per_step, "training_reader_position": reader.pos, "reader_position": reader.pos, "training_reader_state": reader.state_dict() if hasattr(reader, "state_dict") else {"pos": reader.pos}, "seed": seed, **rng_state(), "device_type": selected_device.type, "precision_policy": precision or "torch_default", "gradient_accumulation_position": 0, "best_validation_loss": best_validation_loss, "best_validation_step": best_validation_step, "latest_validation_loss": latest_validation_loss, "completed_projection_event_indexes": completed_projection_event_indexes, "next_projection_event_index": next_projection_event_index, "projection_schedule": cfg.wwpgd.projection_schedule, "metrics_rows": metric_rows, "periodic_weightwatcher_rows": spectral_rows, "periodic_weightwatcher_aggregate_rows": spectral_aggregate_rows, "wwpgd_projection_rows": proj_rows, "immediate_projection_weightwatcher_rows": immediate_spectral_rows, "lr_rows": lr_rows, "composite_spectral_rows": composite_rows, "elapsed_training_time": final_elapsed, "initialization_hash": init_hash, "resolved_stochastic_seeds": resolved_seeds, "compatibility": compatibility, "resolved_config": cfgd, "optimizer_fingerprint": man["optimizer_fingerprint"], "data_hash": data.corpus_hash, "tokenizer_hash": data.tokenizer_manifest["tokenizer_hash"], "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION, "lr_schedule": cfg.train.lr_schedule, "scheduler_implementation": SCHEDULER_IMPLEMENTATION, "layer_lr": cfg.train.layer_lr, "warmup_steps_requested": cfg.train.warmup_steps, "warmup_ratio": cfg.train.warmup_ratio, "resolved_warmup_steps": resolved_warmup_steps, "lr_decay_steps_requested": cfg.train.lr_decay_steps, "resolved_lr_decay_steps": resolved_lr_decay_steps, "min_lr_ratio": cfg.train.min_lr_ratio, "weightwatcher_version": _ww_version(), "weightwatcher_configuration": {"detX": True, "randomize": False, "plot": False}, "wwpgd_commit": WWPGD_COMMIT if extension_name == "wwpgd" else "", "git_commit": man.get("git_commit", "unknown"), "optimizer_name": optimizer_name, "pair_id": pair_id, "level": level, "token_multiplier": token_multiplier, "realized_tokens": realized_tokens, "requested_tokens": target_tokens, "immediate_projection_spectral": immediate_projection_spectral, "run_directory": str(run_dir)})
    torch.save(model.state_dict(), ckpt / f"final_step_{steps:06d}_{seed}.pt")
    _append_only_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "spectral.csv", spectral_rows, overwrite=resume)
    _write_csv(run_dir / "weightwatcher_aggregates.csv", spectral_aggregate_rows, overwrite=resume)
    if cfg.composite_spectral_analysis_enabled:
        _write_csv(run_dir / "composite_spectral.csv", composite_rows, overwrite=resume)
    _write_csv(run_dir / "lrs.csv", lr_rows, overwrite=resume)
    if extension_name == "wwpgd":
        _write_csv(run_dir / "wwpgd_projection.csv", proj_rows, overwrite=resume)
        if immediate_projection_spectral:
            _write_csv(run_dir / "wwpgd_projection_spectral.csv", immediate_spectral_rows, overwrite=resume)
    (run_dir / "events.jsonl").write_text(json.dumps({"event": "complete"}) + "\n")
    write_json(run_dir / "run_complete.json", {"step": steps, "final_val_loss": last_loss, "optimizer_step_count": optimizer_step_count, "wwpgd_call_count": wwpgd_call_count, "projected_matrix_count": projected_matrix_count})
    _log_train_progress(
        f"completed run pair={pair_id} optimizer={optimizer_name} seed={seed} steps={steps} final_val_loss={last_loss:.4f} output={run_dir}"
    )
    return run_dir


CANONICAL_TRIAL_ARMS = ("adamw", "adamw_wwpgd", "muon", "muon_wwpgd", "stable_adamw", "stable_adamw_wwpgd")
CANONICAL_TRIAL_PAIRS = {"adamw": "adamw_wwpgd", "muon": "muon_wwpgd", "stable_adamw": "stable_adamw_wwpgd"}
CANONICAL_TRIAL_BASES = ("adamw", "muon", "stable_adamw")

def _trial_manifest(pair_id: str, level: int, token_multiplier: int, seed: int, cfg: ExperimentConfig, data, init_hash: str) -> dict:
    cfgd = json.loads(json.dumps(asdict(cfg)))
    shared = {
        "trial_id": pair_id, "seed": seed, "level": level, "token_multiplier": token_multiplier,
        "model_config": asdict(cfg.model), "model_configuration_hash": stable_hash(cfgd.get("model", {})),
        "data_manifest": data.data_manifest, "data_hash": data.corpus_hash,
        "tokenizer_manifest": data.tokenizer_manifest, "tokenizer_hash": data.tokenizer_manifest.get("tokenizer_hash"),
        "initialization_hash": init_hash, "train": asdict(cfg.train), "token_budget": {"realized_tokens": data.data_manifest.get("realized_tokens"), "token_multiplier": token_multiplier},
    }
    arms = []
    for base in CANONICAL_TRIAL_BASES:
        for ext in ("none", "wwpgd"):
            arm = make_arm_name(base, ext)
            arms.append({"arm_name": arm, "base_optimizer": base, "extension": ext, "paired_with": CANONICAL_TRIAL_PAIRS.get(base) if ext == "none" else base, "learning_rate": cfg.train.learning_rate, "lr_schedule": cfg.train.lr_schedule, "scheduler_implementation": SCHEDULER_IMPLEMENTATION, "weight_decay": cfg.train.weight_decay, "initialization_hash": init_hash, "batch_order_seed": resolved_stochastic_seeds(seed, level, token_multiplier, optimizer_identity=base)["train_reader_seed"], "token_budget": shared["token_budget"]})
    return {"scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION, "immutable": True, "trial_id": pair_id, "shared": shared, "arms": arms, "pairs": [{"baseline": b, "wwpgd": w} for b, w in CANONICAL_TRIAL_PAIRS.items()]}

def run_canonical_trials(level: int, data_root: Path, results_root: Path, token_multiplier: int, seeds: list[int] | None = None, config_path: Path | None = None, device: str | None = None, ww_interval: int | None = None, eval_interval: int | None = None, checkpoint_interval: int | None = None, spectral_interval: int | None = None, precision: str | None = None, resume: bool = False, immediate_projection_spectral: bool = False, allow_code_version_mismatch: bool = False) -> Path:
    from wwgpt.data import load_prepared_scientific_data
    cfg = load_config(config_path, level)
    data = load_prepared_scientific_data(data_root, level, token_multiplier)
    exp_root = results_root / "experiments" / f"level_{level:02d}" / f"multiplier_{token_multiplier}"
    exp_root.mkdir(parents=True, exist_ok=True)
    for seed in (seeds or cfg.seeds):
        existing_trials = sorted(exp_root.glob(f"trial_{seed}*")) if resume else []
        trial = existing_trials[0] if existing_trials else unique_dir(exp_root, f"trial_{seed}")
        trial_id = trial.name
        init_dir = trial / "initial_state"; init_dir.mkdir(parents=True, exist_ok=True)
        if resume and (init_dir / "model.pt").exists():
            init_state = torch.load(init_dir / "model.pt", map_location="cpu", weights_only=False); init_hash = (init_dir / "initialization_hash.txt").read_text().strip()
        else:
            torch.manual_seed(resolved_stochastic_seeds(seed, level, token_multiplier)["model_init_seed"]); init_model = GPT(cfg.model); init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}; init_hash = _state_hash(init_state); torch.save(init_state, init_dir / "model.pt"); (init_dir / "initialization_hash.txt").write_text(init_hash)
        if not (resume and (trial / "trial_manifest.json").exists()): write_json(trial / "trial_manifest.json", _trial_manifest(trial_id, level, token_multiplier, seed, cfg, data, init_hash))
        for base in CANONICAL_TRIAL_BASES:
            for ext in ("none", "wwpgd"):
                arm_cfg = replace(cfg, wwpgd=replace(cfg.wwpgd, extension=ext, enabled=(ext == "wwpgd")))
                run_scientific_single(trial, make_arm_name(base, ext), seed, arm_cfg, data, trial_id, init_state, init_hash, level, token_multiplier, device, ww_interval, eval_interval, checkpoint_interval, spectral_interval, precision, resume, immediate_projection_spectral, allow_code_version_mismatch)
    return exp_root

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
    run_seeds = seeds or cfg.seeds
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
        init_dir = pair / "initial_state"
        if resume and (init_dir / "model.pt").exists():
            init_state = torch.load(init_dir / "model.pt", map_location="cpu", weights_only=False)
            init_hash = (init_dir / "initialization_hash.txt").read_text().strip()
        else:
            init_seed = resolved_stochastic_seeds(seed, level, token_multiplier)["model_init_seed"]
            torch.manual_seed(init_seed)
            init_model = GPT(cfg.model)
            init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
            init_hash = _state_hash(init_state)
        _log_train_progress(
            f"starting seed {seed_index}/{len(run_seeds)} seed={seed} pair={pair_id}"
        )
        init_dir.mkdir(exist_ok=True)
        if not (init_dir / "model.pt").exists():
            torch.save(init_state, init_dir / "model.pt")
        if not (init_dir / "initialization_hash.txt").exists():
            (init_dir / "initialization_hash.txt").write_text(init_hash)
        if not (resume and (pair / "pair_manifest.json").exists()):
            write_json(
                pair / "pair_manifest.json",
                {
                "pair_id": pair_id,
                "seed": seed,
                "level": level,
                "token_multiplier": token_multiplier,
                "initialization_hash": init_hash,
                "resolved_stochastic_seeds": resolved_stochastic_seeds(seed, level, token_multiplier),
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
