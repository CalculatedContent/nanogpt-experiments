from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from .config import load_config, roots, validate_config
from .model import GPT, GPTConfig
from .optim import make_optimizers

PROTOCOL_VERSION = "isolated_level0_bpe_v2"
METRIC_FIELDS = [
    "step",
    "tokens_seen",
    "elapsed_sec",
    "learning_rate",
    "train_loss",
    "train_perplexity",
    "train_bits_per_token",
    "train_accuracy",
    "val_loss",
    "val_perplexity",
    "val_bits_per_token",
    "val_accuracy",
    "val_generalization_gap",
    "grad_norm",
    "weight_norm",
    "tokens_per_second",
]


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return device_auto()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def config_sha256(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def lr_at(update_index: int, training: dict[str, Any]) -> float:
    """Warmup followed by cosine decay, indexed by zero-based update number."""

    if update_index < training["warmup_steps"]:
        return training["learning_rate"] * (update_index + 1) / max(
            1, training["warmup_steps"]
        )
    if update_index >= training["max_steps"]:
        return training["min_lr"]
    ratio = (update_index - training["warmup_steps"]) / max(
        1,
        training["max_steps"] - training["warmup_steps"],
    )
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return training["min_lr"] + coefficient * (
        training["learning_rate"] - training["min_lr"]
    )


def _batch_from_starts(
    data: np.memmap,
    starts: Iterable[int],
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts_list = [int(value) for value in starts]
    x = np.stack(
        [np.asarray(data[start : start + block_size], dtype=np.int64) for start in starts_list]
    )
    y = np.stack(
        [
            np.asarray(
                data[start + 1 : start + 1 + block_size],
                dtype=np.int64,
            )
            for start in starts_list
        ]
    )
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def random_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    upper = len(data) - block_size - 1
    if upper <= 0:
        raise ValueError("token split is shorter than block_size + 1")
    starts = torch.randint(upper, (batch_size,), generator=generator).tolist()
    return _batch_from_starts(data, starts, block_size, device)


def fixed_eval_starts(
    data_length: int,
    *,
    batch_size: int,
    block_size: int,
    eval_batches: int,
    seed: int,
) -> np.ndarray:
    upper = data_length - block_size - 1
    if upper <= 0:
        raise ValueError("token split is shorter than block_size + 1")
    generator = np.random.default_rng(seed)
    return generator.integers(
        0,
        upper,
        size=(eval_batches, batch_size),
        endpoint=False,
        dtype=np.int64,
    )


@torch.no_grad()
def evaluate(
    model: GPT,
    data: np.memmap,
    starts: np.ndarray,
    block_size: int,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    for batch_starts in starts:
        x, y = _batch_from_starts(data, batch_starts, block_size, device)
        logits, loss = model(x, y)
        if loss is None:
            raise RuntimeError("evaluation did not produce a loss")
        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += y.numel()
    model.train(was_training)
    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "bits_per_token": mean_loss / math.log(2.0),
        "accuracy": correct / max(1, total),
    }


def weight_norm(model: nn.Module) -> float:
    return math.sqrt(
        sum(
            float((parameter.detach().float() ** 2).sum().cpu())
            for parameter in model.parameters()
        )
    )


class _MatrixProbe(nn.Module):
    def __init__(self, matrices: list[tuple[str, torch.Tensor]]):
        super().__init__()
        for name, matrix in matrices:
            layer = nn.Linear(matrix.shape[1], matrix.shape[0], bias=False)
            layer.weight = nn.Parameter(
                matrix.detach().float().cpu().clone(),
                requires_grad=False,
            )
            self.add_module(name, layer)


def run_weightwatcher(
    model: GPT,
    run_dir: Path,
    *,
    step: int,
    tokens_seen: int,
) -> tuple[bool, str]:
    try:
        import weightwatcher as ww
    except ImportError:
        return False, "WeightWatcher is not installed"

    matrices = list(model.spectral_matrices())
    probe = _MatrixProbe(matrices)
    try:
        details = ww.WeightWatcher(model=probe).analyze(randomize=False)
    except Exception as exc:  # diagnostic failure must not destroy training
        message = f"{type(exc).__name__}: {exc}"
        (run_dir / f"weightwatcher_error_step_{step:07d}.json").write_text(
            json.dumps({"step": step, "error": message}, indent=2),
            encoding="utf-8",
        )
        return False, message

    details.insert(0, "tokens_seen", tokens_seen)
    details.insert(0, "step", step)
    source_column = next(
        (name for name in ("longname", "name") if name in details.columns),
        None,
    )
    matrix_names = [name for name, _ in matrices]
    if source_column is not None:
        details["matrix_name"] = details[source_column].astype(str).map(
            lambda value: next(
                (name for name in matrix_names if name in value),
                value,
            )
        )
    else:
        details["matrix_name"] = "unknown"

    output = run_dir / f"weightwatcher_step_{step:07d}.csv"
    details.to_csv(output, index=False)
    return True, ""


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)



def _torch_load(path: Path, *, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _optimizer_state_dicts(
    optimizers: list[torch.optim.Optimizer],
) -> list[dict[str, Any]]:
    return [optimizer.state_dict() for optimizer in optimizers]


def _load_optimizer_state_dicts(
    optimizers: list[torch.optim.Optimizer],
    states: list[dict[str, Any]],
) -> None:
    if len(optimizers) != len(states):
        raise ValueError("checkpoint optimizer count does not match configuration")
    for optimizer, state in zip(optimizers, states, strict=True):
        optimizer.load_state_dict(state)


def _checkpoint_payload(
    *,
    model: GPT,
    optimizers: list[torch.optim.Optimizer],
    step: int,
    cfg: dict[str, Any],
    config_hash: str,
    train_generator: torch.Generator,
    metrics_rows: list[dict[str, Any]],
    best_validation_loss: float,
    best_validation_step: int,
    best_validation_row: dict[str, Any],
    elapsed_sec: float,
    base_lrs: list[float],
    weightwatcher_successes: int,
    weightwatcher_failures: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "config": cfg,
        "config_sha256": config_hash,
        "model": model.state_dict(),
        "optimizers": _optimizer_state_dicts(optimizers),
        "step": int(step),
        "train_generator_state": train_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "metrics_rows": metrics_rows,
        "best_validation_loss": float(best_validation_loss),
        "best_validation_step": int(best_validation_step),
        "best_validation_row": best_validation_row,
        "elapsed_sec": float(elapsed_sec),
        "base_lrs": base_lrs,
        "weightwatcher_successes": int(weightwatcher_successes),
        "weightwatcher_failures": int(weightwatcher_failures),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if (
        torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
    ):
        payload["mps_rng_state"] = torch.mps.get_rng_state()
    return payload


def _restore_rng_state(payload: dict[str, Any]) -> None:
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    if "cuda_rng_state_all" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    if (
        "mps_rng_state" in payload
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(payload["mps_rng_state"])


def _load_data(
    data_root: Path,
    cfg: dict[str, Any],
) -> tuple[dict[str, np.memmap], dict[str, Any]]:
    metadata_path = data_root / "meta.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"missing {metadata_path}; run level0-prepare-data first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("format_version", 0)) < 2:
        raise RuntimeError(
            "prepared data is the obsolete byte-token baseline; prepare the new "
            "GPT-2-BPE dataset under /tmp/nanogpt-level0-bpe/data"
        )
    if not str(metadata.get("tokenizer", "")).startswith("tiktoken:"):
        raise RuntimeError("Level 0 requires tiktoken BPE data")
    expected_vocab = int(cfg["model"]["vocab_size"])
    if int(metadata.get("model_vocab_size", -1)) != expected_vocab:
        raise RuntimeError(
            "prepared data model_vocab_size does not match the model configuration"
        )
    dtype = np.dtype(metadata["dtype"])
    arrays: dict[str, np.memmap] = {}
    for split in ("train", "val", "test"):
        path = data_root / f"{split}.bin"
        if not path.is_file():
            raise FileNotFoundError(f"missing prepared split: {path}")
        arrays[split] = np.memmap(path, dtype=dtype, mode="r")
        expected_tokens = int(metadata["split_tokens"][split])
        if len(arrays[split]) != expected_tokens:
            raise RuntimeError(
                f"{split}.bin has {len(arrays[split]):,} tokens; "
                f"metadata requires {expected_tokens:,}"
            )
    return arrays, metadata


def _log(run_dir: Path, message: str) -> None:
    line = f"[level0-train] {message}"
    print(line, file=sys.stderr, flush=True)
    with open(run_dir / "train.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _record_evaluation(
    *,
    model: GPT,
    arrays: dict[str, np.memmap],
    eval_starts: dict[str, np.ndarray],
    cfg: dict[str, Any],
    device: torch.device,
    step: int,
    tokens_seen: int,
    elapsed_sec: float,
    learning_rate: float,
    grad_norm: float,
) -> dict[str, Any]:
    block_size = cfg["model"]["block_size"]
    train_metrics = evaluate(
        model,
        arrays["train"],
        eval_starts["train"],
        block_size,
        device,
    )
    validation_metrics = evaluate(
        model,
        arrays["val"],
        eval_starts["val"],
        block_size,
        device,
    )
    return {
        "step": step,
        "tokens_seen": tokens_seen,
        "elapsed_sec": elapsed_sec,
        "learning_rate": learning_rate,
        "train_loss": train_metrics["loss"],
        "train_perplexity": train_metrics["perplexity"],
        "train_bits_per_token": train_metrics["bits_per_token"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": validation_metrics["loss"],
        "val_perplexity": validation_metrics["perplexity"],
        "val_bits_per_token": validation_metrics["bits_per_token"],
        "val_accuracy": validation_metrics["accuracy"],
        "val_generalization_gap": validation_metrics["loss"]
        - train_metrics["loss"],
        "grad_norm": grad_norm,
        "weight_norm": weight_norm(model),
        "tokens_per_second": tokens_seen / max(elapsed_sec, 1e-9),
    }


def _evaluate_test(
    model: GPT,
    arrays: dict[str, np.memmap],
    eval_starts: dict[str, np.ndarray],
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    return evaluate(
        model,
        arrays["test"],
        eval_starts["test"],
        cfg["model"]["block_size"],
        device,
    )


def run_experiment(
    cfg: dict[str, Any],
    *,
    data_root: Path,
    results_root: Path,
    device: torch.device,
    resume: bool = False,
    overwrite: bool = False,
    disable_weightwatcher: bool = False,
) -> Path:
    validate_config(cfg)
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")

    training = cfg["training"]
    analysis = cfg["analysis"]
    run_dir = results_root / f"{training['optimizer']}_seed_{training['seed']}"
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    complete_path = run_dir / "run_complete.json"
    if complete_path.exists():
        if resume:
            _log(run_dir, "run is already complete; nothing to resume")
            return run_dir
        raise FileExistsError(
            f"completed run already exists: {run_dir}; use --overwrite explicitly"
        )

    existing_files = [path for path in run_dir.iterdir() if path.name != "train.log"]
    if existing_files and not resume:
        raise FileExistsError(
            f"nonempty run directory exists: {run_dir}; use --resume or --overwrite"
        )

    arrays, data_metadata = _load_data(data_root, cfg)
    seed = int(training["seed"])
    seed_all(seed)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(stable_seed(seed, "training_windows"))

    model_config = GPTConfig(**cfg["model"])
    model = GPT(model_config).to(device)
    if training.get("compile", False):
        if device.type != "cuda":
            raise RuntimeError("torch.compile is only enabled for the CUDA preset")
        model = torch.compile(model)  # type: ignore[assignment]

    optimizers = make_optimizers(model, cfg, device_type=device.type)
    base_lrs = [
        float(group["lr"])
        for optimizer in optimizers
        for group in optimizer.param_groups
    ]
    config_hash = config_sha256(cfg)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tokens_per_step = (
        training["batch_size"]
        * cfg["model"]["block_size"]
        * training["grad_accum_steps"]
    )

    eval_starts = {
        split: fixed_eval_starts(
            len(arrays[split]),
            batch_size=training["batch_size"],
            block_size=cfg["model"]["block_size"],
            eval_batches=training["eval_batches"],
            seed=stable_seed(seed, f"fixed_{split}_probe"),
        )
        for split in ("train", "val", "test")
    }
    eval_probe_hashes = {
        split: hashlib.sha256(starts.tobytes()).hexdigest()
        for split, starts in eval_starts.items()
    }

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "config": cfg,
        "config_sha256": config_hash,
        "device": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "parameter_count": parameter_count,
        "model_config": asdict(model_config),
        "tokens_per_optimizer_step": tokens_per_step,
        "planned_train_tokens": tokens_per_step * training["max_steps"],
        "data_root": str(data_root.resolve()),
        "data_metadata": data_metadata,
        "fixed_probe_hashes": eval_probe_hashes,
        "test_policy": "final_and_validation_selected_checkpoint_only",
        "alpha_policy": "deterministic_nonrandomized_transformer_matrices",
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and resume:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("config_sha256") != config_hash:
            raise RuntimeError("resume configuration does not match the existing run")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    metrics_rows: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_validation_step = 0
    best_validation_row: dict[str, Any] = {}
    weightwatcher_successes = 0
    weightwatcher_failures = 0
    elapsed_prior = 0.0
    start_step = 1
    last_grad_norm = 0.0
    latest_path = run_dir / "checkpoint_latest.pt"

    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(
                f"resume requested but checkpoint is missing: {latest_path}"
            )
        checkpoint = _torch_load(latest_path, map_location=device)
        if checkpoint.get("config_sha256") != config_hash:
            raise RuntimeError("checkpoint configuration does not match")
        model.load_state_dict(checkpoint["model"])
        _load_optimizer_state_dicts(optimizers, checkpoint["optimizers"])
        train_generator.set_state(checkpoint["train_generator_state"])
        _restore_rng_state(checkpoint)
        metrics_rows = list(checkpoint.get("metrics_rows", []))
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_validation_step = int(checkpoint["best_validation_step"])
        best_validation_row = dict(checkpoint["best_validation_row"])
        elapsed_prior = float(checkpoint.get("elapsed_sec", 0.0))
        base_lrs = [float(value) for value in checkpoint.get("base_lrs", base_lrs)]
        weightwatcher_successes = int(
            checkpoint.get("weightwatcher_successes", 0)
        )
        weightwatcher_failures = int(
            checkpoint.get("weightwatcher_failures", 0)
        )
        start_step = int(checkpoint["step"]) + 1
        _log(run_dir, f"resuming from optimizer step {start_step - 1}")
    else:
        start_time = time.monotonic()
        initial_row = _record_evaluation(
            model=model,
            arrays=arrays,
            eval_starts=eval_starts,
            cfg=cfg,
            device=device,
            step=0,
            tokens_seen=0,
            elapsed_sec=0.0,
            learning_rate=0.0,
            grad_norm=0.0,
        )
        metrics_rows.append(initial_row)
        best_validation_loss = float(initial_row["val_loss"])
        best_validation_step = 0
        best_validation_row = dict(initial_row)
        _write_metrics(run_dir / "metrics.csv", metrics_rows)
        _atomic_torch_save(
            {
                "protocol_version": PROTOCOL_VERSION,
                "config_sha256": config_hash,
                "model": model.state_dict(),
                "step": 0,
                "validation_row": initial_row,
            },
            run_dir / "checkpoint_best.pt",
        )
        if analysis["weightwatcher"] and not disable_weightwatcher:
            success, error = run_weightwatcher(model, run_dir, step=0, tokens_seen=0)
            if success:
                weightwatcher_successes += 1
            else:
                weightwatcher_failures += 1
                _log(run_dir, f"WeightWatcher step 0 failed: {error}")
        elapsed_prior = time.monotonic() - start_time
        _atomic_torch_save(
            _checkpoint_payload(
                model=model,
                optimizers=optimizers,
                step=0,
                cfg=cfg,
                config_hash=config_hash,
                train_generator=train_generator,
                metrics_rows=metrics_rows,
                best_validation_loss=best_validation_loss,
                best_validation_step=best_validation_step,
                best_validation_row=best_validation_row,
                elapsed_sec=elapsed_prior,
                base_lrs=base_lrs,
                weightwatcher_successes=weightwatcher_successes,
                weightwatcher_failures=weightwatcher_failures,
            ),
            latest_path,
        )

    _log(
        run_dir,
        "starting "
        f"optimizer={training['optimizer']} seed={seed} device={device} "
        f"parameters={parameter_count:,} steps={training['max_steps']:,} "
        f"tokens_per_step={tokens_per_step:,} "
        f"planned_tokens={tokens_per_step * training['max_steps']:,}",
    )

    started_at = time.monotonic()
    for step in range(start_step, training["max_steps"] + 1):
        update_index = step - 1
        learning_rate = lr_at(update_index, training)
        scale = learning_rate / training["learning_rate"]
        group_index = 0
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = base_lrs[group_index] * scale
                group_index += 1
            optimizer.zero_grad(set_to_none=True)

        minibatch_loss = 0.0
        for _ in range(training["grad_accum_steps"]):
            x, y = random_batch(
                arrays["train"],
                training["batch_size"],
                cfg["model"]["block_size"],
                device,
                train_generator,
            )
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("training forward pass did not produce a loss")
            (loss / training["grad_accum_steps"]).backward()
            minibatch_loss += float(loss.detach().cpu()) / training[
                "grad_accum_steps"
            ]

        if training["grad_clip"] > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training["grad_clip"],
            )
            last_grad_norm = float(gradient_norm.detach().cpu())
        else:
            gradients = [
                parameter.grad.detach().float().norm()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            last_grad_norm = (
                float(torch.linalg.vector_norm(torch.stack(gradients)).cpu())
                if gradients
                else 0.0
            )

        for optimizer in optimizers:
            optimizer.step()
        synchronize(device)

        elapsed = elapsed_prior + time.monotonic() - started_at
        tokens_seen = step * tokens_per_step
        if step % training["log_interval"] == 0 or step == 1:
            rate = tokens_seen / max(elapsed, 1e-9)
            remaining_steps = training["max_steps"] - step
            eta = remaining_steps * tokens_per_step / max(rate, 1e-9)
            _log(
                run_dir,
                f"step={step:,}/{training['max_steps']:,} "
                f"loss={minibatch_loss:.4f} lr={learning_rate:.3e} "
                f"grad={last_grad_norm:.3f} tok/s={rate:,.0f} "
                f"eta_min={eta / 60:.1f}",
            )

        evaluation_due = (
            step % training["eval_interval"] == 0
            or step == training["max_steps"]
        )
        if evaluation_due:
            row = _record_evaluation(
                model=model,
                arrays=arrays,
                eval_starts=eval_starts,
                cfg=cfg,
                device=device,
                step=step,
                tokens_seen=tokens_seen,
                elapsed_sec=elapsed,
                learning_rate=learning_rate,
                grad_norm=last_grad_norm,
            )
            metrics_rows.append(row)
            _write_metrics(run_dir / "metrics.csv", metrics_rows)
            _log(
                run_dir,
                f"eval step={step:,} train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} "
                f"val_ppl={row['val_perplexity']:.2f} "
                f"val_acc={100 * row['val_accuracy']:.2f}%",
            )
            if float(row["val_loss"]) < best_validation_loss:
                best_validation_loss = float(row["val_loss"])
                best_validation_step = step
                best_validation_row = dict(row)
                _atomic_torch_save(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "config_sha256": config_hash,
                        "model": model.state_dict(),
                        "step": step,
                        "validation_row": row,
                    },
                    run_dir / "checkpoint_best.pt",
                )

        weightwatcher_due = (
            analysis["weightwatcher"]
            and not disable_weightwatcher
            and (
                step % analysis["weightwatcher_interval"] == 0
                or step == training["max_steps"]
            )
        )
        if weightwatcher_due:
            success, error = run_weightwatcher(
                model,
                run_dir,
                step=step,
                tokens_seen=tokens_seen,
            )
            if success:
                weightwatcher_successes += 1
            else:
                weightwatcher_failures += 1
                _log(run_dir, f"WeightWatcher step {step} failed: {error}")

        checkpoint_due = (
            step % training["checkpoint_interval"] == 0
            or step == training["max_steps"]
        )
        if checkpoint_due:
            payload = _checkpoint_payload(
                model=model,
                optimizers=optimizers,
                step=step,
                cfg=cfg,
                config_hash=config_hash,
                train_generator=train_generator,
                metrics_rows=metrics_rows,
                best_validation_loss=best_validation_loss,
                best_validation_step=best_validation_step,
                best_validation_row=best_validation_row,
                elapsed_sec=elapsed,
                base_lrs=base_lrs,
                weightwatcher_successes=weightwatcher_successes,
                weightwatcher_failures=weightwatcher_failures,
            )
            _atomic_torch_save(payload, latest_path)
            _atomic_torch_save(
                payload,
                run_dir / f"checkpoint_{step:07d}.pt",
            )

    final_step = training["max_steps"]
    final_tokens = final_step * tokens_per_step
    final_test = _evaluate_test(model, arrays, eval_starts, cfg, device)
    final_validation_row = metrics_rows[-1]
    final_metrics = {
        "checkpoint": "final",
        "step": final_step,
        "tokens_seen": final_tokens,
        "train_loss": final_validation_row["train_loss"],
        "validation_loss": final_validation_row["val_loss"],
        "test_loss": final_test["loss"],
        "test_perplexity": final_test["perplexity"],
        "test_bits_per_token": final_test["bits_per_token"],
        "test_accuracy": final_test["accuracy"],
        "test_generalization_gap": final_test["loss"]
        - final_validation_row["train_loss"],
    }
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    best_checkpoint = _torch_load(
        run_dir / "checkpoint_best.pt", map_location=device
    )
    if best_validation_step == final_step:
        selected_test = final_test
    else:
        selected_model = GPT(model_config).to(device)
        selected_model.load_state_dict(best_checkpoint["model"])
        selected_test = _evaluate_test(
            selected_model,
            arrays,
            eval_starts,
            cfg,
            device,
        )
        del selected_model
    selected_metrics = {
        "checkpoint": "validation_selected",
        "selected_step": best_validation_step,
        "selected_tokens_seen": best_validation_step * tokens_per_step,
        "train_loss": best_validation_row["train_loss"],
        "validation_loss": best_validation_row["val_loss"],
        "test_loss": selected_test["loss"],
        "test_perplexity": selected_test["perplexity"],
        "test_bits_per_token": selected_test["bits_per_token"],
        "test_accuracy": selected_test["accuracy"],
        "test_generalization_gap": selected_test["loss"]
        - best_validation_row["train_loss"],
    }
    (run_dir / "selected_checkpoint_metrics.json").write_text(
        json.dumps(selected_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    final_checkpoint = {
        "protocol_version": PROTOCOL_VERSION,
        "config_sha256": config_hash,
        "model": model.state_dict(),
        "step": final_step,
        "final_metrics": final_metrics,
    }
    _atomic_torch_save(final_checkpoint, run_dir / "checkpoint_final.pt")

    completion = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "optimizer": training["optimizer"],
        "seed": seed,
        "final_step": final_step,
        "tokens_seen": final_tokens,
        "best_validation_step": best_validation_step,
        "best_validation_loss": best_validation_loss,
        "final_validation_loss": final_validation_row["val_loss"],
        "final_test_loss": final_test["loss"],
        "selected_checkpoint_test_loss": selected_test["loss"],
        "weightwatcher_successes": weightwatcher_successes,
        "weightwatcher_failures": weightwatcher_failures,
        "test_evaluation_policy": "final_and_validation_selected_checkpoint_only",
    }
    complete_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _log(
        run_dir,
        f"complete final_val={final_validation_row['val_loss']:.4f} "
        f"final_test={final_test['loss']:.4f} "
        f"selected_step={best_validation_step:,} "
        f"selected_test={selected_test['loss']:.4f}",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the isolated realistic Level 0 nanoGPT baseline"
    )
    parser.add_argument("--config", default="configs/level0.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    parser.add_argument("--optimizer", choices=["adamw", "muon"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-weightwatcher", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.optimizer:
        cfg["training"]["optimizer"] = args.optimizer
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    validate_config(cfg)

    resolved = roots()
    data_root = Path(args.data_root or resolved["data"])
    results_root = Path(args.results_root or resolved["results"])
    results_root.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    run_dir = run_experiment(
        cfg,
        data_root=data_root,
        results_root=results_root,
        device=device,
        resume=args.resume,
        overwrite=args.overwrite,
        disable_weightwatcher=args.no_weightwatcher,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
