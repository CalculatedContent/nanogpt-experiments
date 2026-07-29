from __future__ import annotations

import argparse
import csv
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

from .config import load_config, roots
from .data import validate_prepared_data
from .model import GPT, GPTConfig
from .optim import make_optimizers

RUN_SCHEMA_VERSION = 2


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def learning_rate_at(step_index: int, training: dict[str, Any]) -> float:
    warmup_steps = int(training["warmup_steps"])
    max_steps = int(training["max_steps"])
    learning_rate = float(training["learning_rate"])
    minimum_learning_rate = float(training["min_lr"])
    if step_index < warmup_steps:
        return learning_rate * (step_index + 1) / max(1, warmup_steps)
    if step_index >= max_steps:
        return minimum_learning_rate
    ratio = (step_index - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return minimum_learning_rate + cosine * (
        learning_rate - minimum_learning_rate
    )


def _assert_data_matches_config(
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> None:
    data = config["data"]
    for key in (
        "dataset_name",
        "dataset_config",
        "dataset_split",
        "dataset_revision",
        "tokenizer",
        "dtype",
    ):
        if metadata.get(key) != data.get(key):
            raise ValueError(
                f"prepared data field {key!r} does not match the configured identity"
            )
    expected_splits = {
        "train": int(data["train_tokens"]),
        "val": int(data["val_tokens"]),
        "test": int(data["test_tokens"]),
    }
    if metadata.get("split_tokens") != expected_splits:
        raise ValueError(
            "prepared data split sizes do not match the configured token counts"
        )


def _sample_cpu_batch(
    data: np.ndarray,
    *,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    upper = len(data) - block_size - 1
    if upper <= 0:
        raise ValueError("prepared split is shorter than the configured block size")
    indices = torch.randint(upper, (batch_size,), generator=generator)
    x = torch.stack(
        [
            torch.from_numpy(
                np.asarray(data[int(index) : int(index) + block_size], dtype=np.int64)
            )
            for index in indices
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    data[int(index) + 1 : int(index) + 1 + block_size],
                    dtype=np.int64,
                )
            )
            for index in indices
        ]
    )
    return x, y


def _to_device(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def make_fixed_probe(
    data: np.ndarray,
    *,
    batch_size: int,
    block_size: int,
    batches: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        _sample_cpu_batch(
            data,
            batch_size=batch_size,
            block_size=block_size,
            generator=generator,
        )
        for _ in range(batches)
    ]


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    probe: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    correct = 0
    token_count = 0
    try:
        for cpu_batch in probe:
            x, y = _to_device(cpu_batch, device)
            logits, loss = model(x, y)
            if loss is None:
                raise RuntimeError("evaluation did not return a loss")
            count = y.numel()
            total_nll += float(loss.detach().cpu()) * count
            correct += int((logits.argmax(dim=-1) == y).sum().detach().cpu())
            token_count += count
    finally:
        model.train(was_training)
    loss_value = total_nll / max(token_count, 1)
    return {
        "loss": loss_value,
        "perplexity": math.exp(min(20.0, loss_value)),
        "accuracy": correct / max(token_count, 1),
        "bits_per_token": loss_value / math.log(2.0),
        "tokens": float(token_count),
    }


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms), ord=2)


def _weight_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squares = sum(
        float((parameter.detach().float() ** 2).sum().cpu())
        for parameter in parameters
    )
    return math.sqrt(squares)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


class _SpectralSnapshot(nn.Module):
    def __init__(self, model: GPT):
        super().__init__()
        layers: dict[str, nn.Linear] = {}
        cpu_rng = torch.get_rng_state()
        try:
            for name, source in model.spectral_layers():
                copied = nn.Linear(
                    source.in_features,
                    source.out_features,
                    bias=False,
                    device="cpu",
                )
                copied.weight.data.copy_(
                    source.weight.detach().to(device="cpu", dtype=torch.float32)
                )
                copied.weight.requires_grad_(False)
                layers[name] = copied
        finally:
            torch.set_rng_state(cpu_rng)
        self.layers = nn.ModuleDict(layers)


def _source_layer_from_row(row: dict[str, Any], names: list[str]) -> str:
    text = " ".join(
        str(row.get(key, "")) for key in ("longname", "name", "layer_id")
    )
    matches = [name for name in names if name in text]
    return matches[0] if len(matches) == 1 else ""


def weightwatch(
    model: GPT,
    output_dir: Path,
    *,
    step: int,
    randomize: bool,
) -> dict[str, Any]:
    try:
        import weightwatcher as ww
    except ImportError as exc:
        raise RuntimeError(
            "WeightWatcher is enabled but not installed; install the analysis extra"
        ) from exc

    started = time.perf_counter()
    snapshot = _SpectralSnapshot(model)
    source_names = [name for name, _ in model.spectral_layers()]
    try:
        details = ww.WeightWatcher(model=snapshot).analyze(
            randomize=bool(randomize)
        )
        details.insert(0, "step", int(step))
        rows = details.to_dict(orient="records")
        details.insert(
            1,
            "source_layer",
            [_source_layer_from_row(row, source_names) for row in rows],
        )
        path = output_dir / f"weightwatcher_step_{step:07d}.csv"
        details.to_csv(path, index=False)
        return {
            "step": step,
            "success": True,
            "rows": int(len(details)),
            "path": str(path),
            "elapsed_sec": time.perf_counter() - started,
        }
    except Exception as exc:  # WeightWatcher should not erase the training run.
        payload = {
            "step": step,
            "success": False,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "elapsed_sec": time.perf_counter() - started,
        }
        _atomic_json(
            output_dir / f"weightwatcher_error_step_{step:07d}.json",
            payload,
        )
        print(
            "[level0-train] WeightWatcher failed "
            f"step={step}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return payload
    finally:
        del snapshot


def _optimizer_manifest(optimizers: list[torch.optim.Optimizer]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for optimizer_index, optimizer in enumerate(optimizers):
        for group_index, group in enumerate(optimizer.param_groups):
            rows.append(
                {
                    "optimizer_index": optimizer_index,
                    "optimizer_class": type(optimizer).__name__,
                    "group_index": group_index,
                    "group_name": group.get("group_name", f"group_{group_index}"),
                    "parameter_count": sum(
                        parameter.numel() for parameter in group["params"]
                    ),
                    "initial_lr": float(group.get("initial_lr", group["lr"])),
                    "lr_multiplier": float(group.get("lr_multiplier", 1.0)),
                    "weight_decay": float(group.get("weight_decay", 0.0)),
                }
            )
    return rows


def _save_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizers: list[torch.optim.Optimizer],
    step: int,
    config: dict[str, Any],
    train_generator: torch.Generator,
    best_validation_loss: float,
    best_validation_step: int,
) -> None:
    torch.save(
        {
            "run_schema_version": RUN_SCHEMA_VERSION,
            "model": model.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "step": int(step),
            "config": config,
            "train_generator_state": train_generator.get_state(),
            "cpu_rng_state": torch.get_rng_state(),
            "best_validation_loss": float(best_validation_loss),
            "best_validation_step": int(best_validation_step),
        },
        path,
    )


def _prepare_run_directory(run: Path, *, overwrite: bool) -> None:
    if run.exists() and any(run.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"run directory already contains files: {run}; "
                "use --overwrite or choose a fresh results root"
            )
        shutil.rmtree(run)
    run.mkdir(parents=True, exist_ok=True)


def run_training(
    config: dict[str, Any],
    *,
    data_root: str | Path,
    results_root: str | Path,
    device: str = "auto",
    overwrite: bool = False,
) -> Path:
    training = config["training"]
    model_config = GPTConfig(**config["model"])
    seed = int(training["seed"])
    optimizer_name = str(training["optimizer"]).lower()
    selected_device = device_auto() if device == "auto" else torch.device(device)
    torch.set_float32_matmul_precision("high")
    seed_all(seed)

    data_path = Path(data_root)
    data_metadata = validate_prepared_data(data_path, verify_hashes=True)
    _assert_data_matches_config(data_metadata, config)
    if int(data_metadata["vocab_size"]) > model_config.vocab_size:
        raise ValueError("model vocabulary does not cover the prepared tokenizer")
    arrays = {
        split: np.memmap(
            data_path / f"{split}.bin",
            dtype=np.uint16,
            mode="r",
        )
        for split in ("train", "val", "test")
    }

    run = Path(results_root) / f"{optimizer_name}_seed_{seed}"
    _prepare_run_directory(run, overwrite=overwrite)
    raw_model = GPT(model_config).to(selected_device)
    optimizers = make_optimizers(raw_model, config)
    model: nn.Module = raw_model
    if bool(training.get("compile", False)):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile was requested but is unavailable")
        model = torch.compile(raw_model)

    train_generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    probe_batch_size = int(training.get("eval_batch_size", training["batch_size"]))
    train_probe = make_fixed_probe(
        arrays["train"],
        batch_size=probe_batch_size,
        block_size=model_config.block_size,
        batches=int(training["eval_batches"]),
        seed=1001,
    )
    validation_probe = make_fixed_probe(
        arrays["val"],
        batch_size=probe_batch_size,
        block_size=model_config.block_size,
        batches=int(training["eval_batches"]),
        seed=2001,
    )
    test_probe = make_fixed_probe(
        arrays["test"],
        batch_size=probe_batch_size,
        block_size=model_config.block_size,
        batches=int(training["test_eval_batches"]),
        seed=3001,
    )

    tokens_per_step = (
        int(training["batch_size"])
        * model_config.block_size
        * int(training["grad_accum_steps"])
    )
    max_steps = int(training["max_steps"])
    manifest = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "profile": "level0_gpt2_bpe_realistic_v2",
        "config": config,
        "device": str(selected_device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "parameter_count": raw_model.num_parameters(),
        "non_embedding_parameter_count": raw_model.num_parameters(
            exclude_position_embedding=True
        ),
        "data_root": str(data_path.resolve()),
        "data_metadata": data_metadata,
        "tokens_per_optimizer_step": tokens_per_step,
        "planned_optimizer_steps": max_steps,
        "planned_training_tokens": tokens_per_step * max_steps,
        "evaluation_protocol": {
            "train_and_validation": "fixed_probe_periodic",
            "test": "final_and_validation_selected_only",
            "training_rng_isolated_from_evaluation": True,
            "fixed_probe_seeds": {"train": 1001, "val": 2001, "test": 3001},
        },
        "optimizer_groups": _optimizer_manifest(optimizers),
        "accuracy_definition": "exact_top1_next_gpt2_bpe_token",
    }
    _atomic_json(run / "manifest.json", manifest)

    metric_fields = [
        "step",
        "tokens_seen",
        "elapsed_sec",
        "tokens_per_sec",
        "learning_rate",
        "min_group_learning_rate",
        "max_group_learning_rate",
        "train_loss",
        "train_perplexity",
        "train_accuracy",
        "train_bits_per_token",
        "val_loss",
        "val_perplexity",
        "val_accuracy",
        "val_bits_per_token",
        "val_generalization_gap",
        "grad_norm",
        "weight_norm",
    ]
    best_validation_loss = float("inf")
    best_validation_step = -1
    last_gradient_norm = 0.0
    weightwatcher_records: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    latest_metrics: dict[str, Any] = {}

    initial_scale = learning_rate_at(0, training) / float(training["learning_rate"])
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * initial_scale

    print(
        "[level0-train] starting "
        f"optimizer={optimizer_name} seed={seed} device={selected_device} "
        f"parameters={manifest['parameter_count']:,} steps={max_steps:,} "
        f"tokens_per_step={tokens_per_step:,} "
        f"planned_tokens={tokens_per_step * max_steps:,}",
        file=sys.stderr,
        flush=True,
    )

    with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()

        for step in range(max_steps + 1):
            if step == 0 or step % int(training["eval_interval"]) == 0 or step == max_steps:
                train_metrics = evaluate_probe(model, train_probe, selected_device)
                validation_metrics = evaluate_probe(
                    model,
                    validation_probe,
                    selected_device,
                )
                elapsed = time.perf_counter() - start_time
                group_lrs = [
                    float(group["lr"])
                    for optimizer in optimizers
                    for group in optimizer.param_groups
                ]
                current_lr = learning_rate_at(max(step - 1, 0), training)
                latest_metrics = {
                    "step": step,
                    "tokens_seen": step * tokens_per_step,
                    "elapsed_sec": elapsed,
                    "tokens_per_sec": (
                        step * tokens_per_step / max(elapsed, 1e-9)
                    ),
                    "learning_rate": current_lr,
                    "min_group_learning_rate": min(group_lrs),
                    "max_group_learning_rate": max(group_lrs),
                    "train_loss": train_metrics["loss"],
                    "train_perplexity": train_metrics["perplexity"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_bits_per_token": train_metrics["bits_per_token"],
                    "val_loss": validation_metrics["loss"],
                    "val_perplexity": validation_metrics["perplexity"],
                    "val_accuracy": validation_metrics["accuracy"],
                    "val_bits_per_token": validation_metrics["bits_per_token"],
                    "val_generalization_gap": (
                        validation_metrics["loss"] - train_metrics["loss"]
                    ),
                    "grad_norm": last_gradient_norm,
                    "weight_norm": _weight_norm(raw_model.parameters()),
                }
                writer.writerow(latest_metrics)
                handle.flush()

                if validation_metrics["loss"] < best_validation_loss:
                    best_validation_loss = validation_metrics["loss"]
                    best_validation_step = step
                    torch.save(
                        {
                            "model": raw_model.state_dict(),
                            "step": step,
                            "validation_metrics": validation_metrics,
                            "config": config,
                        },
                        run / "checkpoint_best.pt",
                    )

                remaining_steps = max_steps - step
                steps_per_second = step / max(elapsed, 1e-9)
                eta = (
                    remaining_steps / steps_per_second
                    if steps_per_second > 0
                    else None
                )
                eta_text = f"{eta:.1f}s" if eta is not None else "unknown"
                print(
                    "[level0-train] progress "
                    f"step={step:,}/{max_steps:,} "
                    f"tokens={step * tokens_per_step:,} "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={validation_metrics['loss']:.4f} "
                    f"val_ppl={validation_metrics['perplexity']:.2f} "
                    f"val_acc={100 * validation_metrics['accuracy']:.2f}% "
                    f"elapsed={elapsed:.1f}s eta={eta_text}",
                    file=sys.stderr,
                    flush=True,
                )

            weightwatcher_due = (
                bool(config["analysis"]["weightwatcher"])
                and (
                    step % int(config["analysis"]["weightwatcher_interval"]) == 0
                    or step == max_steps
                )
            )
            if weightwatcher_due:
                weightwatcher_records.append(
                    weightwatch(
                        raw_model,
                        run,
                        step=step,
                        randomize=bool(config["analysis"]["randomize"]),
                    )
                )

            if step == max_steps:
                break

            scheduled_base_lr = learning_rate_at(step, training)
            scale = scheduled_base_lr / float(training["learning_rate"])
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = float(group["initial_lr"]) * scale
                optimizer.zero_grad(set_to_none=True)

            model.train()
            for _ in range(int(training["grad_accum_steps"])):
                cpu_batch = _sample_cpu_batch(
                    arrays["train"],
                    batch_size=int(training["batch_size"]),
                    block_size=model_config.block_size,
                    generator=train_generator,
                )
                x, y = _to_device(cpu_batch, selected_device)
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("training did not return a loss")
                (loss / int(training["grad_accum_steps"])).backward()

            if float(training["grad_clip"]) > 0:
                gradient = torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(),
                    float(training["grad_clip"]),
                )
            else:
                gradient = _gradient_norm(raw_model.parameters())
            last_gradient_norm = float(gradient.detach().cpu())
            for optimizer in optimizers:
                optimizer.step()

            completed_step = step + 1
            if completed_step % int(training["checkpoint_interval"]) == 0:
                _save_checkpoint(
                    run / f"checkpoint_{completed_step:07d}.pt",
                    model=raw_model,
                    optimizers=optimizers,
                    step=completed_step,
                    config=config,
                    train_generator=train_generator,
                    best_validation_loss=best_validation_loss,
                    best_validation_step=best_validation_step,
                )

    _save_checkpoint(
        run / "checkpoint_final.pt",
        model=raw_model,
        optimizers=optimizers,
        step=max_steps,
        config=config,
        train_generator=train_generator,
        best_validation_loss=best_validation_loss,
        best_validation_step=best_validation_step,
    )

    final_test_metrics = evaluate_probe(model, test_probe, selected_device)
    final_metrics = {
        "step": max_steps,
        "validation_loss": float(latest_metrics["val_loss"]),
        "validation_perplexity": float(latest_metrics["val_perplexity"]),
        "validation_accuracy": float(latest_metrics["val_accuracy"]),
        "test_loss": final_test_metrics["loss"],
        "test_perplexity": final_test_metrics["perplexity"],
        "test_accuracy": final_test_metrics["accuracy"],
        "test_bits_per_token": final_test_metrics["bits_per_token"],
    }
    _atomic_json(run / "final_metrics.json", final_metrics)

    best_checkpoint = torch.load(
        run / "checkpoint_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    raw_model.load_state_dict(best_checkpoint["model"])
    selected_test_metrics = evaluate_probe(model, test_probe, selected_device)
    selected_metrics = {
        "selected_step": int(best_checkpoint["step"]),
        "validation_loss": float(
            best_checkpoint["validation_metrics"]["loss"]
        ),
        "validation_perplexity": float(
            best_checkpoint["validation_metrics"]["perplexity"]
        ),
        "validation_accuracy": float(
            best_checkpoint["validation_metrics"]["accuracy"]
        ),
        "test_loss": selected_test_metrics["loss"],
        "test_perplexity": selected_test_metrics["perplexity"],
        "test_accuracy": selected_test_metrics["accuracy"],
        "test_bits_per_token": selected_test_metrics["bits_per_token"],
    }
    _atomic_json(run / "selected_checkpoint_metrics.json", selected_metrics)

    completion = {
        "completed": True,
        "run_schema_version": RUN_SCHEMA_VERSION,
        "optimizer": optimizer_name,
        "seed": seed,
        "optimizer_steps": max_steps,
        "tokens_seen": max_steps * tokens_per_step,
        "best_validation_step": selected_metrics["selected_step"],
        "best_validation_loss": selected_metrics["validation_loss"],
        "final_validation_loss": final_metrics["validation_loss"],
        "selected_test_loss": selected_metrics["test_loss"],
        "final_test_loss": final_metrics["test_loss"],
        "validation_collapse_final_minus_selected": (
            final_metrics["validation_loss"] - selected_metrics["validation_loss"]
        ),
        "test_collapse_final_minus_selected": (
            final_metrics["test_loss"] - selected_metrics["test_loss"]
        ),
        "weightwatcher_measurements": len(weightwatcher_records),
        "weightwatcher_failures": sum(
            not bool(record.get("success")) for record in weightwatcher_records
        ),
        "elapsed_sec": time.perf_counter() - start_time,
    }
    _atomic_json(run / "run_complete.json", completion)
    print(
        "[level0-train] complete "
        f"run={run} selected_step={selected_metrics['selected_step']} "
        f"selected_test_loss={selected_metrics['test_loss']:.4f} "
        f"final_test_loss={final_metrics['test_loss']:.4f}",
        file=sys.stderr,
        flush=True,
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/level0.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    parser.add_argument("--optimizer", choices=["adamw", "muon"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.optimizer:
        config["training"]["optimizer"] = args.optimizer
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    resolved = roots()
    run = run_training(
        config,
        data_root=args.data_root or resolved["data"],
        results_root=args.results_root or resolved["results"],
        device=args.device,
        overwrite=args.overwrite,
    )
    print(run)


if __name__ == "__main__":
    main()
