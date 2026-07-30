from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import load_config, roots
from .model import GPT, GPTConfig, transformer_matrix_items
from .optim import make_optimizers
from .wwpgd_extension import WWPGDExtension, append_projection_rows


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


def learning_rate_at(update_index: int, training: dict[str, Any]) -> float:
    warmup_steps = int(training["warmup_steps"])
    max_steps = int(training["max_steps"])
    peak = float(training["learning_rate"])
    minimum = float(training["min_lr"])
    if update_index < warmup_steps:
        return peak * (update_index + 1) / max(1, warmup_steps)
    progress = (update_index - warmup_steps) / max(
        1, max_steps - warmup_steps - 1
    )
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + cosine * (peak - minimum)


def random_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("data split is too short for the configured block size")
    starts = torch.randint(
        len(data) - block_size - 1,
        (batch_size,),
        generator=generator,
    ).tolist()
    x = torch.stack(
        [
            torch.from_numpy(
                np.asarray(data[start : start + block_size], dtype=np.int64)
            )
            for start in starts
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    data[start + 1 : start + 1 + block_size], dtype=np.int64
                )
            )
            for start in starts
        ]
    )
    return x, y


def fixed_probe(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    n_batches: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        random_batch(data, batch_size, block_size, generator)
        for _ in range(n_batches)
    ]


@torch.inference_mode()
def evaluate_probe(
    model: GPT,
    probe: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    for x_cpu, y_cpu in probe:
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        logits, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(-1) == y).sum().detach().cpu())
        total += y.numel()
    model.train(was_training)
    mean_loss = float(np.mean(losses))
    return mean_loss, math.exp(min(20.0, mean_loss)), correct / max(total, 1)


def gradient_norm(parameters) -> torch.Tensor:
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms), ord=2)


def model_weight_norm(model: nn.Module) -> float:
    return math.sqrt(
        sum(float((parameter.detach().float() ** 2).sum()) for parameter in model.parameters())
    )


class _WeightMatrixHolder(nn.Module):
    def __init__(self, model: GPT):
        super().__init__()
        self.matrix_metadata: list[dict[str, object]] = []
        for name, matrix_type, block_index, weight in transformer_matrix_items(model):
            layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            layer.weight = nn.Parameter(
                weight.detach().float().cpu().clone(), requires_grad=False
            )
            self.add_module(name, layer)
            self.matrix_metadata.append(
                {
                    "matrix_name": name,
                    "matrix_type": matrix_type,
                    "block": block_index,
                }
            )


def _attach_matrix_metadata(frame, metadata: list[dict[str, object]]):
    import pandas as pd

    result = frame.copy().reset_index(drop=True)
    names = [str(item["matrix_name"]) for item in metadata]
    resolved: list[str | None] = [None] * len(result)
    for row_index, row in result.iterrows():
        text = " ".join(
            str(row.get(column, "")) for column in ("longname", "name")
        )
        for name in names:
            if name in text:
                resolved[row_index] = name
                break
    if any(name is None for name in resolved) and len(result) == len(metadata):
        order = list(range(len(result)))
        if "layer_id" in result.columns:
            numeric = pd.to_numeric(result["layer_id"], errors="coerce")
            if numeric.notna().all():
                order = list(numeric.sort_values().index)
        for metadata_index, row_index in enumerate(order):
            resolved[row_index] = names[metadata_index]
    if any(name is None for name in resolved):
        raise RuntimeError(
            "WeightWatcher rows could not be matched to all transformer matrices"
        )
    by_name = {str(item["matrix_name"]): item for item in metadata}
    result.insert(0, "matrix_name", resolved)
    result.insert(
        1,
        "matrix_type",
        [by_name[str(name)]["matrix_type"] for name in resolved],
    )
    result.insert(2, "block", [by_name[str(name)]["block"] for name in resolved])
    return result


def weightwatch(model: GPT, out: Path, step: int, randomize: bool) -> Path:
    try:
        import pandas as pd
        import weightwatcher as ww
    except ImportError as exc:
        raise RuntimeError(
            "WeightWatcher is required; install with pip install -e '.[analysis]'"
        ) from exc

    holder = _WeightMatrixHolder(model)
    frame = ww.WeightWatcher(model=holder).analyze(
        randomize=bool(randomize), plot=False
    )
    if frame is None or len(frame) == 0:
        raise RuntimeError("WeightWatcher returned no transformer-matrix rows")
    frame = _attach_matrix_metadata(frame, holder.matrix_metadata)
    frame.insert(0, "step", int(step))
    frame.insert(1, "tokens_seen", 0)
    path = out / f"weightwatcher_step_{step:07d}.csv"
    frame.to_csv(path, index=False)

    if "alpha" in frame.columns:
        alpha = pd.to_numeric(frame["alpha"], errors="coerce")
        finite = alpha[np.isfinite(alpha)]
        if len(finite):
            print(
                "[level0-train] WeightWatcher "
                f"step={step} matrices={len(frame)} median_alpha={finite.median():.3f}",
                flush=True,
            )
    return path


def _load_data(data_root: Path, model_cfg: GPTConfig):
    metadata_path = data_root / "meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"missing {metadata_path}; run level0-prepare-data first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("tokenizer") != "gpt2":
        raise RuntimeError(
            "incompatible old Level 0 data: expected GPT-2 BPE data. "
            "Prepare a fresh /tmp/nanogpt-level0-bpe data root."
        )
    if int(metadata.get("vocab_size", -1)) != model_cfg.vocab_size:
        raise RuntimeError(
            "data/model vocabulary mismatch: "
            f"data={metadata.get('vocab_size')} model={model_cfg.vocab_size}"
        )
    dtype = np.dtype(str(metadata.get("dtype", "uint16")))
    arrays = {}
    for split in ("train", "val", "test"):
        path = data_root / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(f"missing prepared split: {path}")
        arrays[split] = np.memmap(path, dtype=dtype, mode="r")
        if len(arrays[split]) <= model_cfg.block_size + 1:
            raise RuntimeError(f"{split} split is too short")
    return metadata, arrays


def _save_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizers,
    step: int,
    cfg: dict[str, Any],
    best_validation_loss: float,
    best_validation_step: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "step": int(step),
            "config": cfg,
            "best_validation_loss": float(best_validation_loss),
            "best_validation_step": int(best_validation_step),
        },
        path,
    )


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

    cfg = load_config(args.config)
    training = cfg["training"]
    wwpgd_config = cfg["wwpgd"]
    if args.optimizer:
        training["optimizer"] = args.optimizer
    if args.seed is not None:
        training["seed"] = args.seed

    resolved = roots()
    data_root = Path(args.data_root or resolved["data"])
    results_root = Path(args.results_root or resolved["results"])
    run = results_root / f"{training['optimizer']}_wwpgd_seed_{training['seed']}"
    if run.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"run directory already exists: {run}; use a fresh results root "
                "or pass --overwrite"
            )
        shutil.rmtree(run)
    run.mkdir(parents=True, exist_ok=False)

    device = device_auto() if args.device == "auto" else torch.device(args.device)
    seed = int(training["seed"])
    seed_all(seed)
    train_generator = torch.Generator(device="cpu").manual_seed(seed + 11)

    model_cfg = GPTConfig(**cfg["model"])
    data_metadata, arrays = _load_data(data_root, model_cfg)
    model = GPT(model_cfg).to(device)
    optimizers = make_optimizers(model, cfg)
    extension = WWPGDExtension(model, wwpgd_config)
    base_lrs = [
        float(group["lr"])
        for optimizer in optimizers
        for group in optimizer.param_groups
    ]

    train_probe = fixed_probe(
        arrays["train"],
        training["batch_size"],
        model_cfg.block_size,
        training["eval_batches"],
        seed + 1_001,
    )
    val_probe = fixed_probe(
        arrays["val"],
        training["batch_size"],
        model_cfg.block_size,
        training["eval_batches"],
        seed + 2_001,
    )

    tokens_per_step = (
        int(training["batch_size"])
        * model_cfg.block_size
        * int(training["grad_accum_steps"])
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    matrix_parameters = sum(
        weight.numel() for _, _, _, weight in transformer_matrix_items(model)
    )
    manifest = {
        "schema_version": 2,
        "config": cfg,
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "parameter_count": total_parameters,
        "transformer_matrix_parameter_count": matrix_parameters,
        "data_root": str(data_root.resolve()),
        "data_manifest": data_metadata,
        "tokens_per_optimizer_step": tokens_per_step,
        "effective_batch_sequences": int(training["batch_size"])
        * int(training["grad_accum_steps"]),
        "planned_training_tokens": tokens_per_step * int(training["max_steps"]),
        "test_evaluation_policy": "final_and_validation_selected_checkpoint_only",
        "evaluation_sampling": "fixed_probe_with_independent_rng_streams",
        "layer_learning_rate_policy": "flat",
        "arm_name": f"{training['optimizer']}_wwpgd",
        **extension.manifest_fields(),
    }
    (run / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    fields = [
        "step",
        "tokens_seen",
        "elapsed_sec",
        "learning_rate",
        "train_loss",
        "train_perplexity",
        "train_accuracy",
        "val_loss",
        "val_perplexity",
        "val_accuracy",
        "test_loss",
        "test_perplexity",
        "test_accuracy",
        "val_generalization_gap",
        "test_generalization_gap",
        "grad_norm",
        "weight_norm",
    ]
    metrics_path = run / "metrics.csv"
    best_validation_loss = float("inf")
    best_validation_step = 0
    best_checkpoint = run / "checkpoint_best.pt"
    last_grad_norm = float("nan")
    last_lr = 0.0
    started_at = time.time()
    projection_path = run / "wwpgd_projection.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for completed_steps in range(int(training["max_steps"]) + 1):
            evaluation_due = (
                completed_steps % int(training["eval_interval"]) == 0
                or completed_steps == int(training["max_steps"])
            )
            if evaluation_due:
                train_metrics = evaluate_probe(model, train_probe, device)
                val_metrics = evaluate_probe(model, val_probe, device)
                if val_metrics[0] < best_validation_loss:
                    best_validation_loss = val_metrics[0]
                    best_validation_step = completed_steps
                    _save_checkpoint(
                        best_checkpoint,
                        model=model,
                        optimizers=optimizers,
                        step=completed_steps,
                        cfg=cfg,
                        best_validation_loss=best_validation_loss,
                        best_validation_step=best_validation_step,
                    )

                test_metrics = (float("nan"), float("nan"), float("nan"))
                if completed_steps == int(training["max_steps"]):
                    test_probe = fixed_probe(
                        arrays["test"],
                        training["batch_size"],
                        model_cfg.block_size,
                        training["eval_batches"],
                        seed + 3_001,
                    )
                    test_metrics = evaluate_probe(model, test_probe, device)

                elapsed = time.time() - started_at
                writer.writerow(
                    {
                        "step": completed_steps,
                        "tokens_seen": completed_steps * tokens_per_step,
                        "elapsed_sec": elapsed,
                        "learning_rate": last_lr,
                        "train_loss": train_metrics[0],
                        "train_perplexity": train_metrics[1],
                        "train_accuracy": train_metrics[2],
                        "val_loss": val_metrics[0],
                        "val_perplexity": val_metrics[1],
                        "val_accuracy": val_metrics[2],
                        "test_loss": test_metrics[0],
                        "test_perplexity": test_metrics[1],
                        "test_accuracy": test_metrics[2],
                        "val_generalization_gap": val_metrics[0]
                        - train_metrics[0],
                        "test_generalization_gap": test_metrics[0]
                        - train_metrics[0],
                        "grad_norm": last_grad_norm,
                        "weight_norm": model_weight_norm(model),
                    }
                )
                handle.flush()
                rate = completed_steps / max(elapsed, 1e-9)
                remaining = int(training["max_steps"]) - completed_steps
                eta_seconds = remaining / rate if rate > 0 else float("nan")
                eta_text = (
                    f"{eta_seconds / 60:.1f}m" if math.isfinite(eta_seconds) else "unknown"
                )
                print(
                    "[level0-train] "
                    f"step={completed_steps}/{training['max_steps']} "
                    f"tokens={completed_steps * tokens_per_step:,} "
                    f"lr={last_lr:.3e} train_loss={train_metrics[0]:.4f} "
                    f"val_loss={val_metrics[0]:.4f} val_ppl={val_metrics[1]:.2f} "
                    f"val_acc={100 * val_metrics[2]:.2f}% eta={eta_text}",
                    flush=True,
                )
                if (
                    cfg["analysis"]["weightwatcher"]
                    and completed_steps
                    % int(cfg["analysis"]["weightwatcher_interval"])
                    == 0
                ):
                    path = weightwatch(
                        model,
                        run,
                        completed_steps,
                        bool(cfg["analysis"]["randomize"]),
                    )
                    frame = __import__("pandas").read_csv(path)
                    frame["tokens_seen"] = completed_steps * tokens_per_step
                    frame.to_csv(path, index=False)

            if completed_steps == int(training["max_steps"]):
                break

            update_index = completed_steps
            current_lr = learning_rate_at(update_index, training)
            scale = current_lr / float(training["learning_rate"])
            group_index = 0
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = base_lrs[group_index] * scale
                    group_index += 1
            last_lr = current_lr

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            for _ in range(int(training["grad_accum_steps"])):
                x_cpu, y_cpu = random_batch(
                    arrays["train"],
                    int(training["batch_size"]),
                    model_cfg.block_size,
                    train_generator,
                )
                x = x_cpu.to(device)
                y = y_cpu.to(device)
                _, loss = model(x, y)
                assert loss is not None
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"nonfinite training loss at step {completed_steps + 1}"
                    )
                (loss / int(training["grad_accum_steps"])).backward()

            unclipped_norm = gradient_norm(model.parameters())
            if not torch.isfinite(unclipped_norm):
                raise FloatingPointError(
                    f"nonfinite gradient norm at step {completed_steps + 1}"
                )
            if float(training["grad_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["grad_clip"])
                )
            last_grad_norm = float(unclipped_norm.detach().cpu())
            for optimizer in optimizers:
                optimizer.step()

            next_step = completed_steps + 1
            projection_rows = extension.after_optimizer_step(
                optimizer_step=next_step,
                total_optimizer_steps=int(training["max_steps"]),
                tokens_seen=next_step * tokens_per_step,
            )
            append_projection_rows(projection_path, projection_rows)
            if next_step % int(training["checkpoint_interval"]) == 0:
                _save_checkpoint(
                    run / f"checkpoint_{next_step:07d}.pt",
                    model=model,
                    optimizers=optimizers,
                    step=next_step,
                    cfg=cfg,
                    best_validation_loss=best_validation_loss,
                    best_validation_step=best_validation_step,
                )

    final_checkpoint = run / "checkpoint_final.pt"
    _save_checkpoint(
        final_checkpoint,
        model=model,
        optimizers=optimizers,
        step=int(training["max_steps"]),
        cfg=cfg,
        best_validation_loss=best_validation_loss,
        best_validation_step=best_validation_step,
    )

    test_probe = fixed_probe(
        arrays["test"],
        training["batch_size"],
        model_cfg.block_size,
        training["eval_batches"],
        seed + 3_001,
    )
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    selected_model = GPT(model_cfg).to(device)
    selected_model.load_state_dict(best_payload["model"])
    selected_test = evaluate_probe(selected_model, test_probe, device)
    selected_metrics = {
        "selected_step": best_validation_step,
        "validation_loss": best_validation_loss,
        "test_loss": selected_test[0],
        "test_perplexity": selected_test[1],
        "test_accuracy": selected_test[2],
    }
    (run / "selected_checkpoint_metrics.json").write_text(
        json.dumps(selected_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    completion = {
        "completed": True,
        "optimizer": training["optimizer"],
        "seed": seed,
        "optimizer_steps": int(training["max_steps"]),
        "tokens_seen": int(training["max_steps"]) * tokens_per_step,
        "best_validation_step": best_validation_step,
        "best_validation_loss": best_validation_loss,
        "selected_checkpoint_test_loss": selected_test[0],
        "wwpgd_call_count": extension.call_count,
        "projected_matrix_count": extension.projected_matrix_count,
        "wwpgd_runtime_seconds": extension.runtime_seconds,
        "wwpgd_interval": int(wwpgd_config["interval"]),
        "elapsed_seconds": time.time() - started_at,
    }
    expected_calls = int(training["max_steps"]) // int(wwpgd_config["interval"])
    if extension.call_count != expected_calls:
        raise RuntimeError(
            f"WWPGD cadence mismatch: expected {expected_calls} calls, "
            f"observed {extension.call_count}"
        )
    (run / "run_complete.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[level0-train] complete run={run} best_step={best_validation_step} "
        f"best_val_loss={best_validation_loss:.4f} "
        f"selected_test_loss={selected_test[0]:.4f} "
        f"wwpgd_calls={extension.call_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
