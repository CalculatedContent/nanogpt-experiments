from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

from .checkpoints import (
    load_checkpoint_for_resume,
    prepare_metrics,
    save_checkpoint,
    test_metrics_for_checkpoint,
)
from .config import (
    load_config,
    optimizer_profile,
    protocol_fingerprint,
    roots,
    warmup_steps_for,
)
from .manifest import write_manifest
from .model import GPT, GPTConfig
from .optim import (
    make_optimizer_handles,
    optimizer_step,
    set_learning_rates,
    zero_grad,
)
from .runtime import (
    configure_runtime,
    device_auto,
    evaluate_probe,
    fixed_probe,
    format_eta,
    gradient_norm,
    load_data,
    model_weight_norm,
    mps_memory_megabytes,
    parameter_snapshot,
    random_batch,
    seed_all,
    synchronize_device,
    update_norm_between,
)
from .spectral import run_weightwatcher


METRIC_FIELDS = [
    "step",
    "tokens_seen",
    "epoch",
    "elapsed_sec",
    "tokens_per_sec",
    "primary_lr",
    "auxiliary_lr",
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
    "grad_norm_pre_clip",
    "grad_norm_post_clip",
    "gradient_clipped",
    "weight_norm",
    "update_norm_since_eval",
    "update_to_weight_ratio",
    "mps_current_allocated_mb",
    "mps_driver_allocated_mb",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/level0.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    parser.add_argument(
        "--optimizer", choices=["sgd_momentum", "adamw", "muon"]
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")

    cfg = load_config(args.config)
    cfg = deepcopy(cfg)
    training = cfg["training"]
    optimizer_name = str(args.optimizer or training["optimizer"]).lower()
    seed = int(args.seed if args.seed is not None else training["seed"])
    training["optimizer"] = optimizer_name
    training["seed"] = seed
    profile = optimizer_profile(cfg, optimizer_name)

    resolved = roots()
    data_root = Path(args.data_root or resolved["data"])
    results_root = Path(args.results_root or resolved["results"])
    run_dir = results_root / optimizer_name / f"seed_{seed}"
    completion_path = run_dir / "run_complete.json"
    if completion_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"completed run already exists: {run_dir}; use runner skip logic or --overwrite"
        )
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"incomplete run directory exists: {run_dir}; pass --resume or --overwrite"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    device = device_auto() if args.device == "auto" else torch.device(args.device)
    configure_runtime(device, cfg)
    seed_all(seed)
    train_generator = torch.Generator(device="cpu").manual_seed(seed + 11)

    model_cfg = GPTConfig(**cfg["model"])
    data_metadata, arrays = load_data(data_root, model_cfg)
    train_tokens = int(data_metadata["splits"]["train"])
    val_tokens = int(data_metadata["splits"]["val"])
    test_tokens = int(data_metadata["splits"]["test"])
    model = GPT(model_cfg).to(device)
    handles = make_optimizer_handles(model, profile)
    warmup_steps = warmup_steps_for(profile, int(training["max_steps"]))
    fingerprint = protocol_fingerprint(
        cfg, optimizer=optimizer_name, seed=seed, data_manifest=data_metadata
    )

    forward_model: nn.Module = model
    if bool(training.get("compile", False)):
        forward_model = torch.compile(model)

    batch_size = int(training["batch_size"])
    grad_accum_steps = int(training["grad_accum_steps"])
    max_steps = int(training["max_steps"])
    tokens_per_step = batch_size * model_cfg.block_size * grad_accum_steps
    planned_tokens = tokens_per_step * max_steps
    planned_epochs = planned_tokens / train_tokens
    target_epochs = float(training.get("target_train_epochs", planned_epochs))
    if abs(planned_epochs - target_epochs) > tokens_per_step / train_tokens + 1e-9:
        print(
            "[level0-train] WARNING configured max_steps differs from target epochs: "
            f"planned={planned_epochs:.6f} target={target_epochs:.6f}",
            flush=True,
        )

    train_probe = fixed_probe(
        arrays["train"], batch_size, model_cfg.block_size, training["eval_batches"], seed + 1_001
    )
    val_probe = fixed_probe(
        arrays["val"], batch_size, model_cfg.block_size, training["eval_batches"], seed + 2_001
    )

    start_step = 0
    best_validation_loss = float("inf")
    best_validation_step = 0
    elapsed_offset = 0.0
    latest_checkpoint = run_dir / "checkpoint_latest.pt"
    best_checkpoint = run_dir / "checkpoint_best.pt"
    if args.resume:
        if not latest_checkpoint.exists():
            raise FileNotFoundError(
                f"cannot resume without {latest_checkpoint}"
            )
        (
            start_step,
            best_validation_loss,
            best_validation_step,
            elapsed_offset,
        ) = load_checkpoint_for_resume(
            latest_checkpoint,
            model=model,
            handles=handles,
            expected_fingerprint=fingerprint,
            train_generator=train_generator,
        )
        print(
            f"[level0-train] resuming optimizer={optimizer_name} seed={seed} step={start_step}",
            flush=True,
        )

    write_manifest(
        run_dir,
        cfg=cfg,
        optimizer_name=optimizer_name,
        profile=profile,
        warmup_steps=warmup_steps,
        device=device,
        model=model,
        data_root=data_root,
        data_metadata=data_metadata,
        tokens_per_step=tokens_per_step,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        planned_tokens=planned_tokens,
        planned_epochs=planned_epochs,
        target_epochs=target_epochs,
        fingerprint=fingerprint,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
    )

    metrics_path = run_dir / "metrics.csv"
    prepare_metrics(
        metrics_path,
        fields=METRIC_FIELDS,
        resume_step=start_step if args.resume else None,
    )
    last_grad_pre = float("nan")
    last_grad_post = float("nan")
    last_clipped = False
    last_lrs = {"primary": 0.0, "auxiliary": float("nan")}
    previous_snapshot = parameter_snapshot(model)
    started_at = time.time()

    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        for completed_steps in range(start_step, max_steps + 1):
            evaluation_due = (
                completed_steps % int(training["eval_interval"]) == 0
                or completed_steps == max_steps
            )
            if evaluation_due:
                synchronize_device(device)
                train_metrics = evaluate_probe(forward_model, train_probe, device)
                val_metrics = evaluate_probe(forward_model, val_probe, device)
                elapsed = elapsed_offset + (time.time() - started_at)
                current_snapshot = parameter_snapshot(model)
                update_norm = update_norm_between(previous_snapshot, current_snapshot)
                previous_snapshot = current_snapshot
                weight_norm = model_weight_norm(model)
                update_ratio = update_norm / max(weight_norm, 1e-30)

                if val_metrics[0] < best_validation_loss:
                    best_validation_loss = val_metrics[0]
                    best_validation_step = completed_steps
                    save_checkpoint(
                        best_checkpoint,
                        model=model,
                        handles=handles,
                        step=completed_steps,
                        cfg=cfg,
                        optimizer_name=optimizer_name,
                        profile=profile,
                        best_validation_loss=best_validation_loss,
                        best_validation_step=best_validation_step,
                        fingerprint=fingerprint,
                        train_generator=train_generator,
                        elapsed_seconds=elapsed,
                    )

                test_metrics = (float("nan"), float("nan"), float("nan"))
                if completed_steps == max_steps:
                    test_probe = fixed_probe(
                        arrays["test"],
                        batch_size,
                        model_cfg.block_size,
                        training["eval_batches"],
                        seed + 3_001,
                    )
                    test_metrics = evaluate_probe(forward_model, test_probe, device)

                tokens_seen = completed_steps * tokens_per_step
                epoch = tokens_seen / train_tokens
                current_mps_mb, driver_mps_mb = mps_memory_megabytes(device)
                writer.writerow(
                    {
                        "step": completed_steps,
                        "tokens_seen": tokens_seen,
                        "epoch": epoch,
                        "elapsed_sec": elapsed,
                        "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
                        "primary_lr": last_lrs.get("primary", float("nan")),
                        "auxiliary_lr": last_lrs.get("auxiliary", float("nan")),
                        "train_loss": train_metrics[0],
                        "train_perplexity": train_metrics[1],
                        "train_accuracy": train_metrics[2],
                        "val_loss": val_metrics[0],
                        "val_perplexity": val_metrics[1],
                        "val_accuracy": val_metrics[2],
                        "test_loss": test_metrics[0],
                        "test_perplexity": test_metrics[1],
                        "test_accuracy": test_metrics[2],
                        "val_generalization_gap": val_metrics[0] - train_metrics[0],
                        "test_generalization_gap": test_metrics[0] - train_metrics[0],
                        "grad_norm_pre_clip": last_grad_pre,
                        "grad_norm_post_clip": last_grad_post,
                        "gradient_clipped": int(last_clipped),
                        "weight_norm": weight_norm,
                        "update_norm_since_eval": update_norm,
                        "update_to_weight_ratio": update_ratio,
                        "mps_current_allocated_mb": current_mps_mb,
                        "mps_driver_allocated_mb": driver_mps_mb,
                    }
                )
                handle.flush()

                rate = completed_steps / max(elapsed, 1e-9)
                remaining = max_steps - completed_steps
                eta = remaining / rate if rate > 0 else float("nan")
                print(
                    "[level0-train] "
                    f"optimizer={optimizer_name} seed={seed} "
                    f"step={completed_steps}/{max_steps} epoch={epoch:.3f} "
                    f"lr={last_lrs.get('primary', 0.0):.3e} "
                    f"train_loss={train_metrics[0]:.4f} "
                    f"val_loss={val_metrics[0]:.4f} "
                    f"val_ppl={val_metrics[1]:.2f} "
                    f"val_acc={100 * val_metrics[2]:.2f}% "
                    f"eta={format_eta(eta)}",
                    flush=True,
                )

                analysis_cfg = cfg["analysis"]
                ww_due = (
                    bool(analysis_cfg["weightwatcher"])
                    and (
                        completed_steps % int(analysis_cfg["weightwatcher_interval"])
                        == 0
                        or completed_steps == max_steps
                    )
                )
                raw_ww = (
                    run_dir
                    / "spectral"
                    / "raw"
                    / f"weightwatcher_step_{completed_steps:07d}.csv"
                )
                if ww_due and not raw_ww.exists():
                    summary = run_weightwatcher(
                        model,
                        run_dir,
                        step=completed_steps,
                        tokens_seen=tokens_seen,
                        train_tokens=train_tokens,
                        analysis_cfg=analysis_cfg,
                    )
                    print(
                        "[level0-spectral] "
                        f"step={completed_steps} "
                        f"alpha_median={summary.get('alpha_median', float('nan')):.4f} "
                        f"ERG_gap_median={summary.get('ERG_gap_median', float('nan')):.4f}",
                        flush=True,
                    )
                    if (
                        device.type == "mps"
                        and bool(
                            cfg["runtime"].get(
                                "empty_mps_cache_after_spectral_analysis", True
                            )
                        )
                        and hasattr(torch.mps, "empty_cache")
                    ):
                        torch.mps.empty_cache()

            if completed_steps == max_steps:
                break

            last_lrs = set_learning_rates(
                handles,
                update_index=completed_steps,
                max_steps=max_steps,
                warmup_steps=warmup_steps,
            )
            zero_grad(handles)
            for _ in range(grad_accum_steps):
                x_cpu, y_cpu = random_batch(
                    arrays["train"],
                    batch_size,
                    model_cfg.block_size,
                    train_generator,
                )
                x = x_cpu.to(device)
                y = y_cpu.to(device)
                _, loss = forward_model(x, y)
                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"nonfinite training loss at step {completed_steps + 1}"
                    )
                (loss / grad_accum_steps).backward()

            pre_clip = gradient_norm(model.parameters())
            if not torch.isfinite(pre_clip):
                raise FloatingPointError(
                    f"nonfinite gradient norm at step {completed_steps + 1}"
                )
            grad_clip = float(training["grad_clip"])
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            post_clip = gradient_norm(model.parameters())
            last_grad_pre = float(pre_clip.detach().cpu())
            last_grad_post = float(post_clip.detach().cpu())
            last_clipped = grad_clip > 0 and last_grad_pre > grad_clip
            optimizer_step(handles)

            next_step = completed_steps + 1
            if next_step % int(training["checkpoint_interval"]) == 0:
                elapsed = elapsed_offset + (time.time() - started_at)
                checkpoint_kwargs = dict(
                    model=model,
                    handles=handles,
                    step=next_step,
                    cfg=cfg,
                    optimizer_name=optimizer_name,
                    profile=profile,
                    best_validation_loss=best_validation_loss,
                    best_validation_step=best_validation_step,
                    fingerprint=fingerprint,
                    train_generator=train_generator,
                    elapsed_seconds=elapsed,
                )
                save_checkpoint(latest_checkpoint, **checkpoint_kwargs)
                if bool(training.get("keep_periodic_checkpoints", False)):
                    save_checkpoint(
                        run_dir / f"checkpoint_step_{next_step:07d}.pt",
                        **checkpoint_kwargs,
                    )

    synchronize_device(device)
    total_elapsed = elapsed_offset + (time.time() - started_at)
    final_checkpoint = run_dir / "checkpoint_final.pt"
    final_kwargs = dict(
        model=model,
        handles=handles,
        step=max_steps,
        cfg=cfg,
        optimizer_name=optimizer_name,
        profile=profile,
        best_validation_loss=best_validation_loss,
        best_validation_step=best_validation_step,
        fingerprint=fingerprint,
        train_generator=train_generator,
        elapsed_seconds=total_elapsed,
    )
    save_checkpoint(final_checkpoint, **final_kwargs)
    save_checkpoint(latest_checkpoint, **final_kwargs)

    test_probe = fixed_probe(
        arrays["test"],
        batch_size,
        model_cfg.block_size,
        training["eval_batches"],
        seed + 3_001,
    )
    final_test = test_metrics_for_checkpoint(
        final_checkpoint, model_cfg=model_cfg, test_probe=test_probe, device=device
    )
    selected_test = test_metrics_for_checkpoint(
        best_checkpoint, model_cfg=model_cfg, test_probe=test_probe, device=device
    )
    test_results = {
        "policy": "test evaluated only at final and validation-selected checkpoints",
        "final": {
            "step": final_test[3],
            "loss": final_test[0],
            "perplexity": final_test[1],
            "accuracy": final_test[2],
        },
        "validation_selected": {
            "step": selected_test[3],
            "validation_loss": best_validation_loss,
            "loss": selected_test[0],
            "perplexity": selected_test[1],
            "accuracy": selected_test[2],
        },
    }
    (run_dir / "test_results.json").write_text(
        json.dumps(test_results, indent=2, sort_keys=True), encoding="utf-8"
    )

    completion = {
        "completed": True,
        "optimizer": optimizer_name,
        "seed": seed,
        "optimizer_steps": max_steps,
        "tokens_seen": max_steps * tokens_per_step,
        "train_epochs": planned_epochs,
        "best_validation_step": best_validation_step,
        "best_validation_loss": best_validation_loss,
        "final_test_loss": final_test[0],
        "final_test_perplexity": final_test[1],
        "final_test_accuracy": final_test[2],
        "selected_checkpoint_test_loss": selected_test[0],
        "selected_checkpoint_test_perplexity": selected_test[1],
        "selected_checkpoint_test_accuracy": selected_test[2],
        "elapsed_seconds": total_elapsed,
        "protocol_fingerprint": fingerprint,
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[level0-train] complete run={run_dir} "
        f"best_step={best_validation_step} best_val_loss={best_validation_loss:.4f} "
        f"final_test_loss={final_test[0]:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
