from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config, roots
from .controller import (
    CONTROLLER_WINDOW_FIELDS,
    LAYER_MEASUREMENT_FIELDS,
    PROJECTION_FIELDS,
    AdaptiveWWPGDExtension,
    LayerwiseController,
    append_rows,
    measure_model_layers,
)
from .model import GPT, GPTConfig, projected_modules
from .optim import make_optimizer
from .runtime import (
    device_auto,
    evaluate_probe,
    fixed_probe,
    gradient_norm,
    learning_rate_at,
    load_data,
    model_weight_norm,
    random_batch,
    seed_all,
    snapshot_projected_weights,
)
from .training_support import (
    METRIC_FIELDS,
    BaselineReference,
    save_checkpoint,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=["adamw", "adaptive_wwpgd"], required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    parser.add_argument("--baseline-run")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    training = cfg["training"]
    if args.seed is not None:
        training["seed"] = args.seed
    seed = int(training["seed"])
    scale = str(cfg["experiment"]["scale"])

    resolved = roots()
    data_root = Path(args.data_root or resolved["data"])
    results_root = Path(args.results_root or resolved["results"])
    run_name = (
        f"adamw_seed_{seed}"
        if args.arm == "adamw"
        else f"adamw_adaptive_wwpgd_seed_{seed}"
    )
    run = results_root / run_name
    if run.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"run directory exists: {run}; use a fresh pair root or --overwrite"
            )
        shutil.rmtree(run)
    run.mkdir(parents=True, exist_ok=False)

    device = device_auto() if args.device == "auto" else torch.device(args.device)
    seed_all(seed)
    train_generator = torch.Generator(device="cpu").manual_seed(seed + 11)

    model_cfg = GPTConfig(**cfg["model"])
    data_metadata, arrays = load_data(data_root, model_cfg)
    model = GPT(model_cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    baseline_reference = None
    controller = None
    extension = None
    if args.arm == "adaptive_wwpgd":
        if not args.baseline_run and bool(
            cfg["controller"].get("require_baseline_reference", True)
        ):
            raise ValueError("adaptive_wwpgd requires --baseline-run")
        if args.baseline_run:
            baseline_reference = BaselineReference(
                Path(args.baseline_run), cfg, seed
            )
        controller = LayerwiseController(model, cfg["controller"])

    train_probe = fixed_probe(
        arrays["train"],
        int(training["batch_size"]),
        model_cfg.block_size,
        int(training["eval_batches"]),
        seed + 1_001,
    )
    val_probe = fixed_probe(
        arrays["val"],
        int(training["batch_size"]),
        model_cfg.block_size,
        int(training["eval_batches"]),
        seed + 2_001,
    )

    if controller is not None:
        controller_probe = fixed_probe(
            arrays["train"],
            int(training["batch_size"]),
            model_cfg.block_size,
            int(cfg["controller"]["probe_batches"]),
            seed + 4_001,
        )

        def controller_probe_loss() -> float:
            return float(evaluate_probe(model, controller_probe, device)[0])

        extension = AdaptiveWWPGDExtension(
            model,
            cfg["wwpgd"],
            controller,
            probe_loss_fn=controller_probe_loss,
        )

    tokens_per_step = (
        int(training["batch_size"])
        * model_cfg.block_size
        * int(training["grad_accum_steps"])
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    matrix_parameters = sum(
        module.weight.numel() for _, _, _, module in projected_modules(model)
    )

    manifest = {
        "schema_version": 1,
        "experiment": "experiment_2",
        "scale": scale,
        "arm": args.arm,
        "seed": seed,
        "config": cfg,
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "parameter_count": total_parameters,
        "transformer_matrix_parameter_count": matrix_parameters,
        "data_root": str(data_root.resolve()),
        "data_manifest": data_metadata,
        "tokens_per_optimizer_step": tokens_per_step,
        "planned_training_tokens": tokens_per_step * int(training["max_steps"]),
        "evaluation_sampling": "fixed_probes_with_independent_rng_streams",
        "test_evaluation_policy": "final_and_validation_selected_checkpoint_only",
        "baseline_reference_run": (
            str(baseline_reference.run.resolve()) if baseline_reference else None
        ),
        "baseline_reference_metrics_sha256": (
            sha256(baseline_reference.metrics_path) if baseline_reference else None
        ),
        "controller_semantics": (
            "reference_guided_layer_cohort_credit_is_approximate_not_causal"
            if controller is not None
            else None
        ),
        **(extension.manifest_fields() if extension is not None else {}),
    }
    (run / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    metrics_path = run / "metrics.csv"
    measurements_path = run / "layer_measurements.csv"
    windows_path = run / "controller_windows.csv"
    projection_path = run / "projection_events.csv"
    best_checkpoint = run / "checkpoint_best.pt"
    best_validation_loss = float("inf")
    best_validation_step = 0
    last_grad_norm = float("nan")
    last_lr = 0.0
    started_at = time.time()

    with metrics_path.open("w", newline="", encoding="utf-8") as metric_handle:
        metric_writer = csv.DictWriter(metric_handle, fieldnames=METRIC_FIELDS)
        metric_writer.writeheader()

        for completed_steps in range(int(training["max_steps"]) + 1):
            evaluation_due = (
                completed_steps % int(training["eval_interval"]) == 0
                or completed_steps == int(training["max_steps"])
            )
            baseline_row = None
            if evaluation_due:
                train_metrics = evaluate_probe(model, train_probe, device)
                val_metrics = evaluate_probe(model, val_probe, device)
                if baseline_reference is not None:
                    baseline_row = baseline_reference.at(completed_steps)

                if val_metrics[0] < best_validation_loss:
                    best_validation_loss = val_metrics[0]
                    best_validation_step = completed_steps
                    save_checkpoint(
                        best_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        step=completed_steps,
                        cfg=cfg,
                        best_validation_loss=best_validation_loss,
                        best_validation_step=best_validation_step,
                        controller=controller,
                    )

                test_metrics = (float("nan"), float("nan"), float("nan"))
                if completed_steps == int(training["max_steps"]):
                    test_probe = fixed_probe(
                        arrays["test"],
                        int(training["batch_size"]),
                        model_cfg.block_size,
                        int(training["eval_batches"]),
                        seed + 3_001,
                    )
                    test_metrics = evaluate_probe(model, test_probe, device)

                baseline_train_loss = (
                    float(baseline_row.train_loss)
                    if baseline_row is not None
                    else math.nan
                )
                baseline_val_loss = (
                    float(baseline_row.val_loss)
                    if baseline_row is not None
                    else math.nan
                )
                elapsed = time.time() - started_at
                metric_writer.writerow(
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
                        "val_generalization_gap": val_metrics[0] - train_metrics[0],
                        "test_generalization_gap": test_metrics[0] - train_metrics[0],
                        "baseline_train_loss": baseline_train_loss,
                        "baseline_val_loss": baseline_val_loss,
                        "loss_gap_vs_baseline": (
                            val_metrics[0] - baseline_val_loss
                            if math.isfinite(baseline_val_loss)
                            else math.nan
                        ),
                        "grad_norm": last_grad_norm,
                        "weight_norm": model_weight_norm(model),
                        "active_layer_count": (
                            len(controller.active_cohort) if controller else 0
                        ),
                        "global_pause_until": (
                            controller.global_pause_until if controller else 0
                        ),
                    }
                )
                metric_handle.flush()

                label = f"experiment2-{scale}-{args.arm}"
                rate = completed_steps / max(elapsed, 1e-9)
                remaining = int(training["max_steps"]) - completed_steps
                eta = remaining / rate / 60 if rate > 0 else math.nan
                print(
                    f"[{label}] step={completed_steps}/{training['max_steps']} "
                    f"tokens={completed_steps * tokens_per_step:,} lr={last_lr:.3e} "
                    f"train_loss={train_metrics[0]:.4f} val_loss={val_metrics[0]:.4f} "
                    f"val_acc={100 * val_metrics[2]:.2f}% eta={eta:.1f}m",
                    flush=True,
                )

                if controller is not None:
                    window = controller.on_evaluation(
                        step=completed_steps,
                        train_loss=train_metrics[0],
                        val_loss=val_metrics[0],
                        baseline_train_loss=(
                            baseline_train_loss
                            if math.isfinite(baseline_train_loss)
                            else None
                        ),
                        baseline_val_loss=(
                            baseline_val_loss
                            if math.isfinite(baseline_val_loss)
                            else None
                        ),
                    )
                    append_rows(windows_path, CONTROLLER_WINDOW_FIELDS, [window])

            measurement_due = (
                completed_steps % int(cfg["analysis"]["weightwatcher_interval"]) == 0
                or completed_steps == int(training["max_steps"])
            )
            if measurement_due:
                if extension is not None:
                    measurement_rows = extension.measure_all_layers(
                        step=completed_steps,
                        tokens_seen=completed_steps * tokens_per_step,
                    )
                    controller.update_measurements(
                        measurement_rows,
                        step=completed_steps,
                    )
                    controller.choose_cohort(step=completed_steps)
                    decorated = controller.decorate_measurements(
                        measurement_rows,
                        step=completed_steps,
                    )
                else:
                    measurement_rows = measure_model_layers(
                        model,
                        step=completed_steps,
                        tokens_seen=completed_steps * tokens_per_step,
                    )
                    decorated = [
                        {
                            **row,
                            "alpha_error": (
                                float(row["alpha"])
                                - float(cfg["controller"]["target_alpha"])
                                if row.get("alpha") is not None
                                else math.nan
                            ),
                            "alpha_velocity": math.nan,
                            "credit_ema": 0.0,
                            "credit_observations": 0,
                            "bad_windows": 0,
                            "cooldown_until": 0,
                            "eligible": False,
                            "selected": False,
                            "global_paused": False,
                            "last_projection_status": "baseline_no_projection",
                        }
                        for row in measurement_rows
                    ]
                append_rows(
                    measurements_path,
                    LAYER_MEASUREMENT_FIELDS,
                    decorated,
                )
                alpha = pd.to_numeric(
                    pd.Series([row.get("alpha") for row in measurement_rows]),
                    errors="coerce",
                )
                finite = alpha[np.isfinite(alpha)]
                if len(finite):
                    print(
                        f"[experiment2-{scale}-{args.arm}] "
                        f"spectral step={completed_steps} matrices={len(finite)} "
                        f"median_alpha={finite.median():.3f}",
                        flush=True,
                    )

            if completed_steps == int(training["max_steps"]):
                break

            update_index = completed_steps
            current_lr = learning_rate_at(update_index, training)
            scale_factor = current_lr / float(training["learning_rate"])
            for group, base_lr in zip(
                optimizer.param_groups,
                base_lrs,
                strict=True,
            ):
                group["lr"] = base_lr * scale_factor
            last_lr = current_lr

            pre_optimizer_weights = (
                snapshot_projected_weights(model) if extension is not None else {}
            )
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
            optimizer.step()

            next_step = completed_steps + 1
            if extension is not None:
                projection_rows = extension.after_optimizer_step(
                    optimizer_step=next_step,
                    total_optimizer_steps=int(training["max_steps"]),
                    tokens_seen=next_step * tokens_per_step,
                    pre_optimizer_weights=pre_optimizer_weights,
                )
                append_rows(projection_path, PROJECTION_FIELDS, projection_rows)

            if next_step % int(training["checkpoint_interval"]) == 0:
                save_checkpoint(
                    run / f"checkpoint_{next_step:07d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=next_step,
                    cfg=cfg,
                    best_validation_loss=best_validation_loss,
                    best_validation_step=best_validation_step,
                    controller=controller,
                )

    final_checkpoint = run / "checkpoint_final.pt"
    save_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        step=int(training["max_steps"]),
        cfg=cfg,
        best_validation_loss=best_validation_loss,
        best_validation_step=best_validation_step,
        controller=controller,
    )

    test_probe = fixed_probe(
        arrays["test"],
        int(training["batch_size"]),
        model_cfg.block_size,
        int(training["eval_batches"]),
        seed + 3_001,
    )
    best_payload = torch.load(
        best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
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

    controller_summary = extension.summary() if extension is not None else None
    if controller_summary is not None:
        (run / "controller_summary.json").write_text(
            json.dumps(controller_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    completion = {
        "completed": True,
        "experiment": "experiment_2",
        "scale": scale,
        "arm": args.arm,
        "seed": seed,
        "optimizer_steps": int(training["max_steps"]),
        "tokens_seen": int(training["max_steps"]) * tokens_per_step,
        "best_validation_step": best_validation_step,
        "best_validation_loss": best_validation_loss,
        "selected_checkpoint_test_loss": selected_test[0],
        "elapsed_seconds": time.time() - started_at,
        "controller_summary": controller_summary,
    }
    (run / "run_complete.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[experiment2-{scale}-{args.arm}] complete run={run} "
        f"best_step={best_validation_step} "
        f"best_val_loss={best_validation_loss:.4f} "
        f"selected_test_loss={selected_test[0]:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
