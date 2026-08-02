from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiment_2" / "src"))

import experiment_2.train as train_module  # noqa: E402
from experiment_2.model import projected_modules  # noqa: E402


def _write_data(root: Path) -> None:
    root.mkdir(parents=True)
    rng = np.random.default_rng(123)
    sizes = {"train": 4_000, "val": 1_000, "test": 1_000}
    for split, size in sizes.items():
        rng.integers(0, 64, size=size, dtype=np.uint16).tofile(root / f"{split}.bin")
    (root / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "tokenizer": "gpt2",
                "vocab_size": 64,
                "dtype": "uint16",
                "splits": sizes,
                "document_disjoint_splits": True,
            }
        )
    )


def _write_config(path: Path) -> None:
    cfg = yaml.safe_load(
        (REPO / "experiment_2" / "configs" / "level0.yaml").read_text()
    )
    cfg["model"].update(
        vocab_size=64,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    cfg["training"].update(
        batch_size=2,
        grad_accum_steps=1,
        max_steps=2,
        eval_interval=1,
        eval_batches=1,
        checkpoint_interval=1,
        warmup_steps=1,
        seed=13,
    )
    cfg["analysis"].update(weightwatcher_interval=1)
    cfg["controller"].update(
        start_step=1,
        control_interval=1,
        projection_interval=1,
        max_active_layers=2,
        probe_batches=1,
        layer_cooldown_steps=1,
        global_pause_steps=1,
    )
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _fake_measure(model, *, step: int, tokens_seen: int):
    rows = []
    for index, (name, matrix_type, block, _) in enumerate(projected_modules(model)):
        rows.append(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "layer_name": name,
                "matrix_type": matrix_type,
                "block": block,
                "alpha": 3.0 - 0.05 * index,
                "D": 0.05,
                "xmin": 1.0,
                "num_evals": 16,
            }
        )
    return rows


class _FakeExtension:
    def __init__(self, model, wwpgd, controller, probe_loss_fn=None):
        self.model = model
        self.wwpgd = wwpgd
        self.controller = controller
        self.probe_loss_fn = probe_loss_fn
        self.call_count = 0
        self.projected_matrix_count = 0

    def manifest_fields(self):
        return {
            "extension": "adaptive_layerwise_wwpgd_fake_test",
            "controller_layer_performance_probe": "fixed_independent_training_probe",
        }

    def measure_all_layers(self, *, step: int, tokens_seen: int):
        return _fake_measure(self.model, step=step, tokens_seen=tokens_seen)

    def after_optimizer_step(
        self,
        *,
        optimizer_step: int,
        total_optimizer_steps: int,
        tokens_seen: int,
        pre_optimizer_weights,
    ):
        del total_optimizer_steps, pre_optimizer_weights
        active = self.controller.active_layers(step=optimizer_step)
        if not active:
            return []
        layer_name = active[0]
        state = self.controller.states[layer_name]
        self.call_count += 1
        self.projected_matrix_count += 1
        self.controller.record_projection(
            layer_name=layer_name,
            step=optimizer_step,
            status="projected",
            alignment_cosine=0.5,
            projection_to_adamw_ratio=0.1,
            alpha_after_candidate=2.05,
        )
        return [
            {
                "optimizer_step": optimizer_step,
                "tokens_seen": tokens_seen,
                "projection_event": self.call_count - 1,
                "projection_status": "projected",
                "consecutive_failures": 0,
                "layer_name": layer_name,
                "matrix_type": state.matrix_type,
                "block": state.block,
                "alpha_before": state.alpha,
                "alpha_after_candidate": 2.05,
                "alpha_error_before": state.alpha - 2.0,
                "alpha_error_after_candidate": 0.05,
                "alpha_improvement": abs(state.alpha - 2.0) - 0.05,
                "probe_loss_before": 1.0,
                "probe_loss_after_candidate": 0.999,
                "probe_loss_delta": -0.001,
                "target_alpha": 2.0,
                "adamw_delta_norm": 1.0,
                "projection_delta_norm_requested": 0.1,
                "projection_to_adamw_ratio_requested": 0.1,
                "alignment_cosine": 0.5,
                "relative_frobenius_change_requested": 1e-4,
                "relative_frobenius_change_applied": 1e-4,
                "trust_region_scale": 1.0,
                "update_ratio_scale": 1.0,
                "changed": True,
                "credit_ema": state.credit_ema,
                "cooldown_until": state.cooldown_until,
                "candidate_analyzed_matrix_count": len(active),
                "projection_runtime_seconds": 0.001,
            }
        ]

    def summary(self):
        return {
            "call_count": self.call_count,
            "successful_call_count": self.call_count,
            "failed_call_count": 0,
            "projected_matrix_count": self.projected_matrix_count,
            "rejected_matrix_count": 0,
            "runtime_seconds": 0.0,
            "layers_reached_target": 1,
            "measured_layer_count": len(self.controller.states),
            "median_final_absolute_alpha_error": 1.0,
            "controller": self.controller.snapshot(),
        }


def _run(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    train_module.main()


def test_bounded_paired_training_writes_controller_artifacts(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    baseline_results = tmp_path / "baseline" / "results"
    adaptive_results = tmp_path / "adaptive" / "results"
    config_path = tmp_path / "config.yaml"
    _write_data(data_root)
    _write_config(config_path)

    monkeypatch.setattr(train_module, "measure_model_layers", _fake_measure)
    monkeypatch.setattr(train_module, "AdaptiveWWPGDExtension", _FakeExtension)

    _run(
        monkeypatch,
        [
            "experiment2-train",
            "--config",
            str(config_path),
            "--arm",
            "adamw",
            "--seed",
            "13",
            "--device",
            "cpu",
            "--data-root",
            str(data_root),
            "--results-root",
            str(baseline_results),
        ],
    )
    baseline_run = baseline_results / "adamw_seed_13"
    assert json.loads((baseline_run / "run_complete.json").read_text())["completed"]

    _run(
        monkeypatch,
        [
            "experiment2-train",
            "--config",
            str(config_path),
            "--arm",
            "adaptive_wwpgd",
            "--seed",
            "13",
            "--device",
            "cpu",
            "--data-root",
            str(data_root),
            "--results-root",
            str(adaptive_results),
            "--baseline-run",
            str(baseline_run),
        ],
    )

    adaptive_run = adaptive_results / "adamw_adaptive_wwpgd_seed_13"
    completion = json.loads((adaptive_run / "run_complete.json").read_text())
    assert completion["completed"] is True
    assert completion["arm"] == "adaptive_wwpgd"
    assert completion["optimizer_steps"] == 2

    for name in (
        "metrics.csv",
        "layer_measurements.csv",
        "controller_windows.csv",
        "projection_events.csv",
        "controller_summary.json",
        "checkpoint_best.pt",
        "checkpoint_final.pt",
        "selected_checkpoint_metrics.json",
        "run_complete.json",
    ):
        assert (adaptive_run / name).is_file(), name

    metrics = pd.read_csv(adaptive_run / "metrics.csv")
    projection = pd.read_csv(adaptive_run / "projection_events.csv")
    assert int(metrics.step.max()) == 2
    assert len(projection) >= 1
    assert set(projection.projection_status) == {"projected"}
