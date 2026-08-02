from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .model import GPT, projected_modules
from .policy import LayerwiseController, decide_projection
from .spectral import (
    _ProjectedMatrixHolder,
    _analyze_holder,
    _is_retryable_projection_error,
    _match_weightwatcher_row,
    _module_by_name,
    measure_model_layers,
    preserve_global_rng,
)


class AdaptiveWWPGDExtension:
    """Apply stock WWPGD only to the controller-selected matrix cohort."""

    def __init__(
        self,
        model: GPT,
        wwpgd: dict[str, Any],
        controller: LayerwiseController,
        probe_loss_fn: Callable[[], float] | None = None,
    ):
        self.model = model
        self.wwpgd = dict(wwpgd)
        self.controller = controller
        self.probe_loss_fn = probe_loss_fn
        self.call_count = 0
        self.successful_call_count = 0
        self.failed_call_count = 0
        self.consecutive_failure_count = 0
        self.max_observed_consecutive_failures = 0
        self.projected_matrix_count = 0
        self.rejected_matrix_count = 0
        self.runtime_seconds = 0.0
        self._resolved_config = None

    def _adapter(self):
        try:
            from wwgpt.ww import (
                _external_config_object,
                _external_wwpgd_module,
                external_wwpgd_config_from_experiment,
                external_wwpgd_manifest_fields,
                run_pip_wwpgd_candidate,
            )
        except Exception as exc:
            raise RuntimeError(
                "Install the repository root before running Experiment 2"
            ) from exc
        return (
            _external_config_object,
            _external_wwpgd_module,
            external_wwpgd_config_from_experiment,
            external_wwpgd_manifest_fields,
            run_pip_wwpgd_candidate,
        )

    def resolved_config(self):
        if self._resolved_config is None:
            _, _, resolver, _, _ = self._adapter()
            requested = SimpleNamespace(
                enabled=True,
                extension="wwpgd",
                target_alpha=float(self.wwpgd["target_alpha"]),
                blend_eta=float(self.wwpgd["blend_eta"]),
                cayley_eta=float(self.wwpgd["cayley_eta"]),
                min_tail=int(self.wwpgd["min_tail"]),
                use_detx=bool(self.wwpgd["use_detx"]),
                verbose=False,
                candidate_device="cpu",
                max_relative_frobenius_change=self.wwpgd.get(
                    "max_relative_frobenius_change"
                ),
            )
            self._resolved_config = resolver(requested)
        return self._resolved_config

    def manifest_fields(self) -> dict[str, Any]:
        _, _, _, manifest_builder, _ = self._adapter()
        requested = SimpleNamespace(
            enabled=True,
            extension="wwpgd",
            target_alpha=float(self.wwpgd["target_alpha"]),
            blend_eta=float(self.wwpgd["blend_eta"]),
            cayley_eta=float(self.wwpgd["cayley_eta"]),
            min_tail=int(self.wwpgd["min_tail"]),
            use_detx=bool(self.wwpgd["use_detx"]),
            verbose=False,
            candidate_device="cpu",
            max_relative_frobenius_change=self.wwpgd.get(
                "max_relative_frobenius_change"
            ),
        )
        fields = manifest_builder(True, requested)
        fields.update(
            {
                "extension": "adaptive_layerwise_wwpgd",
                "wwpgd_apply_mode": "event_projection",
                "wwpgd_scope": "controller_selected_transformer_matrices",
                "wwpgd_candidate_device": "cpu",
                "wwpgd_retry_policy": "skip_retryable_event_and_retry_next_step",
                "wwpgd_max_consecutive_failures": int(
                    self.wwpgd["max_consecutive_failures"]
                ),
                "controller_credit_assignment": "shared_across_active_window_cohort",
                "controller_layer_performance_probe": (
                    "fixed_independent_training_probe"
                    if self.probe_loss_fn is not None
                    else "disabled"
                ),
            }
        )
        return fields

    def measure_all_layers(
        self,
        *,
        step: int,
        tokens_seen: int,
    ) -> list[dict[str, Any]]:
        return measure_model_layers(self.model, step=step, tokens_seen=tokens_seen)

    def after_optimizer_step(
        self,
        *,
        optimizer_step: int,
        total_optimizer_steps: int,
        tokens_seen: int,
        pre_optimizer_weights: dict[str, torch.Tensor],
    ) -> list[dict[str, Any]]:
        if optimizer_step % int(self.controller.config["projection_interval"]) != 0:
            return []

        selected = set(self.controller.active_layers(step=optimizer_step))
        if not selected:
            return []

        candidate_holder = _ProjectedMatrixHolder(self.model, selected)
        selected_safe_names = set(candidate_holder.safe_to_live)
        if not selected_safe_names:
            return []

        (
            external_config_object,
            external_module,
            _,
            _,
            run_candidate,
        ) = self._adapter()
        resolved = self.resolved_config()

        def selector(mm: nn.Module, layer_name: str, row: object | None = None):
            del row
            match = next(
                (
                    name
                    for name in selected_safe_names
                    if layer_name == name
                    or str(layer_name).endswith(name)
                    or name in str(layer_name)
                ),
                None,
            )
            return _module_by_name(mm, match) if match is not None else None

        event_index = self.call_count
        self.call_count += 1
        started = time.perf_counter()
        try:
            with preserve_global_rng(), torch.no_grad():
                result = run_candidate(
                    candidate_holder,
                    external_config_object(external_module(), resolved),
                    epoch=max(0, optimizer_step - 1),
                    num_epochs=max(total_optimizer_steps, 1),
                    global_step=optimizer_step,
                    layer_selector=selector,
                )
                after_details = _analyze_holder(candidate_holder)
        except Exception as exc:
            runtime = time.perf_counter() - started
            self.runtime_seconds += runtime
            if not _is_retryable_projection_error(exc):
                raise
            self.failed_call_count += 1
            self.consecutive_failure_count += 1
            self.max_observed_consecutive_failures = max(
                self.max_observed_consecutive_failures,
                self.consecutive_failure_count,
            )
            print(
                "[experiment2-wwpgd] "
                f"step={optimizer_step}/{total_optimizer_steps} "
                f"projection_skipped=true error_type={type(exc).__name__} "
                f"consecutive_failures={self.consecutive_failure_count}/"
                f"{self.wwpgd['max_consecutive_failures']}",
                flush=True,
            )
            if self.consecutive_failure_count >= int(
                self.wwpgd["max_consecutive_failures"]
            ):
                raise RuntimeError(
                    f"Experiment 2 stopped after {self.consecutive_failure_count} "
                    "consecutive numerical WWPGD failures"
                ) from exc
            return [
                {
                    "optimizer_step": optimizer_step,
                    "tokens_seen": tokens_seen,
                    "projection_event": event_index,
                    "projection_status": "skipped_retryable_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc).replace("\n", " "),
                    "consecutive_failures": self.consecutive_failure_count,
                    "changed": False,
                    "candidate_analyzed_matrix_count": len(selected),
                    "projection_runtime_seconds": runtime,
                }
            ]

        candidate_runtime = time.perf_counter() - started
        self.successful_call_count += 1
        self.consecutive_failure_count = 0
        result = result if isinstance(result, dict) else {}
        usable_logs = [
            item
            for item in result.get("ww_logs", [])
            if isinstance(item, pd.DataFrame) and not item.empty
        ]
        if len(usable_logs) != 1:
            raise RuntimeError(
                "stock WWPGD expected exactly one nonempty ww_logs frame; "
                f"found {len(usable_logs)}"
            )
        before_details = usable_logs[0]

        live = {
            name: module
            for name, _, _, module in projected_modules(self.model)
        }
        rows: list[dict[str, Any]] = []
        target = float(self.controller.target_alpha)
        original_weights = {
            layer_name: live[layer_name].weight.detach().clone()
            for layer_name in candidate_holder.safe_to_live.values()
        }
        base_probe_loss = (
            float(self.probe_loss_fn())
            if self.probe_loss_fn is not None
            else math.nan
        )
        pending_weights: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for safe_name, layer_name in candidate_holder.safe_to_live.items():
                matrix_type, block = candidate_holder.metadata[safe_name]
                live_module = live[layer_name]
                original = original_weights[layer_name]
                pre_optimizer = pre_optimizer_weights[layer_name].to(
                    original.device, dtype=original.dtype
                )
                adamw_delta = original - pre_optimizer
                candidate = getattr(candidate_holder, safe_name).weight.detach().to(
                    original.device, dtype=original.dtype
                )
                projection_delta = candidate - original

                before = _match_weightwatcher_row(before_details, safe_name)
                after = _match_weightwatcher_row(after_details, safe_name)
                try:
                    alpha_before = float(before.get("alpha"))
                except (TypeError, ValueError):
                    alpha_before = math.nan
                try:
                    alpha_after = float(after.get("alpha"))
                except (TypeError, ValueError):
                    alpha_after = math.nan

                decision = decide_projection(
                    alpha_before=alpha_before,
                    alpha_after_candidate=alpha_after,
                    target_alpha=target,
                    adamw_delta=adamw_delta,
                    projection_delta=projection_delta,
                    original_weight=original,
                    min_alignment_cosine=float(
                        self.controller.config["min_alignment_cosine"]
                    ),
                    min_alpha_improvement=float(
                        self.controller.config["min_candidate_alpha_improvement"]
                    ),
                    max_projection_to_adamw_ratio=float(
                        self.controller.config["max_projection_to_adamw_ratio"]
                    ),
                    max_relative_frobenius_change=self.wwpgd.get(
                        "max_relative_frobenius_change"
                    ),
                )

                requested_norm = float(projection_delta.float().norm())
                adamw_norm = float(adamw_delta.float().norm())
                original_norm = max(float(original.float().norm()), 1e-12)
                requested_relative = requested_norm / original_norm
                changed = False
                applied_relative = 0.0
                probe_loss_after = math.nan
                probe_loss_delta = math.nan
                status = decision.status
                status_reason = decision.reason

                if decision.status == "projected":
                    proposed = original + decision.scale * projection_delta
                    if not torch.isfinite(proposed).all():
                        status = "rejected_nonfinite_applied_weight"
                        status_reason = "scaled candidate produced non-finite weights"
                    else:
                        if self.probe_loss_fn is not None:
                            live_module.weight.copy_(proposed)
                            try:
                                probe_loss_after = float(self.probe_loss_fn())
                            finally:
                                live_module.weight.copy_(original)
                            probe_loss_delta = probe_loss_after - base_probe_loss
                            if not math.isfinite(probe_loss_after):
                                status = "rejected_nonfinite_probe_loss"
                                status_reason = (
                                    "candidate produced non-finite control-probe loss"
                                )
                            elif probe_loss_delta > float(
                                self.controller.config["max_probe_loss_increase"]
                            ):
                                status = "rejected_probe_loss_regression"
                                status_reason = (
                                    "candidate worsened the fixed independent training probe "
                                    f"by {probe_loss_delta:.6g}"
                                )
                            else:
                                changed = True
                        else:
                            changed = True

                        if changed:
                            pending_weights[layer_name] = proposed.detach().clone()
                            applied_relative = float(
                                (proposed - original).float().norm() / original_norm
                            )

                self.controller.record_projection(
                    layer_name=layer_name,
                    step=optimizer_step,
                    status=status,
                    alignment_cosine=decision.alignment_cosine,
                    projection_to_adamw_ratio=decision.requested_ratio,
                    alpha_after_candidate=alpha_after,
                )
                state = self.controller.states[layer_name]
                if changed:
                    self.projected_matrix_count += 1
                else:
                    self.rejected_matrix_count += 1

                rows.append(
                    {
                        "optimizer_step": optimizer_step,
                        "tokens_seen": tokens_seen,
                        "projection_event": event_index,
                        "projection_status": status,
                        "error_type": "",
                        "error_message": status_reason,
                        "consecutive_failures": 0,
                        "layer_name": layer_name,
                        "matrix_type": matrix_type,
                        "block": block,
                        "alpha_before": alpha_before,
                        "alpha_after_candidate": alpha_after,
                        "alpha_error_before": alpha_before - target,
                        "alpha_error_after_candidate": alpha_after - target,
                        "alpha_improvement": decision.alpha_improvement,
                        "probe_loss_before": base_probe_loss,
                        "probe_loss_after_candidate": probe_loss_after,
                        "probe_loss_delta": probe_loss_delta,
                        "target_alpha": target,
                        "adamw_delta_norm": adamw_norm,
                        "projection_delta_norm_requested": requested_norm,
                        "projection_to_adamw_ratio_requested": decision.requested_ratio,
                        "alignment_cosine": decision.alignment_cosine,
                        "relative_frobenius_change_requested": requested_relative,
                        "relative_frobenius_change_applied": applied_relative,
                        "trust_region_scale": decision.scale,
                        "update_ratio_scale": decision.scale,
                        "changed": changed,
                        "credit_ema": state.credit_ema,
                        "cooldown_until": state.cooldown_until,
                        "candidate_analyzed_matrix_count": len(selected),
                        "projection_runtime_seconds": math.nan,
                    }
                )

            for layer_name, proposed in pending_weights.items():
                live[layer_name].weight.copy_(proposed)

        runtime = time.perf_counter() - started
        self.runtime_seconds += runtime
        for row in rows:
            row["projection_runtime_seconds"] = runtime / max(len(selected), 1)

        if event_index == 0 or optimizer_step % int(
            self.controller.config["control_interval"]
        ) == 0:
            projected = sum(bool(row["changed"]) for row in rows)
            print(
                "[experiment2-wwpgd] "
                f"step={optimizer_step}/{total_optimizer_steps} "
                f"selected={len(selected)} projected={projected} "
                f"rejected={len(rows) - projected} "
                f"candidate_runtime_s={candidate_runtime:.2f} "
                f"total_runtime_s={runtime:.2f}",
                flush=True,
            )
        return rows

    def summary(self) -> dict[str, Any]:
        states = self.controller.states.values()
        final_errors = [
            abs(state.alpha - self.controller.target_alpha)
            for state in states
            if math.isfinite(state.alpha)
        ]
        return {
            "call_count": self.call_count,
            "successful_call_count": self.successful_call_count,
            "failed_call_count": self.failed_call_count,
            "max_observed_consecutive_failures": (
                self.max_observed_consecutive_failures
            ),
            "projected_matrix_count": self.projected_matrix_count,
            "rejected_matrix_count": self.rejected_matrix_count,
            "runtime_seconds": self.runtime_seconds,
            "layers_reached_target": sum(state.reached_target for state in states),
            "measured_layer_count": len(final_errors),
            "median_final_absolute_alpha_error": (
                float(np.median(final_errors)) if final_errors else math.nan
            ),
            "controller": self.controller.snapshot(),
        }
