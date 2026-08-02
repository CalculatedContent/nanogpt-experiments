from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .model import GPT, projected_modules


@dataclass
class LayerState:
    layer_name: str
    matrix_type: str
    block: int
    alpha: float = math.nan
    previous_alpha: float = math.nan
    alpha_step: int = -1
    alpha_velocity: float = math.nan
    credit_ema: float = 0.0
    credit_observations: int = 0
    bad_windows: int = 0
    cooldown_until: int = 0
    selected_count: int = 0
    projection_count: int = 0
    rejection_count: int = 0
    last_selected_step: int = -1
    last_projection_step: int = -1
    last_alignment_cosine: float = math.nan
    last_projection_to_adamw_ratio: float = math.nan
    last_projection_status: str = "unmeasured"
    reached_target: bool = False

    def alpha_error(self, target: float) -> float:
        return self.alpha - target if math.isfinite(self.alpha) else math.nan


@dataclass(frozen=True)
class ProjectionDecision:
    status: str
    scale: float
    alignment_cosine: float
    requested_ratio: float
    alpha_improvement: float
    reason: str


def decide_projection(
    *,
    alpha_before: float,
    alpha_after_candidate: float,
    target_alpha: float,
    adamw_delta: torch.Tensor,
    projection_delta: torch.Tensor,
    original_weight: torch.Tensor,
    min_alignment_cosine: float,
    min_alpha_improvement: float,
    max_projection_to_adamw_ratio: float,
    max_relative_frobenius_change: float | None,
) -> ProjectionDecision:
    """Decide whether and how much of one candidate correction to apply."""

    if not (math.isfinite(alpha_before) and math.isfinite(alpha_after_candidate)):
        return ProjectionDecision(
            "rejected_invalid_alpha",
            0.0,
            math.nan,
            math.nan,
            math.nan,
            "invalid alpha",
        )

    before_error = abs(alpha_before - target_alpha)
    after_error = abs(alpha_after_candidate - target_alpha)
    alpha_improvement = before_error - after_error
    if alpha_improvement < min_alpha_improvement:
        return ProjectionDecision(
            "rejected_no_alpha_improvement",
            0.0,
            math.nan,
            math.nan,
            alpha_improvement,
            "candidate did not move alpha sufficiently toward target",
        )

    adamw_flat = adamw_delta.detach().float().reshape(-1)
    projection_flat = projection_delta.detach().float().reshape(-1)
    adamw_norm = float(adamw_flat.norm())
    projection_norm = float(projection_flat.norm())

    if not math.isfinite(projection_norm) or projection_norm <= 0:
        return ProjectionDecision(
            "rejected_empty_or_nonfinite_projection",
            0.0,
            math.nan,
            math.nan,
            alpha_improvement,
            "projection delta is empty or non-finite",
        )

    if adamw_norm > 0:
        alignment = float(
            torch.dot(adamw_flat, projection_flat)
            / (adamw_norm * projection_norm)
        )
        requested_ratio = projection_norm / adamw_norm
    else:
        alignment = 0.0
        requested_ratio = math.inf

    if math.isfinite(alignment) and alignment < min_alignment_cosine:
        return ProjectionDecision(
            "rejected_adamw_opposition",
            0.0,
            alignment,
            requested_ratio,
            alpha_improvement,
            "projection correction opposes the AdamW update",
        )

    scale = 1.0
    if (
        math.isfinite(requested_ratio)
        and requested_ratio > max_projection_to_adamw_ratio
    ):
        scale = min(scale, max_projection_to_adamw_ratio / requested_ratio)

    original_norm = max(float(original_weight.detach().float().norm()), 1e-12)
    relative_change = projection_norm / original_norm
    if (
        max_relative_frobenius_change is not None
        and relative_change > max_relative_frobenius_change
    ):
        scale = min(scale, max_relative_frobenius_change / relative_change)

    if not math.isfinite(scale) or scale <= 0:
        return ProjectionDecision(
            "rejected_invalid_scale",
            0.0,
            alignment,
            requested_ratio,
            alpha_improvement,
            "computed projection scale is invalid",
        )

    return ProjectionDecision(
        "projected",
        scale,
        alignment,
        requested_ratio,
        alpha_improvement,
        "candidate improves alpha and passes optimization guardrails",
    )


class LayerwiseController:
    """Reference-guided controller with layer-wise hysteresis and safety gates."""

    def __init__(self, model: GPT, config: dict[str, Any]):
        self.config = dict(config)
        self.target_alpha = float(config["target_alpha"])
        self.states = {
            name: LayerState(name, matrix_type, block)
            for name, matrix_type, block, _ in projected_modules(model)
        }
        self.active_cohort: list[str] = []
        self.window_layers: set[str] = set()
        self.global_bad_windows = 0
        self.global_pause_until = 0
        self.previous_eval: dict[str, float] | None = None
        self.last_decision_reason = "not_initialized"

    def update_measurements(
        self,
        rows: list[dict[str, Any]],
        *,
        step: int,
    ) -> None:
        for row in rows:
            name = str(row["layer_name"])
            state = self.states[name]
            try:
                alpha = float(row.get("alpha"))
            except (TypeError, ValueError):
                alpha = math.nan
            if math.isfinite(alpha):
                state.previous_alpha = state.alpha
                previous_step = state.alpha_step
                state.alpha = alpha
                state.alpha_step = step
                if (
                    math.isfinite(state.previous_alpha)
                    and previous_step >= 0
                    and step > previous_step
                ):
                    state.alpha_velocity = (
                        alpha - state.previous_alpha
                    ) / (step - previous_step)
                else:
                    state.alpha_velocity = math.nan
                if abs(alpha - self.target_alpha) <= float(
                    self.config["exit_tolerance"]
                ):
                    state.reached_target = True

    def on_evaluation(
        self,
        *,
        step: int,
        train_loss: float,
        val_loss: float,
        baseline_train_loss: float | None,
        baseline_val_loss: float | None,
    ) -> dict[str, Any]:
        previous = self.previous_eval
        adaptive_progress = math.nan
        baseline_progress = math.nan
        advantage = math.nan
        loss_gap = math.nan
        credited_layers = sorted(self.window_layers)

        if baseline_val_loss is not None and math.isfinite(baseline_val_loss):
            loss_gap = val_loss - baseline_val_loss

        if previous is not None:
            adaptive_progress = previous["val_loss"] - val_loss
            if (
                baseline_val_loss is not None
                and previous.get("baseline_val_loss") is not None
                and math.isfinite(float(previous["baseline_val_loss"]))
            ):
                baseline_progress = (
                    float(previous["baseline_val_loss"]) - baseline_val_loss
                )
                advantage = adaptive_progress - baseline_progress

        if math.isfinite(advantage) and credited_layers:
            beta = float(self.config["credit_ema_beta"])
            harmful = advantage < -float(self.config["layer_harm_margin"])
            for name in credited_layers:
                state = self.states[name]
                if state.credit_observations == 0:
                    state.credit_ema = advantage
                else:
                    state.credit_ema = (
                        beta * state.credit_ema + (1.0 - beta) * advantage
                    )
                state.credit_observations += 1
                if harmful:
                    state.bad_windows += 1
                else:
                    state.bad_windows = max(0, state.bad_windows - 1)
                if state.bad_windows >= int(self.config["layer_harm_patience"]):
                    state.cooldown_until = max(
                        state.cooldown_until,
                        step + int(self.config["layer_cooldown_steps"]),
                    )
                    state.bad_windows = 0

        if math.isfinite(loss_gap):
            if loss_gap > float(self.config["global_loss_gap_margin"]):
                self.global_bad_windows += 1
            elif loss_gap <= float(self.config["recovery_gap_margin"]):
                self.global_bad_windows = 0
            if self.global_bad_windows >= int(
                self.config["global_loss_gap_patience"]
            ):
                self.global_pause_until = max(
                    self.global_pause_until,
                    step + int(self.config["global_pause_steps"]),
                )
                self.global_bad_windows = 0
                self.last_decision_reason = "global_loss_guardrail_pause"

        self.previous_eval = {
            "step": float(step),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "baseline_train_loss": baseline_train_loss,
            "baseline_val_loss": baseline_val_loss,
        }
        self.window_layers.clear()

        return {
            "step": step,
            "previous_step": int(previous["step"]) if previous is not None else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "baseline_train_loss": baseline_train_loss,
            "baseline_val_loss": baseline_val_loss,
            "adaptive_progress": adaptive_progress,
            "baseline_progress": baseline_progress,
            "progress_advantage": advantage,
            "loss_gap": loss_gap,
            "active_layers_in_window": ",".join(credited_layers),
            "credited_layers": len(credited_layers),
            "global_bad_windows": self.global_bad_windows,
            "global_pause_until": self.global_pause_until,
            "decision_reason": self.last_decision_reason,
        }

    def choose_cohort(self, *, step: int) -> list[str]:
        if step < int(self.config["start_step"]):
            self.active_cohort = []
            self.last_decision_reason = "before_start_step"
            return []
        if step < self.global_pause_until:
            self.active_cohort = []
            self.last_decision_reason = "global_pause"
            return []

        enter_tolerance = float(self.config["enter_tolerance"])
        eligible: list[LayerState] = []
        for state in self.states.values():
            if not math.isfinite(state.alpha):
                continue
            if step < state.cooldown_until:
                continue
            if abs(state.alpha - self.target_alpha) <= enter_tolerance:
                continue
            eligible.append(state)

        def priority(state: LayerState) -> tuple[float, float, float, int]:
            unseen = 0.0 if state.selected_count == 0 else 1.0
            alpha_priority = -abs(state.alpha - self.target_alpha)
            credit_priority = (
                -state.credit_ema if state.credit_observations else 0.0
            )
            return (
                unseen,
                alpha_priority,
                credit_priority,
                state.last_selected_step,
            )

        eligible.sort(key=priority)
        chosen = eligible[: int(self.config["max_active_layers"])]
        self.active_cohort = [state.layer_name for state in chosen]
        for state in chosen:
            state.selected_count += 1
            state.last_selected_step = step
        self.last_decision_reason = (
            "selected_highest_alpha_error_cohort"
            if chosen
            else "all_layers_in_band_or_cooldown"
        )
        return list(self.active_cohort)

    def active_layers(self, *, step: int) -> list[str]:
        if step < self.global_pause_until:
            return []
        exit_tolerance = float(self.config["exit_tolerance"])
        result = []
        for name in self.active_cohort:
            state = self.states[name]
            if step < state.cooldown_until:
                continue
            if (
                math.isfinite(state.alpha)
                and abs(state.alpha - self.target_alpha) <= exit_tolerance
            ):
                continue
            result.append(name)
        return result

    def record_projection(
        self,
        *,
        layer_name: str,
        step: int,
        status: str,
        alignment_cosine: float,
        projection_to_adamw_ratio: float,
        alpha_after_candidate: float,
    ) -> None:
        state = self.states[layer_name]
        state.last_projection_status = status
        state.last_projection_step = step
        state.last_alignment_cosine = alignment_cosine
        state.last_projection_to_adamw_ratio = projection_to_adamw_ratio
        if status == "projected":
            state.projection_count += 1
            self.window_layers.add(layer_name)
            if math.isfinite(alpha_after_candidate) and abs(
                alpha_after_candidate - self.target_alpha
            ) <= float(self.config["exit_tolerance"]):
                state.reached_target = True
                if layer_name in self.active_cohort:
                    self.active_cohort.remove(layer_name)
        else:
            state.rejection_count += 1
            if status in {
                "rejected_adamw_opposition",
                "rejected_no_alpha_improvement",
                "rejected_probe_loss_regression",
            }:
                state.cooldown_until = max(
                    state.cooldown_until,
                    step + int(self.config["layer_cooldown_steps"]),
                )
                if layer_name in self.active_cohort:
                    self.active_cohort.remove(layer_name)

    def decorate_measurements(
        self,
        rows: list[dict[str, Any]],
        *,
        step: int,
    ) -> list[dict[str, Any]]:
        active = set(self.active_cohort)
        paused = step < self.global_pause_until
        decorated = []
        for row in rows:
            name = str(row["layer_name"])
            state = self.states[name]
            error = state.alpha_error(self.target_alpha)
            decorated.append(
                {
                    **row,
                    "alpha_error": error,
                    "alpha_velocity": state.alpha_velocity,
                    "credit_ema": state.credit_ema,
                    "credit_observations": state.credit_observations,
                    "bad_windows": state.bad_windows,
                    "cooldown_until": state.cooldown_until,
                    "eligible": (
                        math.isfinite(error)
                        and abs(error) > float(self.config["enter_tolerance"])
                        and step >= state.cooldown_until
                    ),
                    "selected": name in active,
                    "global_paused": paused,
                    "last_projection_status": state.last_projection_status,
                }
            )
        return decorated

    def snapshot(self) -> dict[str, Any]:
        return {
            "target_alpha": self.target_alpha,
            "active_cohort": list(self.active_cohort),
            "global_bad_windows": self.global_bad_windows,
            "global_pause_until": self.global_pause_until,
            "last_decision_reason": self.last_decision_reason,
            "states": {name: asdict(state) for name, state in self.states.items()},
        }
