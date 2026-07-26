from __future__ import annotations

import fnmatch
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

CONTROLLER_VERSION = "adaptive_wwpgd_two_timescale_v1"
MATRIX_TYPES = {"W_K", "W_Q", "W_V", "W_O", "W_MLP_IN", "W_MLP_OUT"}
DIRECTIONS = {"above_target", "below_target", "both"}
RESPONSES = {"linear", "smoothstep"}
DOSE_SCHEDULES = {"bounded_refresh_fraction", "fixed_per_step_gain"}
PRECEDENCE = [
    "global adaptive configuration",
    "matrix-type override",
    "matching layer-glob overrides",
    "exact layer-name override",
]
OVERRIDE_FIELDS = {
    "enabled",
    "direction",
    "above_target",
    "below_target",
    "max_D",
    "max_relative_frobenius_change",
    "cooldown_events",
    "min_observations",
    "alpha_ema_beta",
    # Legacy controller fields remain accepted for explicit ablations.
    "deadband_above_target",
    "full_strength_alpha",
    "max_hardness",
    "response_curve",
    "piecewise_points",
}


@dataclass(frozen=True)
class AdaptiveAlphaSideConfig:
    enabled: bool = True
    deadband: float = 0.0
    full_strength_distance: float = 1.0
    max_hardness: float = 1.0
    response_curve: str = "linear"
    piecewise_points: list[list[float]] = field(default_factory=list)


@dataclass(frozen=True)
class AdaptiveWWPGDConfig:
    apply_mode: str = "event_projection"
    mode: str = "uniform"
    direction: str = "above_target"
    response_curve: str = "linear"
    start_step: int = 0
    min_observations: int = 1
    alpha_ema_beta: float = 0.0
    enabled: bool = True
    deadband_above_target: float = 0.0
    full_strength_alpha: float = 4.0
    # In uniform mode this is the fixed hardness applied to every eligible layer.
    max_hardness: float = 1.0
    max_D: float | None = None
    max_relative_frobenius_change: float | None = None
    cooldown_events: int = 0
    piecewise_points: list[list[float]] = field(
        default_factory=lambda: [[2.0, 0.0], [4.0, 1.0]]
    )
    above_target: AdaptiveAlphaSideConfig = field(
        default_factory=lambda: AdaptiveAlphaSideConfig(
            deadband=0.4,
            full_strength_distance=2.0,
            response_curve="smoothstep",
        )
    )
    below_target: AdaptiveAlphaSideConfig = field(
        default_factory=lambda: AdaptiveAlphaSideConfig(
            deadband=0.2,
            full_strength_distance=1.0,
            response_curve="smoothstep",
        )
    )
    matrix_type_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    layer_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    measurement_interval: int | None = None
    measurement_source: str = "evaluation_interval"
    apply_interval: int = 1
    max_per_step_gain: float = 0.02
    max_endpoint_fraction_per_refresh: float = 0.40
    max_cumulative_relative_frobenius_change_per_refresh: float = 0.025
    dose_schedule: str = "bounded_refresh_fraction"
    max_relative_frobenius_change_per_step: float | None = None
    endpoint_stop_relative_distance: float = 0.0001
    max_endpoint_age_steps: int = 50
    stale_distance_multiplier: float = 2.0
    skip_fast_apply_on_measurement_step: bool = True
    cache_endpoint_on_cpu: bool = False
    refresh_at_final_step: bool = True
    log_every_fast_step: bool = True


@dataclass
class CachedLayerEndpoint:
    layer_name: str
    endpoint_tensor: Any
    measurement_weight_tensor: Any
    measurement_step: int
    measurement_index: int
    raw_alpha: float
    smoothed_alpha: float
    target_alpha: float
    signed_alpha_error: float
    alpha_distance: float
    alpha_side: str
    alpha_hardness: float
    global_event_hardness: float
    combined_horizon_hardness: float
    initial_endpoint_relative_distance: float
    latest_endpoint_relative_distance: float
    stock_candidate_relative_change: float
    D: float
    xmin: float
    detX_num: float
    num_evals: float
    active: bool = True
    invalidation_reason: str = ""
    last_applied_step: int | None = None
    cumulative_applied_relative_change: float = 0.0


def resolve_endpoint_measurement_interval(
    adaptive_cfg: AdaptiveWWPGDConfig,
    runtime_eval_interval: int,
) -> int:
    """Resolve the cached endpoint sampling cadence in exactly one place."""
    value = (
        runtime_eval_interval
        if adaptive_cfg.measurement_source == "evaluation_interval"
        else adaptive_cfg.measurement_interval
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("endpoint measurement interval must be a positive integer")
    return value


def eligible_fast_steps(
    measurement_interval: int,
    apply_interval: int,
    skip_measurement_step: bool = True,
) -> int:
    """Return the exact eligible fast applications in a full refresh window."""
    if measurement_interval <= 0 or apply_interval <= 0:
        return 0
    return sum(
        1
        for offset in range(1, measurement_interval + 1)
        if offset % apply_interval == 0
        and not (skip_measurement_step and offset == measurement_interval)
    )


def derived_interval_gain(cfg: AdaptiveWWPGDConfig, measurement_interval: int) -> float:
    """Derive a cadence-normalized gain for one endpoint refresh window."""
    n_fast = eligible_fast_steps(
        measurement_interval,
        cfg.apply_interval,
        cfg.skip_fast_apply_on_measurement_step,
    )
    if n_fast == 0:
        return 0.0
    if cfg.dose_schedule == "fixed_per_step_gain":
        return cfg.max_per_step_gain
    return 1.0 - (1.0 - cfg.max_endpoint_fraction_per_refresh) ** (1.0 / n_fast)


def effective_base_gain(cfg: AdaptiveWWPGDConfig, measurement_interval: int) -> float:
    """Apply the configured per-step gain as a hard upper bound."""
    return min(cfg.max_per_step_gain, derived_interval_gain(cfg, measurement_interval))


def _measurement_steps(
    total_optimizer_steps: int,
    measurement_interval: int,
    refresh_at_final_step: bool,
) -> list[int]:
    steps = list(range(measurement_interval, total_optimizer_steps + 1, measurement_interval))
    if refresh_at_final_step and total_optimizer_steps > 0 and total_optimizer_steps not in steps:
        steps.append(total_optimizer_steps)
    return sorted(steps)


def validate_adaptive_level_schedule(
    cfg: AdaptiveWWPGDConfig,
    total_optimizer_steps: int,
    measurement_interval: int | None = None,
) -> dict[str, Any]:
    """Validate and describe cached-endpoint cadence independently of model level.

    Measurements before ``start_step`` still count toward ``min_observations``;
    only intervention is gated by ``start_step``.  This mirrors the live
    controller rather than pessimistically resetting the observation count at
    the start step.
    """
    interval = measurement_interval or cfg.measurement_interval
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("measurement_interval is required for schedule validation")
    if total_optimizer_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")

    measurements = _measurement_steps(
        total_optimizer_steps,
        interval,
        cfg.refresh_at_final_step,
    )
    if not measurements:
        raise ValueError("no endpoint measurement occurs during the run")

    eligible_measurements = [
        step
        for observation_index, step in enumerate(measurements, start=1)
        if step >= cfg.start_step and observation_index >= cfg.min_observations
    ]
    if not eligible_measurements:
        raise ValueError(
            "no endpoint measurement can satisfy both start_step and min_observations"
        )

    n_fast = eligible_fast_steps(
        interval,
        cfg.apply_interval,
        cfg.skip_fast_apply_on_measurement_step,
    )
    if cfg.apply_mode == "cached_endpoint_relaxation" and n_fast == 0:
        raise ValueError("no eligible fast steps in refresh window")

    interval_gain = derived_interval_gain(cfg, interval)
    gain = effective_base_gain(cfg, interval)
    if not math.isfinite(interval_gain) or not math.isfinite(gain):
        raise ValueError("effective gain must be finite")

    fraction = 1.0 - (1.0 - gain) ** n_fast
    if (
        cfg.dose_schedule == "bounded_refresh_fraction"
        and fraction > cfg.max_endpoint_fraction_per_refresh + 1e-12
    ):
        raise ValueError("configured schedule exceeds endpoint-fraction bound")

    first = eligible_measurements[0]
    first_observation_index = measurements.index(first) + 1
    return {
        "measurement_steps": measurements,
        "first_possible_active_endpoint_step": first,
        "observations_before_activation": first_observation_index,
        "measurements_before_start_step": sum(step < cfg.start_step for step in measurements),
        "fast_steps_per_refresh_window": n_fast,
        "derived_interval_gain": interval_gain,
        "effective_base_gain": gain,
        "maximum_endpoint_fraction_per_refresh": cfg.max_endpoint_fraction_per_refresh,
        "worst_case_endpoint_fraction_per_refresh": fraction,
        "per_step_frobenius_cap": cfg.max_relative_frobenius_change_per_step,
        "cumulative_refresh_cap": cfg.max_cumulative_relative_frobenius_change_per_refresh,
        "expected_endpoint_opportunities": len(eligible_measurements),
    }


def _side_from_any(value: Any) -> AdaptiveAlphaSideConfig:
    if isinstance(value, AdaptiveAlphaSideConfig):
        return value
    if isinstance(value, dict):
        return AdaptiveAlphaSideConfig(**value)
    raise ValueError("adaptive side configuration must be a mapping")


def _validate_side(
    side: AdaptiveAlphaSideConfig,
    prefix: str,
    *,
    piecewise: bool = False,
) -> None:
    if side.deadband < 0:
        raise ValueError(f"{prefix}.deadband must be nonnegative")
    if side.full_strength_distance <= side.deadband:
        raise ValueError(f"{prefix}.full_strength_distance must exceed deadband")
    if not 0.0 <= side.max_hardness <= 1.0:
        raise ValueError(f"{prefix}.max_hardness must be in [0,1]")
    if side.response_curve not in RESPONSES:
        raise ValueError(f"{prefix}.response_curve must be linear or smoothstep")
    if piecewise or side.piecewise_points:
        validate_piecewise_points(
            side.piecewise_points,
            prefix=f"{prefix}.piecewise_points",
            distance=True,
        )


def validate_adaptive_config(cfg: AdaptiveWWPGDConfig, target_alpha: float) -> None:
    if not math.isfinite(target_alpha) or target_alpha <= 1.0:
        raise ValueError("wwpgd.target_alpha must be finite and greater than 1")
    if cfg.apply_mode not in {"event_projection", "cached_endpoint_relaxation"}:
        raise ValueError(
            "wwpgd.adaptive.apply_mode must be event_projection or cached_endpoint_relaxation"
        )
    if cfg.measurement_source not in {"explicit_interval", "evaluation_interval"}:
        raise ValueError(
            "wwpgd.adaptive.measurement_source must be explicit_interval or evaluation_interval"
        )
    if cfg.measurement_interval is not None and cfg.measurement_interval <= 0:
        raise ValueError(
            "wwpgd.adaptive.measurement_interval must be positive when supplied"
        )
    if cfg.measurement_source == "explicit_interval" and cfg.measurement_interval is None:
        raise ValueError("explicit_interval requires wwpgd.adaptive.measurement_interval")
    if cfg.apply_interval <= 0:
        raise ValueError("wwpgd.adaptive.apply_interval must be positive")
    if not 0 <= cfg.max_per_step_gain <= 1:
        raise ValueError("wwpgd.adaptive.max_per_step_gain must be in [0,1]")
    if (
        cfg.max_relative_frobenius_change_per_step is not None
        and cfg.max_relative_frobenius_change_per_step <= 0
    ):
        raise ValueError(
            "wwpgd.adaptive.max_relative_frobenius_change_per_step must be positive"
        )
    if cfg.dose_schedule not in DOSE_SCHEDULES:
        raise ValueError("wwpgd.adaptive.dose_schedule is invalid")
    if not 0 < cfg.max_endpoint_fraction_per_refresh <= 1:
        raise ValueError(
            "wwpgd.adaptive.max_endpoint_fraction_per_refresh must be in (0,1]"
        )
    if cfg.max_cumulative_relative_frobenius_change_per_refresh <= 0:
        raise ValueError("wwpgd.adaptive cumulative refresh cap must be positive")
    if cfg.endpoint_stop_relative_distance < 0:
        raise ValueError(
            "wwpgd.adaptive.endpoint_stop_relative_distance must be nonnegative"
        )
    if cfg.max_endpoint_age_steps <= 0:
        raise ValueError("wwpgd.adaptive.max_endpoint_age_steps must be positive")
    if cfg.stale_distance_multiplier < 1:
        raise ValueError(
            "wwpgd.adaptive.stale_distance_multiplier must be at least 1"
        )
    if cfg.mode not in {"uniform", "alpha_linear", "alpha_piecewise", "alpha_distance"}:
        raise ValueError(f"unknown wwpgd.adaptive.mode {cfg.mode}")
    if cfg.direction not in DIRECTIONS:
        raise ValueError(
            "wwpgd.adaptive.direction must be above_target, below_target, or both"
        )
    if cfg.response_curve not in RESPONSES:
        raise ValueError(
            "wwpgd.adaptive.response_curve must be linear or smoothstep"
        )
    if cfg.start_step < 0 or cfg.min_observations < 0 or cfg.cooldown_events < 0:
        raise ValueError("wwpgd.adaptive step/count fields must be nonnegative")
    if not 0.0 <= cfg.alpha_ema_beta < 1.0:
        raise ValueError("wwpgd.adaptive.alpha_ema_beta must satisfy 0 <= beta < 1")
    if cfg.deadband_above_target < 0 or not 0 <= cfg.max_hardness <= 1:
        raise ValueError("wwpgd.adaptive legacy hardness fields are invalid")
    if cfg.max_D is not None and cfg.max_D < 0:
        raise ValueError("wwpgd.adaptive.max_D must be nonnegative")
    if (
        cfg.max_relative_frobenius_change is not None
        and cfg.max_relative_frobenius_change <= 0
    ):
        raise ValueError(
            "wwpgd.adaptive.max_relative_frobenius_change must be positive"
        )
    if cfg.mode == "alpha_linear" and not (
        cfg.full_strength_alpha > target_alpha + cfg.deadband_above_target
    ):
        raise ValueError(
            "wwpgd.adaptive.full_strength_alpha must exceed target_alpha + deadband"
        )
    if cfg.mode == "alpha_piecewise":
        validate_piecewise_points(cfg.piecewise_points)
    if cfg.mode == "alpha_distance":
        above = _side_from_any(cfg.above_target)
        below = _side_from_any(cfg.below_target)
        _validate_side(
            above,
            "wwpgd.adaptive.above_target",
            piecewise=bool(above.piecewise_points),
        )
        _validate_side(
            below,
            "wwpgd.adaptive.below_target",
            piecewise=bool(below.piecewise_points),
        )
    for matrix_name, override in cfg.matrix_type_overrides.items():
        if matrix_name not in MATRIX_TYPES:
            raise ValueError(f"unknown wwpgd.adaptive matrix type {matrix_name}")
        _validate_override(override)
    for pattern, override in cfg.layer_overrides.items():
        if not pattern:
            raise ValueError("wwpgd.adaptive.layer_overrides keys must be nonempty")
        _validate_override(override)


def _validate_override(override: dict[str, Any]) -> None:
    bad = set(override) - OVERRIDE_FIELDS
    if bad:
        raise ValueError(
            f"unknown wwpgd.adaptive override key(s): {', '.join(sorted(bad))}"
        )
    for side_name in ("above_target", "below_target"):
        if side_name in override and override[side_name] is not None:
            side = _side_from_any(override[side_name])
            _validate_side(
                side,
                f"wwpgd.adaptive.{side_name}",
                piecewise=bool(side.piecewise_points),
            )


def validate_piecewise_points(
    points: list[list[float]],
    prefix: str = "wwpgd.adaptive.piecewise_points",
    *,
    distance: bool = False,
) -> None:
    if len(points) < 2:
        raise ValueError(f"{prefix} requires at least two points")
    previous = -math.inf
    for x_value, hardness in points:
        x_value = float(x_value)
        hardness = float(hardness)
        if distance and x_value < 0:
            raise ValueError(f"{prefix} distances must be nonnegative")
        if not x_value > previous:
            raise ValueError(f"{prefix} values must be strictly increasing")
        if not 0 <= hardness <= 1:
            raise ValueError(f"{prefix} hardness values must be in [0,1]")
        previous = x_value


def matrix_type(layer_name: str) -> str:
    suffixes = {
        "attn.key": "W_K",
        "attn.query": "W_Q",
        "attn.value": "W_V",
        "attn.proj": "W_O",
        "mlp.0": "W_MLP_IN",
        "mlp.2": "W_MLP_OUT",
    }
    return next(
        (matrix_name for suffix, matrix_name in suffixes.items() if layer_name.endswith(suffix)),
        "",
    )


def block_index(layer_name: str) -> int | None:
    match = re.match(r"blocks\.(\d+)\.", layer_name)
    return int(match.group(1)) if match else None


def _merge_side(
    base: AdaptiveAlphaSideConfig | dict[str, Any],
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    resolved = asdict(_side_from_any(base))
    if override:
        for key, value in override.items():
            if value is not None:
                resolved[key] = value
    return resolved


def _specificity(pattern: str) -> tuple[int, int, str]:
    literal = sum(1 for character in pattern if character not in "*?[]")
    return literal, len(pattern), pattern


def resolve_layer_config(
    global_cfg: AdaptiveWWPGDConfig,
    layer_name: str,
    target_alpha: float,
) -> dict[str, Any]:
    resolved = asdict(global_cfg)
    resolved["target_alpha"] = target_alpha
    resolved["above_target"] = asdict(_side_from_any(global_cfg.above_target))
    resolved["below_target"] = asdict(_side_from_any(global_cfg.below_target))

    def apply(override: dict[str, Any]) -> None:
        if "target_alpha" in override:
            raise ValueError("per-layer target_alpha overrides are not supported")
        for key, value in override.items():
            if value is None:
                continue
            if key in {"above_target", "below_target"}:
                resolved[key] = _merge_side(resolved[key], value)
            else:
                resolved[key] = value

    resolved_matrix_type = matrix_type(layer_name)
    if resolved_matrix_type in global_cfg.matrix_type_overrides:
        apply(global_cfg.matrix_type_overrides[resolved_matrix_type])
    matching_globs = [
        pattern
        for pattern in global_cfg.layer_overrides
        if any(character in pattern for character in "*?[")
        and fnmatch.fnmatchcase(layer_name, pattern)
    ]
    for pattern in sorted(matching_globs, key=_specificity):
        apply(global_cfg.layer_overrides[pattern])
    if layer_name in global_cfg.layer_overrides:
        apply(global_cfg.layer_overrides[layer_name])
    return resolved


def _interp(points: list[list[float]], distance: float) -> float:
    resolved_points = [(float(x), float(hardness)) for x, hardness in points]
    if distance <= resolved_points[0][0]:
        return resolved_points[0][1]
    if distance >= resolved_points[-1][0]:
        return resolved_points[-1][1]
    for (x0, h0), (x1, h1) in zip(resolved_points, resolved_points[1:]):
        if x0 <= distance <= x1:
            return h0 + (distance - x0) / (x1 - x0) * (h1 - h0)
    return resolved_points[-1][1]


def hardness_for_alpha(
    alpha: float,
    cfg: dict[str, Any],
) -> tuple[float, float, float]:
    """Return layer hardness, normalized distance, and active boundary.

    ``uniform`` is the fixed-strength mode.  It now respects ``max_hardness``
    rather than silently forcing full strength.
    """
    target = float(cfg["target_alpha"])
    if cfg.get("mode") == "uniform":
        hardness = max(0.0, min(1.0, float(cfg.get("max_hardness", 1.0))))
        return hardness, math.nan, target
    if not math.isfinite(float(alpha)):
        return 0.0, math.nan, target
    if cfg.get("mode") == "alpha_piecewise":
        hardness = _interp(cfg["piecewise_points"], float(alpha))
        return min(hardness, float(cfg.get("max_hardness", 1.0))), math.nan, target
    if cfg.get("mode") != "alpha_distance":
        boundary = target + float(cfg.get("deadband_above_target", 0.0))
        denominator = float(cfg["full_strength_alpha"]) - boundary
        normalized = (
            max(0.0, min(1.0, (float(alpha) - boundary) / denominator))
            if denominator > 0
            else 0.0
        )
        response = (
            normalized
            if cfg.get("response_curve") == "linear"
            else normalized * normalized * (3 - 2 * normalized)
        )
        return min(response, float(cfg.get("max_hardness", 1.0))), normalized, boundary

    signed_error = float(alpha) - target
    if signed_error == 0:
        return 0.0, 0.0, target
    side_name = "above_target" if signed_error > 0 else "below_target"
    if cfg.get("direction") not in {side_name, "both"}:
        return 0.0, 0.0, target
    side = cfg[side_name]
    distance = abs(signed_error)
    if not side.get("enabled", True) or distance <= float(side["deadband"]):
        return 0.0, 0.0, target
    if side.get("piecewise_points"):
        response = _interp(side["piecewise_points"], distance)
        normalized = math.nan
    else:
        normalized = max(
            0.0,
            min(
                1.0,
                (distance - float(side["deadband"]))
                / (float(side["full_strength_distance"]) - float(side["deadband"])),
            ),
        )
        response = (
            normalized
            if side.get("response_curve") == "linear"
            else normalized * normalized * (3 - 2 * normalized)
        )
    max_hardness = float(side.get("max_hardness", 1.0))
    return (
        max(0.0, min(max_hardness, max_hardness * response)),
        normalized,
        target,
    )


class AdaptiveWWPGDController:
    def __init__(self, cfg: AdaptiveWWPGDConfig, target_alpha: float):
        self.cfg = cfg
        self.target_alpha = target_alpha
        self.state = {
            "observation_count": {},
            "latest_raw_alpha": {},
            "alpha_ema": {},
            "last_projection_event": {},
            "last_applied_hardness": {},
            "last_alpha_side": {},
            "last_signed_alpha_error": {},
            "last_alpha_distance": {},
            "completed_scheduled_event_indexes": [],
        }

    def state_dict(self) -> dict[str, Any]:
        return {"version": CONTROLLER_VERSION, **self.state}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if state:
            self.state.update({key: value for key, value in state.items() if key in self.state})

    def observe(
        self,
        layer_name: str,
        alpha: float,
        *,
        beta: float | None = None,
    ) -> tuple[int, float]:
        count = int(self.state["observation_count"].get(layer_name, 0)) + 1
        self.state["observation_count"][layer_name] = count
        self.state["latest_raw_alpha"][layer_name] = alpha
        beta = self.cfg.alpha_ema_beta if beta is None else beta
        if not 0 <= beta < 1:
            raise ValueError("alpha EMA beta must satisfy 0 <= beta < 1")
        previous = self.state["alpha_ema"].get(layer_name)
        ema = (
            alpha
            if beta == 0 or previous is None or not math.isfinite(float(previous))
            else beta * float(previous) + (1 - beta) * alpha
        )
        self.state["alpha_ema"][layer_name] = ema
        signed_error = (
            float(ema) - self.target_alpha if math.isfinite(float(ema)) else math.nan
        )
        side = (
            "above_target"
            if signed_error > 0
            else "below_target"
            if signed_error < 0
            else "at_target"
        )
        self.state["last_alpha_side"][layer_name] = side
        self.state["last_signed_alpha_error"][layer_name] = signed_error
        self.state["last_alpha_distance"][layer_name] = (
            abs(signed_error) if math.isfinite(signed_error) else math.nan
        )
        return count, ema
