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
PRECEDENCE = ["global adaptive configuration", "matrix-type override", "matching layer-glob overrides", "exact layer-name override"]
OVERRIDE_FIELDS = {"enabled", "direction", "above_target", "below_target", "max_D", "max_relative_frobenius_change", "cooldown_events", "min_observations", "alpha_ema_beta"}


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
    max_hardness: float = 1.0
    max_D: float | None = None
    max_relative_frobenius_change: float | None = None
    cooldown_events: int = 0
    piecewise_points: list[list[float]] = field(default_factory=lambda: [[2.0, 0.0], [4.0, 1.0]])
    above_target: AdaptiveAlphaSideConfig = field(default_factory=lambda: AdaptiveAlphaSideConfig(deadband=0.4, full_strength_distance=2.0, response_curve="smoothstep"))
    below_target: AdaptiveAlphaSideConfig = field(default_factory=lambda: AdaptiveAlphaSideConfig(deadband=0.2, full_strength_distance=1.0, response_curve="smoothstep"))
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
    adaptive_cfg: AdaptiveWWPGDConfig, runtime_eval_interval: int
) -> int:
    """Resolve the cached endpoint sampling cadence in exactly one place."""
    value = (runtime_eval_interval if adaptive_cfg.measurement_source == "evaluation_interval"
             else adaptive_cfg.measurement_interval)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("endpoint measurement interval must be a positive integer")
    return value


def eligible_fast_steps(measurement_interval: int, apply_interval: int,
                        skip_measurement_step: bool = True) -> int:
    """Exact eligible applications in a full refresh window."""
    if measurement_interval <= 0 or apply_interval <= 0:
        return 0
    return sum(1 for offset in range(1, measurement_interval + 1)
               if offset % apply_interval == 0
               and not (skip_measurement_step and offset == measurement_interval))


def derived_interval_gain(cfg: AdaptiveWWPGDConfig, measurement_interval: int) -> float:
    n = eligible_fast_steps(measurement_interval, cfg.apply_interval,
                            cfg.skip_fast_apply_on_measurement_step)
    if n == 0:
        return 0.0
    if cfg.dose_schedule == "fixed_per_step_gain":
        return cfg.max_per_step_gain
    return 1.0 - (1.0 - cfg.max_endpoint_fraction_per_refresh) ** (1.0 / n)


def effective_base_gain(cfg: AdaptiveWWPGDConfig, measurement_interval: int) -> float:
    return min(cfg.max_per_step_gain, derived_interval_gain(cfg, measurement_interval))


def validate_adaptive_level_schedule(cfg: AdaptiveWWPGDConfig, total_optimizer_steps: int,
                                     measurement_interval: int | None = None) -> dict[str, Any]:
    """Validate and describe cached-endpoint cadence independently of model level."""
    interval = measurement_interval or cfg.measurement_interval
    if not interval:
        raise ValueError("measurement_interval is required for schedule validation")
    measurements = list(range(interval, total_optimizer_steps + 1, interval))
    after_start = [s for s in measurements if s >= cfg.start_step]
    if not after_start:
        raise ValueError("no possible endpoint measurement after start_step")
    if len(after_start) < cfg.min_observations:
        raise ValueError("min_observations cannot be reached")
    nfast = eligible_fast_steps(interval, cfg.apply_interval, cfg.skip_fast_apply_on_measurement_step)
    if cfg.apply_mode == "cached_endpoint_relaxation" and nfast == 0:
        raise ValueError("no eligible fast steps in refresh window")
    gain = effective_base_gain(cfg, interval)
    if not math.isfinite(gain):
        raise ValueError("effective gain must be finite")
    fraction = 1.0 - (1.0 - gain) ** nfast
    if cfg.dose_schedule == "bounded_refresh_fraction" and fraction > cfg.max_endpoint_fraction_per_refresh + 1e-12:
        raise ValueError("configured schedule exceeds endpoint-fraction bound")
    first = after_start[cfg.min_observations - 1]
    return {"measurement_steps": measurements, "first_possible_active_endpoint_step": first,
            "observations_before_activation": cfg.min_observations,
            "fast_steps_per_refresh_window": nfast, "derived_interval_gain": derived_interval_gain(cfg, interval),
            "effective_base_gain": gain, "maximum_endpoint_fraction_per_refresh": cfg.max_endpoint_fraction_per_refresh,
            "worst_case_endpoint_fraction_per_refresh": fraction,
            "per_step_frobenius_cap": cfg.max_relative_frobenius_change_per_step,
            "cumulative_refresh_cap": cfg.max_cumulative_relative_frobenius_change_per_refresh,
            "expected_endpoint_opportunities": max(0, len(after_start) - cfg.min_observations + 1)}


def _side_from_any(value: Any) -> AdaptiveAlphaSideConfig:
    if isinstance(value, AdaptiveAlphaSideConfig):
        return value
    if isinstance(value, dict):
        return AdaptiveAlphaSideConfig(**value)
    raise ValueError("adaptive side configuration must be a mapping")


def _validate_side(side: AdaptiveAlphaSideConfig, prefix: str, *, piecewise: bool = False) -> None:
    if side.deadband < 0:
        raise ValueError(f"{prefix}.deadband must be nonnegative")
    if side.full_strength_distance <= side.deadband:
        raise ValueError(f"{prefix}.full_strength_distance must exceed deadband")
    if not 0.0 <= side.max_hardness <= 1.0:
        raise ValueError(f"{prefix}.max_hardness must be in [0,1]")
    if side.response_curve not in RESPONSES:
        raise ValueError(f"{prefix}.response_curve must be linear or smoothstep")
    if piecewise or side.piecewise_points:
        validate_piecewise_points(side.piecewise_points, prefix=f"{prefix}.piecewise_points", distance=True)


def validate_adaptive_config(cfg: AdaptiveWWPGDConfig, target_alpha: float) -> None:
    if not math.isfinite(target_alpha) or target_alpha <= 1.0:
        raise ValueError("wwpgd.target_alpha must be finite and greater than 1")
    if cfg.apply_mode not in {"event_projection", "cached_endpoint_relaxation"}:
        raise ValueError("wwpgd.adaptive.apply_mode must be event_projection or cached_endpoint_relaxation")
    if cfg.measurement_source not in {"explicit_interval", "evaluation_interval"}:
        raise ValueError("wwpgd.adaptive.measurement_source must be explicit_interval or evaluation_interval")
    if cfg.measurement_interval is not None and cfg.measurement_interval <= 0:
        raise ValueError("wwpgd.adaptive.measurement_interval must be positive when supplied")
    if cfg.measurement_source == "explicit_interval" and cfg.measurement_interval is None:
        raise ValueError("explicit_interval requires wwpgd.adaptive.measurement_interval")
    if cfg.apply_interval <= 0:
        raise ValueError("wwpgd.adaptive.apply_interval must be positive")
    if not 0 <= cfg.max_per_step_gain <= 1:
        raise ValueError("wwpgd.adaptive.max_per_step_gain must be in [0,1]")
    if cfg.max_relative_frobenius_change_per_step is not None and cfg.max_relative_frobenius_change_per_step <= 0:
        raise ValueError("wwpgd.adaptive.max_relative_frobenius_change_per_step must be positive")
    if cfg.dose_schedule not in DOSE_SCHEDULES:
        raise ValueError("wwpgd.adaptive.dose_schedule is invalid")
    if not 0 < cfg.max_endpoint_fraction_per_refresh <= 1:
        raise ValueError("wwpgd.adaptive.max_endpoint_fraction_per_refresh must be in (0,1]")
    if cfg.max_cumulative_relative_frobenius_change_per_refresh <= 0:
        raise ValueError("wwpgd.adaptive cumulative refresh cap must be positive")
    if cfg.endpoint_stop_relative_distance < 0:
        raise ValueError("wwpgd.adaptive.endpoint_stop_relative_distance must be nonnegative")
    if cfg.max_endpoint_age_steps <= 0:
        raise ValueError("wwpgd.adaptive.max_endpoint_age_steps must be positive")
    if cfg.stale_distance_multiplier < 1:
        raise ValueError("wwpgd.adaptive.stale_distance_multiplier must be at least 1")
    if cfg.mode not in {"uniform", "alpha_linear", "alpha_piecewise", "alpha_distance"}:
        raise ValueError(f"unknown wwpgd.adaptive.mode {cfg.mode}")
    if cfg.direction not in DIRECTIONS:
        raise ValueError("wwpgd.adaptive.direction must be above_target, below_target, or both")
    if cfg.response_curve not in RESPONSES:
        raise ValueError("wwpgd.adaptive.response_curve must be linear or smoothstep")
    if cfg.start_step < 0 or cfg.min_observations < 0 or cfg.cooldown_events < 0:
        raise ValueError("wwpgd.adaptive step/count fields must be nonnegative")
    if not 0.0 <= cfg.alpha_ema_beta < 1.0:
        raise ValueError("wwpgd.adaptive.alpha_ema_beta must satisfy 0 <= beta < 1")
    if cfg.deadband_above_target < 0 or not 0 <= cfg.max_hardness <= 1:
        raise ValueError("wwpgd.adaptive legacy hardness fields are invalid")
    if cfg.max_D is not None and cfg.max_D < 0:
        raise ValueError("wwpgd.adaptive.max_D must be nonnegative")
    if cfg.max_relative_frobenius_change is not None and cfg.max_relative_frobenius_change <= 0:
        raise ValueError("wwpgd.adaptive.max_relative_frobenius_change must be positive")
    if cfg.mode == "alpha_linear" and not (cfg.full_strength_alpha > target_alpha + cfg.deadband_above_target):
        raise ValueError("wwpgd.adaptive.full_strength_alpha must exceed target_alpha + deadband")
    if cfg.mode == "alpha_piecewise":
        validate_piecewise_points(cfg.piecewise_points)
    if cfg.mode == "alpha_distance":
        _validate_side(_side_from_any(cfg.above_target), "wwpgd.adaptive.above_target", piecewise=bool(_side_from_any(cfg.above_target).piecewise_points))
        _validate_side(_side_from_any(cfg.below_target), "wwpgd.adaptive.below_target", piecewise=bool(_side_from_any(cfg.below_target).piecewise_points))
    for mt, ov in cfg.matrix_type_overrides.items():
        if mt not in MATRIX_TYPES:
            raise ValueError(f"unknown wwpgd.adaptive matrix type {mt}")
        _validate_override(ov)
    for pat, ov in cfg.layer_overrides.items():
        if not pat:
            raise ValueError("wwpgd.adaptive.layer_overrides keys must be nonempty")
        _validate_override(ov)


def _validate_override(ov: dict[str, Any]) -> None:
    bad = set(ov) - OVERRIDE_FIELDS
    if bad:
        raise ValueError(f"unknown wwpgd.adaptive override key(s): {', '.join(sorted(bad))}")
    for side_name in ("above_target", "below_target"):
        if side_name in ov and ov[side_name] is not None:
            _validate_side(_side_from_any(ov[side_name]), f"wwpgd.adaptive.{side_name}", piecewise=bool(_side_from_any(ov[side_name]).piecewise_points))


def validate_piecewise_points(points: list[list[float]], prefix: str = "wwpgd.adaptive.piecewise_points", *, distance: bool = False) -> None:
    if len(points) < 2:
        raise ValueError(f"{prefix} requires at least two points")
    prev = -math.inf
    for x, h in points:
        x = float(x); h = float(h)
        if distance and x < 0:
            raise ValueError(f"{prefix} distances must be nonnegative")
        if not x > prev:
            raise ValueError(f"{prefix} values must be strictly increasing")
        if not 0 <= h <= 1:
            raise ValueError(f"{prefix} hardness values must be in [0,1]")
        prev = x


def matrix_type(layer_name: str) -> str:
    suffixes = {"attn.key": "W_K", "attn.query": "W_Q", "attn.value": "W_V", "attn.proj": "W_O", "mlp.0": "W_MLP_IN", "mlp.2": "W_MLP_OUT"}
    return next((t for s, t in suffixes.items() if layer_name.endswith(s)), "")


def block_index(layer_name: str) -> int | None:
    m = re.match(r"blocks\.(\d+)\.", layer_name)
    return int(m.group(1)) if m else None


def _merge_side(base: AdaptiveAlphaSideConfig | dict[str, Any], ov: dict[str, Any] | None) -> dict[str, Any]:
    d = asdict(_side_from_any(base))
    if ov:
        for k, v in ov.items():
            if v is not None:
                d[k] = v
    return d


def _specificity(pat: str) -> tuple[int, int, str]:
    literal = sum(1 for ch in pat if ch not in "*?[]")
    return literal, len(pat), pat


def resolve_layer_config(global_cfg: AdaptiveWWPGDConfig, layer_name: str, target_alpha: float) -> dict[str, Any]:
    d = asdict(global_cfg)
    d["target_alpha"] = target_alpha
    d["above_target"] = asdict(_side_from_any(global_cfg.above_target))
    d["below_target"] = asdict(_side_from_any(global_cfg.below_target))

    def apply(ov: dict[str, Any]) -> None:
        if "target_alpha" in ov:
            raise ValueError("per-layer target_alpha overrides are not supported")
        for k, v in ov.items():
            if v is None:
                continue
            if k in {"above_target", "below_target"}:
                d[k] = _merge_side(d[k], v)
            else:
                d[k] = v

    mt = matrix_type(layer_name)
    if mt in global_cfg.matrix_type_overrides:
        apply(global_cfg.matrix_type_overrides[mt])
    globs = [p for p in global_cfg.layer_overrides if any(ch in p for ch in "*?[") and fnmatch.fnmatchcase(layer_name, p)]
    for pat in sorted(globs, key=_specificity):
        apply(global_cfg.layer_overrides[pat])
    if layer_name in global_cfg.layer_overrides:
        apply(global_cfg.layer_overrides[layer_name])
    return d


def _interp(points: list[list[float]], distance: float) -> float:
    pts = [(float(x), float(h)) for x, h in points]
    if distance <= pts[0][0]:
        return pts[0][1]
    if distance >= pts[-1][0]:
        return pts[-1][1]
    for (x0, h0), (x1, h1) in zip(pts, pts[1:]):
        if x0 <= distance <= x1:
            return h0 + (distance - x0) / (x1 - x0) * (h1 - h0)
    return pts[-1][1]


def hardness_for_alpha(alpha: float, cfg: dict[str, Any]) -> tuple[float, float, float]:
    target = float(cfg["target_alpha"])
    if cfg.get("mode") == "uniform":
        return 1.0, math.nan, target
    if not math.isfinite(float(alpha)):
        return 0.0, math.nan, target
    if cfg.get("mode") == "alpha_piecewise":
        h = _interp(cfg["piecewise_points"], float(alpha))
        return min(h, float(cfg.get("max_hardness", 1.0))), math.nan, target
    if cfg.get("mode") != "alpha_distance":
        dead = target + float(cfg.get("deadband_above_target", 0.0))
        denom = float(cfg["full_strength_alpha"]) - dead
        norm = max(0.0, min(1.0, (float(alpha) - dead) / denom)) if denom > 0 else 0.0
        h = norm if cfg.get("response_curve") == "linear" else norm * norm * (3 - 2 * norm)
        return min(h, float(cfg.get("max_hardness", 1.0))), norm, dead
    signed = float(alpha) - target
    if signed == 0:
        return 0.0, 0.0, target
    side_name = "above_target" if signed > 0 else "below_target"
    if cfg.get("direction") not in {side_name, "both"}:
        return 0.0, 0.0, target
    side = cfg[side_name]
    distance = abs(signed)
    if not side.get("enabled", True) or distance <= float(side["deadband"]):
        return 0.0, 0.0, target
    if side.get("piecewise_points"):
        response = _interp(side["piecewise_points"], distance)
        norm = math.nan
    else:
        norm = max(0.0, min(1.0, (distance - float(side["deadband"])) / (float(side["full_strength_distance"]) - float(side["deadband"]))))
        response = norm if side.get("response_curve") == "linear" else norm * norm * (3 - 2 * norm)
    max_h = float(side.get("max_hardness", 1.0))
    return max(0.0, min(max_h, max_h * response)), norm, target


class AdaptiveWWPGDController:
    def __init__(self, cfg: AdaptiveWWPGDConfig, target_alpha: float):
        self.cfg = cfg
        self.target_alpha = target_alpha
        self.state = {"observation_count": {}, "latest_raw_alpha": {}, "alpha_ema": {}, "last_projection_event": {}, "last_applied_hardness": {}, "last_alpha_side": {}, "last_signed_alpha_error": {}, "last_alpha_distance": {}, "completed_scheduled_event_indexes": []}

    def state_dict(self):
        return {"version": CONTROLLER_VERSION, **self.state}

    def load_state_dict(self, state):
        if state:
            self.state.update({k: v for k, v in state.items() if k in self.state})

    def observe(self, layer_name: str, alpha: float, *, beta: float | None = None) -> tuple[int, float]:
        c = int(self.state["observation_count"].get(layer_name, 0)) + 1
        self.state["observation_count"][layer_name] = c
        self.state["latest_raw_alpha"][layer_name] = alpha
        beta = self.cfg.alpha_ema_beta if beta is None else beta
        if not 0 <= beta < 1:
            raise ValueError("alpha EMA beta must satisfy 0 <= beta < 1")
        prev = self.state["alpha_ema"].get(layer_name)
        ema = alpha if beta == 0 or prev is None or not math.isfinite(float(prev)) else beta * float(prev) + (1 - beta) * alpha
        self.state["alpha_ema"][layer_name] = ema
        signed = float(ema) - self.target_alpha if math.isfinite(float(ema)) else math.nan
        side = "above_target" if signed > 0 else "below_target" if signed < 0 else "at_target"
        self.state["last_alpha_side"][layer_name] = side
        self.state["last_signed_alpha_error"][layer_name] = signed
        self.state["last_alpha_distance"][layer_name] = abs(signed) if math.isfinite(signed) else math.nan
        return c, ema
