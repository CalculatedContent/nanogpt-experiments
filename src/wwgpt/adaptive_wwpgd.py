from __future__ import annotations

import fnmatch, math, re
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import pandas as pd

CONTROLLER_VERSION = "adaptive_wwpgd_v1"
MATRIX_TYPES = {"W_K", "W_Q", "W_V", "W_O", "W_MLP_IN", "W_MLP_OUT"}
OVERRIDE_FIELDS = {"enabled","target_alpha","deadband_above_target","full_strength_alpha","max_hardness","response_curve","piecewise_points","max_D","max_relative_frobenius_change"}
PRECEDENCE = ["global adaptive defaults", "matrix-type override", "matching layer-glob override", "exact layer-name override"]

@dataclass(frozen=True)
class AdaptiveWWPGDOverride:
    enabled: bool | None = None
    target_alpha: float | None = None
    deadband_above_target: float | None = None
    full_strength_alpha: float | None = None
    max_hardness: float | None = None
    response_curve: str | None = None
    piecewise_points: list[list[float]] | None = None
    max_D: float | None = None
    max_relative_frobenius_change: float | None = None

@dataclass(frozen=True)
class AdaptiveWWPGDConfig:
    mode: str = "uniform"
    direction: str = "above_target"
    response_curve: str = "linear"
    start_step: int = 0
    min_observations: int = 1
    alpha_ema_beta: float = 0.0
    deadband_above_target: float = 0.0
    full_strength_alpha: float = 4.0
    max_hardness: float = 1.0
    max_D: float | None = None
    max_relative_frobenius_change: float | None = None
    cooldown_events: int = 0
    piecewise_points: list[list[float]] = field(default_factory=lambda: [[2.0, 0.0], [4.0, 1.0]])
    matrix_type_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    layer_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

def validate_adaptive_config(cfg: AdaptiveWWPGDConfig, target_alpha: float) -> None:
    if cfg.mode not in {"uniform","alpha_linear","alpha_piecewise"}: raise ValueError(f"unknown wwpgd.adaptive.mode {cfg.mode}")
    if cfg.direction != "above_target": raise ValueError("wwpgd.adaptive.direction currently supports above_target only")
    if cfg.response_curve not in {"linear","smoothstep"}: raise ValueError("wwpgd.adaptive.response_curve must be linear or smoothstep")
    if cfg.start_step < 0 or cfg.min_observations < 0 or cfg.cooldown_events < 0: raise ValueError("wwpgd.adaptive step/count fields must be nonnegative")
    if not 0.0 <= cfg.alpha_ema_beta < 1.0: raise ValueError("wwpgd.adaptive.alpha_ema_beta must satisfy 0 <= beta < 1")
    if cfg.deadband_above_target < 0: raise ValueError("wwpgd.adaptive.deadband_above_target must be nonnegative")
    if not 0 <= cfg.max_hardness <= 1: raise ValueError("wwpgd.adaptive.max_hardness must be in [0,1]")
    if cfg.max_D is not None and cfg.max_D < 0: raise ValueError("wwpgd.adaptive.max_D must be nonnegative")
    if cfg.max_relative_frobenius_change is not None and cfg.max_relative_frobenius_change <= 0: raise ValueError("wwpgd.adaptive.max_relative_frobenius_change must be positive")
    if cfg.mode == "alpha_linear" and not (cfg.full_strength_alpha > target_alpha + cfg.deadband_above_target): raise ValueError("wwpgd.adaptive.full_strength_alpha must exceed target_alpha + deadband")
    validate_piecewise_points(cfg.piecewise_points)
    for mt, ov in cfg.matrix_type_overrides.items():
        if mt not in MATRIX_TYPES: raise ValueError(f"unknown wwpgd.adaptive matrix type {mt}")
        _validate_override(ov)
    for pat, ov in cfg.layer_overrides.items():
        if not pat: raise ValueError("wwpgd.adaptive.layer_overrides keys must be nonempty")
        _validate_override(ov)

def _validate_override(ov: dict[str, Any]) -> None:
    bad=set(ov)-OVERRIDE_FIELDS
    if bad: raise ValueError(f"unknown wwpgd.adaptive override key(s): {', '.join(sorted(bad))}")
    if "piecewise_points" in ov and ov["piecewise_points"] is not None: validate_piecewise_points(ov["piecewise_points"])

def validate_piecewise_points(points):
    if len(points) < 2: raise ValueError("wwpgd.adaptive.piecewise_points requires at least two points")
    prev=-math.inf
    for a,h in points:
        if not float(a) > prev: raise ValueError("wwpgd.adaptive.piecewise_points alpha values must be strictly increasing")
        if not 0 <= float(h) <= 1: raise ValueError("wwpgd.adaptive.piecewise_points hardness values must be in [0,1]")
        prev=float(a)

def matrix_type(layer_name: str) -> str:
    suffixes = {"attn.key":"W_K","attn.query":"W_Q","attn.value":"W_V","attn.proj":"W_O","mlp.0":"W_MLP_IN","mlp.2":"W_MLP_OUT"}
    for s,t in suffixes.items():
        if layer_name.endswith(s): return t
    return ""

def block_index(layer_name: str) -> int | None:
    m=re.match(r"blocks\.(\d+)\.", layer_name)
    return int(m.group(1)) if m else None

def resolve_layer_config(global_cfg: AdaptiveWWPGDConfig, layer_name: str, target_alpha: float) -> dict[str, Any]:
    d=asdict(global_cfg); d["target_alpha"] = target_alpha; mt=matrix_type(layer_name)
    def apply(ov):
        for k,v in ov.items():
            if v is not None: d[k]=v
    if mt and mt in global_cfg.matrix_type_overrides: apply(global_cfg.matrix_type_overrides[mt])
    for pat in sorted(global_cfg.layer_overrides):
        if any(ch in pat for ch in "*?[") and fnmatch.fnmatchcase(layer_name, pat): apply(global_cfg.layer_overrides[pat])
    if layer_name in global_cfg.layer_overrides: apply(global_cfg.layer_overrides[layer_name])
    return d

def hardness_for_alpha(alpha: float, cfg: dict[str, Any]) -> tuple[float, float, float]:
    if cfg.get("mode") == "uniform": return 1.0, math.nan, cfg["target_alpha"] + cfg.get("deadband_above_target",0.0)
    if not math.isfinite(float(alpha)): return 0.0, math.nan, cfg["target_alpha"] + cfg.get("deadband_above_target",0.0)
    if cfg.get("mode") == "alpha_piecewise":
        pts=[(float(a),float(h)) for a,h in cfg["piecewise_points"]]
        if alpha <= pts[0][0]: h=pts[0][1]
        elif alpha >= pts[-1][0]: h=pts[-1][1]
        else:
            h=pts[-1][1]
            for (a0,h0),(a1,h1) in zip(pts, pts[1:]):
                if a0 <= alpha <= a1:
                    t=(alpha-a0)/(a1-a0); h=h0+t*(h1-h0); break
        return min(h, float(cfg.get("max_hardness",1.0))), math.nan, cfg["target_alpha"] + cfg.get("deadband_above_target",0.0)
    dead=cfg["target_alpha"] + cfg.get("deadband_above_target",0.0)
    denom=cfg["full_strength_alpha"]-dead
    norm=max(0.0,min(1.0,(alpha-dead)/denom)) if denom>0 else 0.0
    h=norm if cfg.get("response_curve") == "linear" else norm*norm*(3-2*norm)
    return min(h,float(cfg.get("max_hardness",1.0))), norm, dead

class AdaptiveWWPGDController:
    def __init__(self, cfg: AdaptiveWWPGDConfig, target_alpha: float):
        self.cfg=cfg; self.target_alpha=target_alpha; self.state={"observation_count":{},"latest_raw_alpha":{},"alpha_ema":{},"last_projection_event":{},"last_applied_hardness":{}}
    def state_dict(self): return {"version": CONTROLLER_VERSION, **self.state}
    def load_state_dict(self, state):
        if state: self.state.update({k:v for k,v in state.items() if k in self.state})
    def observe(self, layer_name: str, alpha: float) -> tuple[int,float]:
        c=int(self.state["observation_count"].get(layer_name,0))+1; self.state["observation_count"][layer_name]=c; self.state["latest_raw_alpha"][layer_name]=alpha
        beta=self.cfg.alpha_ema_beta; prev=self.state["alpha_ema"].get(layer_name)
        ema=alpha if beta==0 or prev is None or not math.isfinite(float(prev)) else beta*float(prev)+(1-beta)*alpha
        self.state["alpha_ema"][layer_name]=ema; return c, ema
