from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import yaml
import numpy as np

from wwgpt.config import (
    DEFAULT_SEEDS,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    WWPGDConfig,
    MeasurementConfig,
    load_config,
)
from wwgpt.adaptive_wwpgd import AdaptiveWWPGDConfig, AdaptiveWWPGDController, CachedLayerEndpoint, CONTROLLER_VERSION, PRECEDENCE, resolve_layer_config, hardness_for_alpha, matrix_type, block_index, resolve_endpoint_measurement_interval
from wwgpt.optim import ARM_DISPLAY, SCHEDULER_IMPLEMENTATION, arm_name as make_arm_name, build_optimizer_bundle, apply_lr_schedule, optimizer_fingerprint, resolve_lr_decay_steps, resolve_warmup_steps
from wwgpt.data import NonRepeatingTokenReader, RandomWindowTokenReader, prepare_local_text, fixed_probe, random_probe, stable_seed
from wwgpt.model import GPT
from wwgpt.utils import environment, sha256_bytes, unique_dir, write_json
from wwgpt.scaling import resolve_optimizer_steps
from wwgpt.device import autocast_context, device_summary, memory_stats, optimizer_step, precision_policy, synchronize_device
from wwgpt.checkpointing import CODE_VERSION_COMPAT, assert_checkpoint_compatible, load_latest_checkpoint, rng_state, restore_rng_state, save_checkpoint, stable_hash, compatibility_mismatches
from wwgpt.ww import (
    apply_external_wwpgd,
    build_stock_wwpgd_candidate,
    external_wwpgd_config_from_experiment,
    fallback_spectral_summary,
    spectral_summary,
    composite_spectral_summary,
    weightwatcher_details,
    nonmutating_weightwatcher_details,
    measured_projection_spectral_rows,
    weightwatcher_details,
    weightwatcher_run_aggregates,
    alpha_measurement_rows,
    _ww_version,
    WWPGD_COMMIT,
    SCIENTIFIC_SCHEMA_VERSION,
    external_wwpgd_config_from_experiment,
    external_wwpgd_manifest_fields,
)

INTERVENTION_EXTENSIONS = {"wwpgd", "norm_matched_sham", "delayed_onset"}
CONTROL_EXTENSIONS = {"measurement_only", "norm_matched_sham", "delayed_onset"}



def resolved_stochastic_seeds(user_seed: int, level: int, token_multiplier: int, *, split: str = "train", optimizer_identity: str | None = None) -> dict[str, int]:
    """Resolve stochastic seeds from stable scientific identity only.

    Storage identifiers such as run IDs, pair IDs, paths, timestamps, and UUIDs must
    never enter this derivation. Optimizer identity is included only for the explicit
    optimizer-scoped stream.
    """
    base = ("wwgpt_scientific_seed_v1", int(user_seed), int(level), int(token_multiplier))
    seeds = {
        "model_init_seed": stable_seed(*base, "model_init"),
        "dropout_seed": stable_seed(*base, split, "dropout"),
        "train_reader_seed": stable_seed(*base, "train", "reader"),
        "train_eval_probe_seed_base": stable_seed(*base, "train", "eval_probe"),
        "val_eval_probe_seed_base": stable_seed(*base, "val", "eval_probe"),
    }
    if optimizer_identity is not None:
        seeds["optimizer_seed"] = stable_seed(*base, "optimizer", optimizer_identity)
    return seeds


def _initial_minibatch_indices(tokens, block_size: int, batch_size: int, sampling: str, reader_seed: int) -> list[int]:
    if sampling == "random_window":
        rng = np.random.default_rng(reader_seed)
        return [int(x) for x in rng.integers(0, len(tokens) - block_size, size=batch_size)]
    return list(range(0, batch_size * block_size, block_size))

def _repository_version() -> dict[str, object]:
    """Return the exact source-tree identity used by checkpoint compatibility."""
    root = Path(__file__).resolve().parents[2]
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}


def _select_resume_run(arm_dir: Path, expected_identity: dict[str, object], expected_compatibility: dict[str, object], *, allow_code_version_mismatch: bool = False) -> tuple[str, Path, dict]:
    if not arm_dir.exists():
        return "new", arm_dir, {"incompatible_runs": []}
    incomplete=[]; completed=[]; incompatible=[]
    for run in sorted([p for p in arm_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        if not (run / "checkpoints" / "latest.json").exists():
            continue
        try:
            manifest=json.loads((run / "manifest.json").read_text())
            data_manifest=json.loads((run / "data_manifest.json").read_text())
            tok_manifest=json.loads((run / "tokenizer_manifest.json").read_text())
            init=(run / "initialization_hash.txt").read_text().strip()
        except Exception as exc:
            incompatible.append({"run": str(run), "error": str(exc)})
            continue
        observed={
            "pair_id": manifest.get("pair_id"),
            "arm_name": manifest.get("arm_name", manifest.get("optimizer")),
            "seed": manifest.get("seed"),
            "configuration_hash": manifest.get("configuration_hash"),
            "data_hash": manifest.get("data_hash", data_manifest.get("corpus_hash")),
            "tokenizer_hash": manifest.get("tokenizer_hash", tok_manifest.get("tokenizer_hash")),
            "initialization_hash": manifest.get("initialization_hash", init),
            "optimizer_fingerprint": manifest.get("optimizer_fingerprint"),
            "immediate_projection_spectral": manifest.get("immediate_projection_spectral"),
            "resolved_optimizer_steps": manifest.get("resolved_optimizer_steps", manifest.get("optimizer_steps")),
        }
        identity_mm={k:{"expected": v, "found": observed.get(k)} for k,v in expected_identity.items() if observed.get(k) != v}
        try:
            checkpoint = load_latest_checkpoint(run)
            checkpoint_mm = compatibility_mismatches(checkpoint, expected_compatibility)
        except Exception as exc:
            incompatible.append({"run": str(run), "error": f"checkpoint verification failed: {exc}"}); continue
        blocking = {k: v for k, v in checkpoint_mm.items() if not (allow_code_version_mismatch and k in CODE_VERSION_COMPAT)}
        if identity_mm or blocking:
            incompatible.append({"run": str(run), "identity_mismatches": identity_mm, "checkpoint_mismatches": checkpoint_mm})
            continue
        item=(run, checkpoint_mm)
        if (run / "run_complete.json").exists():
            try:
                completion = json.loads((run / "run_complete.json").read_text())
                expected_steps = int(expected_identity.get("resolved_optimizer_steps", 0))
                checkpoint_step = int(checkpoint.get("current_step", checkpoint.get("step", -1)))
                if checkpoint_step < expected_steps or int(completion.get("optimizer_step_count", expected_steps)) < expected_steps:
                    raise ValueError(f"completion step mismatch: checkpoint={checkpoint_step}, expected={expected_steps}")
            except Exception as exc:
                incompatible.append({"run": str(run), "error": f"completion verification failed: {exc}"})
                continue
            completed.append(item)
        else:
            incomplete.append(item)
    if len(incomplete)>1:
        raise RuntimeError("multiple compatible incomplete runs found; refusing ambiguous resume: " + json.dumps([str(p) for p, _ in incomplete]))
    if completed:
        if len(completed)>1:
            raise RuntimeError("multiple compatible completed runs found; refusing ambiguous selection: " + json.dumps([str(p) for p, _ in completed]))
        return "complete", *completed[0]
    if incomplete:
        return "resume", *incomplete[0]
    return "new", arm_dir, {"incompatible_runs": incompatible}


class TrainingExtension:
    name = "base"

    def after_optimizer_step(self, *, model, optimizer_step: int, total_optimizer_steps: int, tokens_seen: int, collect_pre_details: bool = False, **kwargs):
        return None, [], []


class NoExtension(TrainingExtension):
    name = "none"


class MeasurementOnlyExtension(TrainingExtension):
    """Run the intervention arm's event measurements without making a candidate."""
    name = "measurement_only"

    def __init__(self, interval: int):
        self.interval = interval

    def after_optimizer_step(self, *, model, optimizer_step: int, collect_pre_details: bool = False, **kwargs):
        if optimizer_step % self.interval:
            return None, [], []
        # This wrapper restores every RNG stream and never mutates model weights.
        details = nonmutating_weightwatcher_details(model, randomize=False)
        return details, [], []


@dataclass
class FastRelaxationResult:
    all_rows: list[dict[str, object]]
    changed_rows: list[dict[str, object]]
    changed_layer_count: int
    changed_step: bool


@dataclass
class EndpointMeasurementResult:
    pre_projection_details: object | None
    measurement_rows: list[dict[str, object]]
    controller_rows: list[dict[str, object]]
    stock_wwpgd_invoked: bool
    endpoint_activation_count: int
    measurement_index: int


class WWPGDExtension(TrainingExtension):
    name = "wwpgd"

    def __init__(self, cfg: WWPGDConfig, interval: int = 1, *, control: str = "wwpgd", scientific_seed: int | None = None):
        interval = _validate_wwpgd_interval(interval, source="WWPGDExtension.interval")
        self.cfg = cfg
        self.control = control
        self.scientific_seed = scientific_seed
        self.interval = interval
        self.controller = AdaptiveWWPGDController(getattr(cfg, "adaptive", AdaptiveWWPGDConfig()), cfg.target_alpha)
        self.decision_rows: list[dict[str, object]] = []
        self.endpoint_cache: dict[str, CachedLayerEndpoint] = {}
        self.measurement_rows: list[dict[str, object]] = []
        self.relaxation_rows: list[dict[str, object]] = []
        self.counters = {"measurement_count": 0, "candidate_generation_count": 0,
                         "fast_control_step_count": 0, "changed_fast_control_step_count": 0,
                         "endpoint_convergence_count": 0, "endpoint_invalidation_count": 0,
                         "last_measurement_step": None, "next_measurement_step": None}

    def state_dict(self) -> dict[str, object]:
        return {"adaptive_controller": self.controller.state_dict(), "decision_rows": list(self.decision_rows),
                "endpoint_cache": self.endpoint_cache, "measurement_rows": self.measurement_rows,
                "relaxation_rows": self.relaxation_rows, "counters": self.counters,
                "controller_version": CONTROLLER_VERSION, "adapter_mode": self.cfg.adaptive.apply_mode}

    def load_state_dict(self, state: dict[str, object] | None) -> None:
        if not state: return
        self.controller.load_state_dict(state.get("adaptive_controller", state))
        self.decision_rows = list(state.get("decision_rows", []))
        self.endpoint_cache = dict(state.get("endpoint_cache", {}))
        self.measurement_rows = list(state.get("measurement_rows", []))
        self.relaxation_rows = list(state.get("relaxation_rows", []))
        self.counters.update(state.get("counters", {}))

    def after_optimizer_step(
        self,
        *,
        model,
        optimizer_step: int,
        total_optimizer_steps: int,
        tokens_seen: int,
        collect_pre_details: bool = False,
        seed: int = 0,
        pair_id: str = "",
        base_optimizer: str = "",
        arm_name: str = "",
        measurement_interval: int | None = None,
    ) -> tuple[object | None, list[dict[str, object]], list[dict[str, object]]]:
        if getattr(getattr(self.cfg, "adaptive", None), "apply_mode", "event_projection") == "cached_endpoint_relaxation":
            result = self.after_optimizer_step_fast(model=model, optimizer_step=optimizer_step,
                                                  total_optimizer_steps=total_optimizer_steps,
                                                  measurement_interval=measurement_interval)
            return None, result.changed_rows, result.all_rows
        if optimizer_step % self.interval != 0:
            return None, [], []
        if self.control == "delayed_onset" and optimizer_step < int(self.cfg.delayed_onset_step or 1):
            return None, [], []
        event = optimizer_step // self.interval - 1
        frac = max(0.0, min(1.0, optimizer_step / max(1, total_optimizer_steps)))
        adaptive_cfg = getattr(self.cfg, "adaptive", AdaptiveWWPGDConfig())
        mode = adaptive_cfg.mode
        if mode == "uniform":
            details = weightwatcher_details(model) if collect_pre_details else None
            call_kwargs = {}
            if self.control == "norm_matched_sham":
                call_kwargs["sham_seed"] = self.scientific_seed
            rows = apply_external_wwpgd(model, event_index=event, scheduled_token_fraction=frac, actual_step=optimizer_step, actual_tokens_seen=tokens_seen, cfg=external_wwpgd_config_from_experiment(self.cfg), **call_kwargs)
            return details, rows, []
        # Observation-only gates use ordinary WeightWatcher; eligible adaptive events use the single stock WW_PGD ww_logs details.
        observation_only = (not bool(adaptive_cfg.enabled)) or optimizer_step < adaptive_cfg.start_step
        details = weightwatcher_details(model) if observation_only else None
        if event < self.cfg.warmup_events:
            global_event_hardness = 0.0
        elif event >= self.cfg.warmup_events + self.cfg.ramp_events:
            global_event_hardness = 1.0
        else:
            global_event_hardness = (event - self.cfg.warmup_events + 1) / max(self.cfg.ramp_events, 1)
        layer_hardness: dict[str, float] = {}
        layer_limits: dict[str, float | None] = {}
        decisions=[]
        candidate = None
        from wwgpt.ww import alpha_measurement_exclusion_reason, external_projected_layer_names, _match_ww_row
        if not observation_only and global_event_hardness > 0.0:
            candidate = build_stock_wwpgd_candidate(model, event_index=event, actual_step=optimizer_step, cfg=external_wwpgd_config_from_experiment(self.cfg))
            details = candidate.pre_projection_details
        elif details is None:
            details = weightwatcher_details(model)
        for lname in external_projected_layer_names(model):
            row=_match_ww_row(details, lname) or {}
            raw=float(row.get("alpha", float("nan"))) if row else float("nan")
            count, ema = self.controller.observe(lname, raw) if math.isfinite(raw) else (int(self.controller.state["observation_count"].get(lname,0)), float("nan"))
            rcfg=resolve_layer_config(self.cfg.adaptive, lname, self.cfg.target_alpha)
            requested, norm, dead_high = hardness_for_alpha(ema, rcfg)
            signed = ema - self.cfg.target_alpha if math.isfinite(float(ema)) else float("nan")
            alpha_side = "above_target" if signed > 0 else "below_target" if signed < 0 else "at_target"
            side_cfg = rcfg.get(alpha_side, {}) if alpha_side in {"above_target", "below_target"} else {}
            skip=""; projected=False; trust=1.0; applied=requested*global_event_hardness
            xmin=float(row.get("xmin", float("nan"))) if row else float("nan")
            D=float(row.get("D", float("nan"))) if row else float("nan")
            tail=float("nan")
            quality_reason = alpha_measurement_exclusion_reason(
                {**dict(row), "spectral_estimator": "weightwatcher", "projected": True}, max_D=rcfg.get("max_D"),
                min_tail=self.cfg.min_tail, require_projected=True,
            )
            if not bool(self.cfg.adaptive.enabled): skip="controller_disabled"
            elif not bool(rcfg.get("enabled", True)): skip="layer_disabled"
            elif alpha_side in {"above_target", "below_target"} and rcfg.get("direction") not in {alpha_side, "both"}: skip="direction_excluded"
            elif optimizer_step < self.cfg.adaptive.start_step: skip="before_start_step"
            elif quality_reason: skip=quality_reason
            elif not math.isfinite(ema): skip="invalid_smoothed_alpha"
            elif count < int(rcfg.get("min_observations", self.cfg.adaptive.min_observations)): skip="min_observations"
            elif lname in self.controller.state["last_projection_event"] and event - int(self.controller.state["last_projection_event"][lname]) <= self.cfg.adaptive.cooldown_events: skip="cooldown"
            elif applied <= 0: skip="zero_hardness"
            if skip:
                applied=0.0
            else:
                projected=True; layer_hardness[lname]=requested; layer_limits[lname]=rcfg.get("max_relative_frobenius_change")
            stock_changed = bool(candidate.stock_candidate_changed.get(lname, False)) if candidate is not None else False
            if not skip and not stock_changed: skip="stock_candidate_unchanged"
            dec={"seed":seed,"pair_id":pair_id,"base_optimizer":base_optimizer,"arm_name":arm_name,"optimizer_step":optimizer_step,"tokens_seen":tokens_seen,"projection_event":event,"layer_name":lname,"block":block_index(lname),"matrix_type":matrix_type(lname),"controller_mode":mode,"direction":rcfg.get("direction"),"alpha_side":alpha_side,"raw_alpha":raw,"smoothed_alpha":ema,"target_alpha":rcfg.get("target_alpha"),"signed_alpha_error":signed,"alpha_distance":abs(signed) if math.isfinite(signed) else float("nan"),"side_enabled":side_cfg.get("enabled"),"side_deadband":side_cfg.get("deadband"),"side_full_strength_distance":side_cfg.get("full_strength_distance"),"side_max_hardness":side_cfg.get("max_hardness"),"side_response_curve":side_cfg.get("response_curve"),"deadband_high":dead_high,"full_strength_alpha":rcfg.get("full_strength_alpha"),"normalized_alpha_distance":norm,"normalized_alpha_error":norm,"response_curve":rcfg.get("response_curve"),"layer_controller_hardness":requested,"layer_hardness_requested":requested,"global_event_hardness":global_event_hardness,"combined_hardness_requested":requested*global_event_hardness,"trust_region_scale":trust,"combined_hardness_applied":applied,"effective_blend_eta_requested":self.cfg.blend_eta*requested*global_event_hardness,"effective_blend_eta_applied":self.cfg.blend_eta*applied,"effective_cayley_eta_requested":self.cfg.cayley_eta*requested*global_event_hardness,"effective_cayley_eta_applied":self.cfg.cayley_eta*applied,"effective_blend_eta":self.cfg.blend_eta*applied,"effective_cayley_eta":self.cfg.cayley_eta*applied,"D":D,"xmin":xmin,"detX_num":row.get("detX_num", float("nan")),"num_evals":row.get("num_evals", float("nan")),"relative_frobenius_change_requested":float("nan"),"relative_frobenius_change_applied":float("nan"),"stock_candidate_changed":stock_changed,"maximum_blend_eta":self.cfg.blend_eta,"maximum_cayley_eta":self.cfg.cayley_eta,"wwpgd_adapter_mode":"stock_candidate_displacement_scaling_v1","projection_requested": requested > 0,"projection_attempted": False,"projected":projected,"skip_reason":skip,"observation_count":count,"last_projected_event":self.controller.state["last_projection_event"].get(lname),"controller_version":CONTROLLER_VERSION}
            decisions.append(dec)
        call_kwargs = {"sham_seed": self.scientific_seed} if self.control == "norm_matched_sham" else {}
        rows = apply_external_wwpgd(model, event_index=event, scheduled_token_fraction=frac, actual_step=optimizer_step, actual_tokens_seen=tokens_seen, cfg=external_wwpgd_config_from_experiment(self.cfg), layer_hardness=layer_hardness, global_event_hardness=global_event_hardness, layer_max_relative_change=layer_limits, stock_candidate=candidate, **call_kwargs) if layer_hardness else []
        byname={str(r.get("layer_name", "")):r for r in rows}
        for dec in decisions:
            r=byname.get(dec["layer_name"])
            if r:
                req=float(r.get("relative_frobenius_change_requested", r.get("relative_frobenius_change", 0.0)) or 0.0)
                app=float(r.get("relative_frobenius_change_applied", r.get("relative_frobenius_change", 0.0)) or 0.0)
                dec["relative_frobenius_change_requested"]=req; dec["relative_frobenius_change_applied"]=app
                dec["trust_region_scale"]=float(r.get("trust_region_scale", 1.0) or 1.0)
                dec["combined_hardness_applied"]=float(r.get("combined_hardness_applied", dec["combined_hardness_requested"] * dec["trust_region_scale"]) or 0.0)
                dec["effective_blend_eta_applied"]=float(r.get("effective_blend_eta_applied", self.cfg.blend_eta*dec["combined_hardness_applied"]) or 0.0)
                dec["effective_cayley_eta_applied"]=float(r.get("effective_cayley_eta_applied", self.cfg.cayley_eta*dec["combined_hardness_applied"]) or 0.0)
                dec["projection_attempted"]=bool(r.get("projection_attempted", True))
                dec["projected"]=bool(r.get("changed", True)) and app > 0
                if dec["projected"]:
                    self.controller.state["last_projection_event"][dec["layer_name"]]=event
                    self.controller.state["last_applied_hardness"][dec["layer_name"]]=dec["combined_hardness_applied"]
        self.decision_rows.extend(decisions)
        return (details if collect_pre_details else None), rows, decisions

    def _invalidate(self, endpoint: CachedLayerEndpoint, reason: str) -> None:
        if endpoint.active:
            self.counters["endpoint_invalidation_count"] += 1
        endpoint.active = False
        endpoint.invalidation_reason = reason

    def after_optimizer_step_fast(self, *, model, optimizer_step: int, total_optimizer_steps: int | None = None,
                                  measurement_interval: int | None = None, **_kwargs) -> FastRelaxationResult:
        """Apply cached residuals only; this hook performs no analysis, WW-PGD call, or SVD."""
        cfg = self.cfg.adaptive
        if cfg.apply_mode != "cached_endpoint_relaxation" or optimizer_step % cfg.apply_interval:
            return FastRelaxationResult([], [], 0, False)
        cadence = resolve_endpoint_measurement_interval(cfg, measurement_interval or self.interval)
        measurement_step = optimizer_step % cadence == 0 or (cfg.refresh_at_final_step and optimizer_step == total_optimizer_steps)
        if cfg.skip_fast_apply_on_measurement_step and measurement_step:
            return FastRelaxationResult([], [], 0, False)
        self.counters["fast_control_step_count"] += 1
        live = dict(__import__("wwgpt.ww", fromlist=["projected_matrix_modules"]).projected_matrix_modules(model))
        rows = []
        changed_step = False
        eps = 1e-12
        with torch.no_grad():
            for name, ep in self.endpoint_cache.items():
                if not ep.active:
                    continue
                if name not in live:
                    self._invalidate(ep, "layer_missing_from_model")
                    rows.append(self._terminal_fast_row(name, ep, optimizer_step, "layer_missing_from_model"))
                    continue
                weight = live[name]
                endpoint = ep.endpoint_tensor.to(weight.device, dtype=weight.dtype)
                age = optimizer_step - ep.measurement_step
                reason = ""
                if age > cfg.max_endpoint_age_steps: reason = "endpoint_age_exceeded"
                elif not torch.isfinite(weight).all() or not torch.isfinite(endpoint).all(): reason = "nonfinite_tensor"
                denom = max(float(torch.linalg.norm(weight.float())), eps)
                before = float(torch.linalg.norm((endpoint - weight).float())) / denom
                if not reason and before > cfg.stale_distance_multiplier * ep.initial_endpoint_relative_distance:
                    reason = "endpoint_distance_growth"
                if reason:
                    self._invalidate(ep, reason)
                    rows.append(self._terminal_fast_row(name, ep, optimizer_step, reason, before=before))
                    continue
                if before <= cfg.endpoint_stop_relative_distance:
                    ep.active = False; ep.invalidation_reason = "endpoint_converged"
                    self.counters["endpoint_convergence_count"] += 1
                    rows.append(self._terminal_fast_row(name, ep, optimizer_step, "endpoint_converged", before=before, converged=True))
                    continue
                gain = max(0.0, min(cfg.max_per_step_gain,
                                    cfg.max_per_step_gain * ep.alpha_hardness * ep.global_event_hardness))
                delta = gain * (endpoint - weight)
                requested = float(torch.linalg.norm(delta.float())) / denom
                limit = cfg.max_relative_frobenius_change_per_step
                scale = min(1.0, float(limit) / max(requested, eps)) if limit is not None else 1.0
                applied = scale * delta
                applied_rel = float(torch.linalg.norm(applied.float())) / denom
                old = weight.detach().clone()
                weight.add_(applied)
                after = float(torch.linalg.norm((endpoint - weight).float())) / max(float(torch.linalg.norm(weight.float())), eps)
                changed = not torch.equal(old, weight)
                changed_step |= changed
                ep.latest_endpoint_relative_distance = after
                ep.last_applied_step = optimizer_step
                ep.cumulative_applied_relative_change += applied_rel
                row = {"optimizer_step": optimizer_step, "layer_name": name, "block": block_index(name),
                       "matrix_type": matrix_type(name), "endpoint_measurement_step": ep.measurement_step,
                       "endpoint_age_steps": age, "cached_raw_alpha": ep.raw_alpha,
                       "cached_smoothed_alpha": ep.smoothed_alpha, "cached_alpha_distance": ep.alpha_distance,
                       "cached_alpha_hardness": ep.alpha_hardness, "global_event_hardness": ep.global_event_hardness,
                       "max_per_step_gain": cfg.max_per_step_gain, "controller_gain_requested": gain,
                       "controller_gain_applied": gain * scale,
                       "initial_endpoint_relative_distance": ep.initial_endpoint_relative_distance,
                       "endpoint_relative_distance_before": before, "endpoint_relative_distance_after": after,
                       "endpoint_progress_ratio_before": 1-before/max(ep.initial_endpoint_relative_distance, eps),
                       "endpoint_progress_ratio": 1-after/max(ep.initial_endpoint_relative_distance, eps),
                       "requested_relative_frobenius_change": requested,
                       "applied_relative_frobenius_change": applied_rel, "trust_region_limit": limit,
                       "trust_region_scale": scale, "changed": changed, "converged": False,
                       "invalidated": False, "invalidation_reason": "", "controller_version": CONTROLLER_VERSION,
                       "adapter_mode": "cached_endpoint_relaxation_v1", "action_type": "fast_endpoint_relaxation"}
                rows.append(row)
        if changed_step: self.counters["changed_fast_control_step_count"] += 1
        if cfg.log_every_fast_step:
            self.relaxation_rows.extend(rows)
        else:
            self.relaxation_rows.extend(r for r in rows if r.get("converged") or r.get("invalidated"))
        self.decision_rows.extend(rows)
        changed_rows = [row for row in rows if bool(row.get("changed"))]
        return FastRelaxationResult(rows, changed_rows, len(changed_rows), changed_step)

    def _terminal_fast_row(self, name, ep, step, reason, *, before=math.nan, converged=False):
        return {"optimizer_step": step, "layer_name": name, "endpoint_measurement_step": ep.measurement_step,
                "endpoint_age_steps": step - ep.measurement_step, "cached_raw_alpha": ep.raw_alpha,
                "cached_smoothed_alpha": ep.smoothed_alpha, "cached_alpha_hardness": ep.alpha_hardness,
                "global_event_hardness": ep.global_event_hardness,
                "endpoint_relative_distance_before": before, "endpoint_relative_distance_after": before,
                "changed": False, "converged": converged, "invalidated": not converged,
                "invalidation_reason": reason, "controller_version": CONTROLLER_VERSION,
                "adapter_mode": "cached_endpoint_relaxation_v1", "action_type": "fast_endpoint_relaxation"}

    def after_metrics_measurement(self, *, model, optimizer_step: int, total_optimizer_steps: int,
                                  tokens_seen: int = 0, force: bool = False, measurement_interval: int | None = None, **_metrics) -> EndpointMeasurementResult:
        """Refresh sample-and-hold alpha state and endpoints after reporting metrics."""
        cfg = self.cfg.adaptive
        if cfg.apply_mode != "cached_endpoint_relaxation": return EndpointMeasurementResult(None, [], [], False, 0, -1)
        interval = resolve_endpoint_measurement_interval(cfg, measurement_interval or self.interval)
        due = optimizer_step % interval == 0 or (cfg.refresh_at_final_step and optimizer_step == total_optimizer_steps)
        if not (due or force): return EndpointMeasurementResult(None, [], [], False, 0, -1)
        from wwgpt.ww import _module_by_name
        selected: dict[str, dict] = {}
        measurement_index = int(self.counters["measurement_count"])
        was_training = model.training
        if measurement_index < self.cfg.warmup_events:
            global_hardness = 0.0
        elif measurement_index >= self.cfg.warmup_events + self.cfg.ramp_events:
            global_hardness = 1.0
        else:
            global_hardness = (measurement_index - self.cfg.warmup_events + 1) / max(self.cfg.ramp_events, 1)
        def selector(mm, name, row=None):
            from wwgpt.ww import alpha_measurement_exclusion_reason, is_projected_layer
            if not is_projected_layer(name):
                return None
            data = row.to_dict() if hasattr(row, "to_dict") else dict(row or {})
            raw = float(data.get("alpha", math.nan))
            rcfg = resolve_layer_config(cfg, name, self.cfg.target_alpha)
            count, ema = self.controller.observe(name, raw, beta=float(rcfg["alpha_ema_beta"])) if math.isfinite(raw) else (0, math.nan)
            hardness, _, _ = hardness_for_alpha(ema, rcfg)
            D=float(data.get("D", math.nan)); xmin=float(data.get("xmin", math.nan))
            signed=ema-self.cfg.target_alpha if math.isfinite(ema) else math.nan
            info_side="above_target" if signed>0 else "below_target" if signed<0 else "at_target"
            quality_reason = alpha_measurement_exclusion_reason(
                {**data, "spectral_estimator": "weightwatcher", "projected": True},
                max_D=rcfg.get("max_D"), min_tail=self.cfg.min_tail,
            )
            reason = ""
            if not cfg.enabled: reason="controller_disabled"
            elif not rcfg.get("enabled", True): reason="layer_disabled"
            elif optimizer_step < cfg.start_step: reason="before_start_step"
            elif quality_reason: reason=quality_reason
            elif count < rcfg.get("min_observations", cfg.min_observations): reason="insufficient_observations"
            elif hardness <= 0: reason="inside_deadband"
            elif info_side not in {rcfg.get("direction"), "at_target"} and rcfg.get("direction") != "both": reason="direction_excluded"
            elif global_hardness <= 0: reason="zero_hardness"
            selected[name]={"row":data,"raw":raw,"ema":ema,"count":count,"hardness":hardness,
                            "signed":signed,"side":info_side,
                            "reason":reason}
            if reason:
                if name in self.endpoint_cache: self._invalidate(self.endpoint_cache[name], reason)
                return None
            return _module_by_name(mm, name)
        observation_only = not cfg.enabled or optimizer_step < cfg.start_step or global_hardness == 0
        candidate = None
        from wwgpt.ww import external_projected_layer_names
        expected_names = set(external_projected_layer_names(model))
        try:
            model.eval()
            if observation_only:
                from wwgpt.ww import _match_ww_row, external_projected_layer_names
                details = weightwatcher_details(model)
                for name in external_projected_layer_names(model):
                    matched = _match_ww_row(details, name)
                    selector(model, name, {} if matched is None else matched)
            else:
                candidate = build_stock_wwpgd_candidate(model, event_index=measurement_index,
                    actual_step=optimizer_step, cfg=external_wwpgd_config_from_experiment(self.cfg), layer_selector=selector)
                details = candidate.pre_projection_details
                self.counters["candidate_generation_count"] += 1
        finally:
            model.train(was_training)
        self.counters["measurement_count"] += 1
        self.counters["last_measurement_step"] = optimizer_step
        self.counters["next_measurement_step"] = optimizer_step + interval
        # The fresh WeightWatcher table is authoritative.  A missing current row
        # explicitly retires a cached endpoint rather than allowing stale motion.
        for name in sorted(expected_names - selected.keys()):
            reason = "missing_current_weightwatcher_row"
            if name in self.endpoint_cache:
                self._invalidate(self.endpoint_cache[name], reason)
            selected[name] = {"row": {}, "raw": math.nan, "ema": math.nan, "count": 0,
                              "hardness": 0.0, "signed": math.nan, "side": "at_target", "reason": reason}
        rows=[]
        for name, info in selected.items():
            data=info["row"]; changed=bool(candidate and candidate.stock_candidate_changed.get(name, False)); active=False
            reason=info["reason"] or ("" if changed else "stock_candidate_unchanged")
            if reason and name in self.endpoint_cache:
                self._invalidate(self.endpoint_cache[name], reason)
            initial=float(candidate.original_to_candidate_relative_change.get(name, 0.0)) if candidate else 0.0
            if not reason and candidate is not None and name in candidate.candidate_weights:
                endpoint=candidate.candidate_weights[name].detach().clone()
                original=candidate.original_weights[name].detach().clone()
                if cfg.cache_endpoint_on_cpu: endpoint=endpoint.cpu(); original=original.cpu()
                self.endpoint_cache[name]=CachedLayerEndpoint(name,endpoint,original,optimizer_step,measurement_index,
                    info["raw"],info["ema"],self.cfg.target_alpha,info["signed"],abs(info["signed"]),info["side"],
                    info["hardness"],global_hardness,info["hardness"]*global_hardness,initial,initial,initial,float(data.get("D",math.nan)),
                    float(data.get("xmin",math.nan)),float(data.get("detX_num",math.nan)),float(data.get("num_evals",math.nan)))
                active=True
            row={"optimizer_step":optimizer_step,"measurement_index":measurement_index,"layer_name":name,
                 "block":block_index(name),"matrix_type":matrix_type(name),"raw_alpha":info["raw"],
                 "smoothed_alpha":info["ema"],"target_alpha":self.cfg.target_alpha,"signed_alpha_error":info["signed"],
                 "alpha_distance":abs(info["signed"]),"alpha_side":info["side"],"alpha_hardness":info["hardness"],
                 "D":data.get("D"),"xmin":data.get("xmin"),"detX_num":data.get("detX_num"),"num_evals":data.get("num_evals"),
                 "selected_for_candidate":not bool(info["reason"]),"stock_candidate_changed":changed,
                 "initial_endpoint_relative_distance":initial,"global_event_hardness":global_hardness,"cache_activated":active,
                 "cache_invalidation_reason":reason,"skip_reason":reason,
                 "measurement_runtime":candidate.runtime if candidate else 0.0,
                 "controller_version":CONTROLLER_VERSION,"adapter_mode":"cached_endpoint_relaxation_v1",
                 "action_type":"slow_measurement"}
            rows.append(row)
        self.measurement_rows.extend(rows); self.decision_rows.extend(rows)
        return EndpointMeasurementResult(details, rows, rows, candidate is not None,
                                         sum(bool(r["cache_activated"]) for r in rows), measurement_index)


def _validate_wwpgd_interval(value: int | None, *, source: str) -> int:
    if value is None:
        return 1
    if isinstance(value, bool):
        raise ValueError(f"{source} must be a positive integer")
    try:
        interval = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a positive integer") from exc
    if interval != value and not (isinstance(value, str) and str(interval) == value):
        raise ValueError(f"{source} must be a positive integer")
    if interval < 1:
        raise ValueError(f"{source} must be a positive integer")
    return interval


def _resolve_wwpgd_interval(cfg: ExperimentConfig, extension_name: str, ww_interval: int | None) -> int:
    if extension_name not in INTERVENTION_EXTENSIONS | {"measurement_only"}:
        return 1
    return _validate_wwpgd_interval(ww_interval if ww_interval is not None else cfg.train.wwpgd_interval, source="WW-PGD interval")


def _expected_projection_optimizer_steps(total_optimizer_steps: int, interval: int, extension_name: str) -> list[int]:
    if extension_name not in INTERVENTION_EXTENSIONS:
        return []
    return list(range(interval, total_optimizer_steps + 1, interval))


def resolved_baseline_hyperparameters(cfg: ExperimentConfig, *, resolved_warmup_steps: int | None = None, resolved_lr_decay_steps: int | None = None, resolved_llrd_gamma: float | None = None) -> dict[str, object]:
    """Return the fully resolved nanoGPT baseline settings recorded in run metadata."""
    model = cfg.model
    train = cfg.train
    out: dict[str, object] = {
        "learning_rate": train.learning_rate,
        "weight_decay": train.weight_decay,
        "grad_clip": train.grad_clip,
        "adamw_betas": tuple(train.betas),
        "adamw_epsilon": train.epsilon,
        "lr_schedule": train.lr_schedule,
        "warmup_steps_requested": train.warmup_steps,
        "warmup_ratio": train.warmup_ratio,
        "lr_decay_steps_requested": train.lr_decay_steps,
        "min_lr_ratio": train.min_lr_ratio,
        "layer_lr": train.layer_lr,
        "llrd_gamma": resolved_llrd_gamma,
        "matrix_lr_multipliers": dict(train.matrix_lr_multipliers),
        "tie_weights": model.tie_weights,
        "init_mode": model.init_mode,
        "residual_projection_init_std": 0.02 / (2 * model.n_layer) ** 0.5,
        "causal_attention": True,
        "attention_implementation": "torch_scaled_dot_product_attention_is_causal",
        "attention_dropout": model.dropout,
        "residual_dropout": model.dropout,
        "embedding_dropout": model.dropout,
        "separate_qkv_projections": True,
        "linear_bias": model.linear_bias,
        "layernorm_bias": model.layernorm_bias,
        "model_architecture_version": model.model_architecture_version,
        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,
    }
    if resolved_warmup_steps is not None:
        out["resolved_warmup_steps"] = resolved_warmup_steps
    if resolved_lr_decay_steps is not None:
        out["resolved_lr_decay_steps"] = resolved_lr_decay_steps
    return out


def _gradient_norm(parameters) -> torch.Tensor:
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms), ord=2)

def _log_train_progress(message: str) -> None:
    print(f"[wwgpt run-multiseed] {message}", file=sys.stderr, flush=True)


def _write_csv(path: Path, rows: list[dict[str, object]], *, overwrite: bool = False) -> None:
    if not rows:
        return
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("w", newline="") as f:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_csv_union_schema(path: Path, rows: list[dict[str, object]], *, empty_fields: list[str] | None = None) -> None:
    """Persist append-only logical rows with a deterministic, resume-safe union schema."""
    existing_rows: list[dict[str, object]] = []
    existing_fields: list[str] = []
    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)
    pending = rows[len(existing_rows):]
    if not pending and path.exists():
        return
    fields = existing_fields + list(empty_fields or []) + [key for row in rows for key in row if key not in existing_fields]
    fields = list(dict.fromkeys(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(pending)


def _append_only_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Append missing rows without rewriting existing raw metric logs."""
    if not rows:
        return
    existing = 0
    existing_fields: list[str] | None = None
    if path.exists():
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing = sum(1 for _ in reader)
    pending = rows[existing:]
    if not pending:
        return
    if existing_fields:
        fieldnames = existing_fields
    else:
        fieldnames = list(rows[0])
        extras = [k for r in pending for k in r if k not in fieldnames]
        fieldnames = fieldnames + list(dict.fromkeys(extras))
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if existing == 0 and not existing_fields:
            w.writeheader()
        w.writerows(pending)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metric_row(*, checkpoint_path: Path, checkpoint_hash: str, step: int,
                           selection_metric: str, train_metrics: dict, validation_metrics: dict,
                           test_metrics: dict | None, probe_hashes: dict[str, str]) -> dict[str, object]:
    """Build a self-contained metric record for exactly one model artifact."""
    row: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path), "checkpoint_hash": checkpoint_hash,
        "checkpoint_sha256": checkpoint_hash, "selected_step": step,
        "selected_checkpoint_path": str(checkpoint_path), "selected_checkpoint_hash": checkpoint_hash,
        "selected_checkpoint_step": step, "selection_metric": selection_metric,
        "train_loss": train_metrics["loss"], "train_perplexity": train_metrics["perplexity"],
        "train_accuracy": train_metrics["top1_accuracy"], "train_top1_accuracy": train_metrics["top1_accuracy"],
        "validation_loss": validation_metrics["loss"],
        "validation_perplexity": validation_metrics["perplexity"],
        "validation_accuracy": validation_metrics["top1_accuracy"],
        "validation_top1_accuracy": validation_metrics["top1_accuracy"],
        "train_validation_gap": validation_metrics["loss"] - train_metrics["loss"], **probe_hashes,
    }
    if test_metrics is None:
        row.update({"test_loss": None, "test_perplexity": None, "test_accuracy": None,
                    "test_top1_accuracy": None, "train_test_gap": None, "test_evaluated": False})
    else:
        row.update({"test_loss": test_metrics["loss"], "test_perplexity": test_metrics["perplexity"],
                    "test_accuracy": test_metrics["top1_accuracy"],
                    "test_top1_accuracy": test_metrics["top1_accuracy"],
                    "train_test_gap": test_metrics["loss"] - train_metrics["loss"],
                    "test_evaluated": True})
    return row


def _perplexity_from_cross_entropy(loss: float) -> float:
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


def _metrics(loss: float, logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
    top1 = float((pred[..., 0] == y).float().mean())
    top5 = float((pred == y.unsqueeze(-1)).any(dim=-1).float().mean())
    return {
        "loss": loss,
        "perplexity": _perplexity_from_cross_entropy(loss),
        "bits_per_token": loss / math.log(2),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "token_error": 1 - top1,
    }


def _evaluate_probe_batches(
    model: GPT, probe_x: np.ndarray, probe_y: np.ndarray, device: torch.device
) -> tuple[dict[str, float], float]:
    loss_sum = 0.0
    token_count = 0
    top1_correct = 0
    top5_correct = 0
    for batch_x, batch_y in zip(probe_x, probe_y, strict=True):
        x = torch.tensor(batch_x, device=device)
        y = torch.tensor(batch_y, device=device)
        logits, loss = model(x, y)
        assert loss is not None
        tokens = int(y.numel())
        loss_sum += float(loss.detach().cpu()) * tokens
        token_count += tokens
        pred = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
        top1_correct += int((pred[..., 0] == y).sum().detach().cpu())
        top5_correct += int((pred == y.unsqueeze(-1)).any(dim=-1).sum().detach().cpu())
    mean_loss = loss_sum / max(token_count, 1)
    top1 = top1_correct / max(token_count, 1)
    top5 = top5_correct / max(token_count, 1)
    return {
        "loss": mean_loss,
        "perplexity": _perplexity_from_cross_entropy(mean_loss),
        "bits_per_token": mean_loss / math.log(2),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "token_error": 1 - top1,
    }, mean_loss


def run_single(
    run_parent: Path,
    optimizer_name: str,
    seed: int,
    cfg: ExperimentConfig,
    train_tokens: list[int],
    val_tokens: list[int],
    pair_id: str,
    max_steps: int | None = None,
    init_state: dict[str, torch.Tensor] | None = None,
) -> Path:
    torch.manual_seed(seed)
    if optimizer_name in {"adamw_wwpgd_reference", "adamw_wwpgd"}:
        base_optimizer, extension_name = "adamw", "wwpgd"
    elif optimizer_name in {"muon_wwpgd", "stableadamw_wwpgd", "stable_adamw_wwpgd"}:
        base_optimizer, extension_name = optimizer_name.removesuffix("_wwpgd"), "wwpgd"
    elif optimizer_name in {"adamw", "muon", "stableadamw", "stable_adamw"}:
        base_optimizer, extension_name = optimizer_name, getattr(cfg.wwpgd, "extension", "none")
    else:
        base_optimizer, extension_name = optimizer_name, getattr(cfg.wwpgd, "extension", "none")
    optimizer_name = make_arm_name(base_optimizer, extension_name)
    run_dir = unique_dir(run_parent / optimizer_name, "run")
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir()
    model_cfg = ModelConfig(
        **{**asdict(cfg.model), "vocab_size": max(train_tokens + val_tokens) + 1}
    )
    model = GPT(model_cfg)
    if init_state is not None:
        model.load_state_dict(init_state)
    init_hash = sha256_bytes(
        b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())
    )
    (run_dir / "initialization_hash.txt").write_text(init_hash)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=cfg.train.betas,
        eps=cfg.train.epsilon,
        weight_decay=cfg.train.weight_decay,
    )
    steps = max_steps or cfg.train.max_steps or 3
    reader = NonRepeatingTokenReader(train_tokens, model_cfg.block_size)
    val_reader = NonRepeatingTokenReader(val_tokens + train_tokens, model_cfg.block_size)
    metric_rows = []
    spectral_rows = []
    proj_rows = []
    controller_rows = []
    immediate_spectral_rows = []
    write_json(run_dir / "environment.json", environment())
    write_json(
        run_dir / "manifest.json",
        {
            "optimizer": optimizer_name,
            "base_optimizer": base_optimizer,
            "extension": extension_name,
            "arm_name": optimizer_name,
            "arm_display_name": ARM_DISPLAY[optimizer_name],
            "seed": seed,
            "pair_id": pair_id,
            "smoke_test": True,
            "valid_for_science": False,
            "parameter_report": model.report_dict(),
            "resolved_baseline_hyperparameters": resolved_baseline_hyperparameters(cfg),
        },
    )
    write_json(
        run_dir / "data_manifest.json",
        {
            "dataset": "local_text",
            "corpus_hash": sha256_bytes(bytes([x % 256 for x in train_tokens])),
        },
    )
    write_json(
        run_dir / "tokenizer_manifest.json",
        {"tokenizer": "char-smoke", "vocab_size": model_cfg.vocab_size},
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(asdict(cfg)))))
    write_json(run_dir / "config.json", json.loads(json.dumps(asdict(cfg))))
    torch.save(model.state_dict(), ckpt / f"initial_step_000000_{seed}.pt")
    _log_train_progress(
        f"starting smoke run optimizer={optimizer_name} seed={seed} pair={pair_id} steps={steps} output={run_dir}"
    )
    start = time.perf_counter()
    last_loss = 0.0
    for step in range(1, steps + 1):
        xb, yb = reader.next_batch(cfg.train.batch_size)
        x = torch.tensor(xb)
        y = torch.tensor(yb)
        _, loss = model(x, y)
        assert loss is not None
        opt.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        proj_time = 0.0
        if optimizer_name in {"adamw_wwpgd", "adamw_wwpgd_reference"}:
            pstart = time.perf_counter()
            proj_rows.extend(apply_external_wwpgd(model, event_index=step, actual_step=step, cfg=external_wwpgd_config_from_experiment(cfg.wwpgd)))
            proj_time = time.perf_counter() - pstart
        with torch.no_grad():
            vx, vy = val_reader.next_batch(cfg.train.batch_size)
            vlogits, vloss = model(torch.tensor(vx), torch.tensor(vy))
            assert vloss is not None
            tlogits, tloss = model(x, y)
            assert tloss is not None
        tm = _metrics(float(tloss.detach()), tlogits, y)
        vm = _metrics(float(vloss.detach()), vlogits, torch.tensor(vy))
        elapsed = time.perf_counter() - start
        last_loss = float(vloss.detach())
        metric_rows.append(
            {
                "step": step,
                "tokens_processed": step * cfg.train.batch_size * model_cfg.block_size,
                "elapsed_time": elapsed,
                "learning_rate": cfg.train.learning_rate,
                "gradient_norm": float(grad.detach()),
                "train_minibatch_loss": float(loss.detach()),
                "train_loss": tm["loss"],
                "val_loss": vm["loss"],
                "train_perplexity": tm["perplexity"],
                "val_perplexity": vm["perplexity"],
                "train_bits_per_token": tm["bits_per_token"],
                "val_bits_per_token": vm["bits_per_token"],
                "train_top1_accuracy": tm["top1_accuracy"],
                "val_top1_accuracy": vm["top1_accuracy"],
                "train_top5_accuracy": tm["top5_accuracy"],
                "val_top5_accuracy": vm["top5_accuracy"],
                "train_token_error": tm["token_error"],
                "val_token_error": vm["token_error"],
                "generalization_gap": vm["loss"] - tm["loss"],
                "tokens_per_second": (step * cfg.train.batch_size * model_cfg.block_size)
                / max(elapsed, 1e-9),
                "examples_per_second": (step * cfg.train.batch_size) / max(elapsed, 1e-9),
                "weightwatcher_overhead": 0.0,
                "projection_overhead": proj_time,
                "peak_memory": 0.0,
            }
        )
        spectral_rows.extend(
            fallback_spectral_summary(
                model,
                step=step,
                tokens_seen=step * cfg.train.batch_size * model_cfg.block_size,
                optimizer=optimizer_name,
                seed=seed,
                pair_id=pair_id,
            )
        )
        _log_train_progress(
            f"smoke progress optimizer={optimizer_name} seed={seed} step={step}/{steps} val_loss={last_loss:.4f} elapsed_s={elapsed:.1f}"
        )
        torch.save(
            {"model": model.state_dict(), "step": step}, ckpt / f"latest_step_{step:06d}_{seed}.pt"
        )
    torch.save(model.state_dict(), ckpt / f"final_step_{steps:06d}_{seed}.pt")
    torch.save(model.state_dict(), ckpt / f"best_val_step_{steps:06d}_{seed}.pt")
    _write_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "spectral.csv", spectral_rows)
    if optimizer_name in {"adamw_wwpgd", "adamw_wwpgd_reference"}:
        _write_csv(run_dir / "wwpgd_projection.csv", proj_rows)
    (run_dir / "events.jsonl").write_text(json.dumps({"event": "complete"}) + "\n")
    skip_counts={}
    for r in controller_rows:
        reason=str(r.get("skip_reason", ""))
        if reason: skip_counts[reason]=skip_counts.get(reason,0)+1
    applied=[float(r.get("combined_hardness_applied", 0.0) or 0.0) for r in controller_rows]
    rels=[float(r.get("relative_frobenius_change_applied", 0.0) or 0.0) for r in controller_rows if math.isfinite(float(r.get("relative_frobenius_change_applied", 0.0) or 0.0))]
    write_json(run_dir / "run_complete.json", {"step": steps, "final_val_loss": last_loss})
    _log_train_progress(
        f"completed smoke run optimizer={optimizer_name} seed={seed} steps={steps} final_val_loss={last_loss:.4f} output={run_dir}"
    )
    return run_dir


def smoke(root: Path, steps: int = 3, seeds: list[int] | None = None) -> Path:
    run_seeds = seeds or [1337]
    smoke_dir = unique_dir(root, "wwgpt_invalid_smoke")
    text = (
        "WeightWatcher PGD smoke corpus. This is not Tiny Shakespeare and is invalid for science. "
        * 400
    ).split(".")
    cfg = ExperimentConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=32, block_size=16, vocab_size=128),
        train=TrainConfig(batch_size=2, max_steps=steps, eval_interval=1),
        wwpgd=WWPGDConfig(enabled=True, extension="wwpgd"),
    )
    data = prepare_local_text(
        smoke_dir / "data",
        [t + "." for t in text],
        min_train_tokens=steps * cfg.train.batch_size * cfg.model.block_size * 2 + 1,
    )
    pair_parent = smoke_dir / "level_00" / "pair_invalid"
    for seed in run_seeds:
        torch.manual_seed(seed)
        init = GPT(ModelConfig(**{**asdict(cfg.model), "vocab_size": data.vocab_size})).state_dict()
        for opt in ["adamw", "adamw_wwpgd_reference"]:
            run_single(
                pair_parent,
                opt,
                seed,
                cfg,
                data.train,
                data.val,
                f"pair_invalid_seed_{seed}",
                steps,
                init,
            )
    return smoke_dir


def select_device(override: str | None = None):
    from wwgpt.device import resolve_device
    return resolve_device(override or "auto")


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    return sha256_bytes(b"".join(state[k].detach().cpu().numpy().tobytes() for k in sorted(state)))



def _optimizer_step_resolution(cfg: ExperimentConfig, token_multiplier: int, parameter_count_used: int) -> dict[str, int | str | None]:
    tokens_per_step = cfg.train.batch_size * cfg.model.block_size * cfg.train.gradient_accumulation
    if cfg.train.max_train_tokens is not None:
        budget_target_tokens = int(cfg.train.max_train_tokens)
        budget_derived_steps = max(1, math.ceil(budget_target_tokens / tokens_per_step))
        budget_source = "max_train_tokens"
    else:
        budget_target_tokens = int(parameter_count_used * token_multiplier)
        budget_derived_steps = max(1, math.ceil(budget_target_tokens / tokens_per_step))
        budget_source = "token_multiplier"
    resolved_steps = resolve_optimizer_steps(budget_derived_steps, cfg.train.max_steps)
    resolved_tokens = resolved_steps * tokens_per_step
    return {
        "budget_derived_optimizer_steps": budget_derived_steps,
        "configured_max_steps": cfg.train.max_steps,
        "resolved_optimizer_steps": resolved_steps,
        "tokens_per_optimizer_step": tokens_per_step,
        "resolved_train_tokens": resolved_tokens,
        "optimizer_step_limit_source": "configured_max_steps" if cfg.train.max_steps is not None and resolved_steps < budget_derived_steps else budget_source,
        "budget_source": budget_source,
        "requested_tokens": budget_target_tokens,
    }

def _compatibility(cfg: ExperimentConfig, data, init_hash: str, validation_probe_hash: str, training_probe_hash: str) -> dict[str, object]:
    cfgd = json.loads(json.dumps(asdict(cfg)))
    return {
        "configuration_hash": stable_hash(cfgd),
        "model_configuration_hash": stable_hash(cfgd.get("model", {})),
        "training_configuration_hash": stable_hash(cfgd.get("train", {})),
        "wwpgd_configuration_hash": stable_hash(cfgd.get("wwpgd", {})),
        "data_hash": data.corpus_hash,
        "tokenizer_hash": data.tokenizer_manifest.get("tokenizer_hash") or data.tokenizer_manifest.get("hash"),
        "initialization_hash": init_hash,
        "validation_probe_hash": validation_probe_hash,
        "training_probe_hash": training_probe_hash,
        "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
    }


def run_scientific_single(
    run_parent: Path,
    optimizer_name: str,
    seed: int,
    cfg: ExperimentConfig,
    data,
    pair_id: str,
    init_state: dict[str, torch.Tensor],
    init_hash: str,
    level: int,
    token_multiplier: int,
    device: str | None = None,
    ww_interval: int | None = None,
    eval_interval: int | None = None,
    checkpoint_interval: int | None = None,
    spectral_interval: int | None = None,
    precision: str | None = None,
    resume: bool = False,
    immediate_projection_spectral: bool = False,
    allow_code_version_mismatch: bool = False,
    audit_override_code_version_mismatch: bool = False,
) -> Path:
    if optimizer_name in {"adamw_wwpgd_reference", "adamw_wwpgd"}:
        base_optimizer, extension_name = "adamw", "wwpgd"
    elif optimizer_name in {"muon_wwpgd", "stableadamw_wwpgd", "stable_adamw_wwpgd"}:
        base_optimizer, extension_name = optimizer_name.removesuffix("_wwpgd"), "wwpgd"
    elif optimizer_name in {"adamw", "muon", "stableadamw", "stable_adamw"}:
        base_optimizer, extension_name = optimizer_name, getattr(cfg.wwpgd, "extension", "none")
    else:
        base_optimizer, extension_name = optimizer_name, getattr(cfg.wwpgd, "extension", "none")
    optimizer_name = make_arm_name(base_optimizer, extension_name)
    run_dir = None
    ckpt = None
    resolved_seeds = resolved_stochastic_seeds(seed, level, token_multiplier, split="train", optimizer_identity=base_optimizer)
    selected_device = select_device(device)
    selected_device_summary = device_summary(device or "auto")
    _log_train_progress(f"device selection: {selected_device_summary['selection_reason']}; single_device_only={selected_device_summary['single_device_only']}")
    precision_info = precision_policy(selected_device, precision)
    model = GPT(cfg.model).to(selected_device)
    model.load_state_dict(init_state)
    torch.manual_seed(resolved_seeds["dropout_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seeds["dropout_seed"])
    bundle, resolved_llrd_gamma = build_optimizer_bundle(model, cfg.train, base_optimizer)
    report = model.parameter_report()
    from wwgpt.scaling import selected_parameter_count
    parameter_count_used = selected_parameter_count(report, cfg.parameter_count_convention)
    resolution = _optimizer_step_resolution(cfg, token_multiplier, parameter_count_used)
    tokens_per_step = int(resolution["tokens_per_optimizer_step"])
    budget_derived_steps = int(resolution["budget_derived_optimizer_steps"])
    budget_source = str(resolution["budget_source"])
    budget_target_tokens = int(resolution["requested_tokens"])
    steps = int(resolution["resolved_optimizer_steps"])
    target_tokens = int(resolution["resolved_train_tokens"])
    realized_tokens = target_tokens
    optimizer_step_limit_source = str(resolution["optimizer_step_limit_source"])
    resolved_lr_decay_steps = resolve_lr_decay_steps(steps, cfg.train.lr_decay_steps)
    resolved_warmup_steps = resolve_warmup_steps(steps, cfg.train.warmup_ratio, cfg.train.warmup_steps, cfg.train.lr_decay_steps)
    wwpgd_interval = _resolve_wwpgd_interval(cfg, extension_name, ww_interval)
    expected_projection_optimizer_steps = _expected_projection_optimizer_steps(steps, wwpgd_interval, extension_name)
    cached_mode = extension_name == "wwpgd" and cfg.wwpgd.adaptive.apply_mode == "cached_endpoint_relaxation"
    endpoint_measurement_interval = (cfg.measurement.alpha_interval
                                     if cached_mode else None)
    expected_endpoint_measurement_steps = (list(range(endpoint_measurement_interval, steps + 1, endpoint_measurement_interval))
                                           if cached_mode else [])
    if cached_mode and cfg.wwpgd.adaptive.refresh_at_final_step and steps not in expected_endpoint_measurement_steps:
        expected_endpoint_measurement_steps.append(steps)
    expected_fast_apply_steps = ([step for step in range(cfg.wwpgd.adaptive.apply_interval, steps + 1,
                                                         cfg.wwpgd.adaptive.apply_interval)
                                  if not (cfg.wwpgd.adaptive.skip_fast_apply_on_measurement_step
                                          and step in expected_endpoint_measurement_steps)] if cached_mode else [])
    control_seed = stable_seed("wwgpt_control_seed_v1", seed, level, token_multiplier, base_optimizer, extension_name)
    if extension_name == "measurement_only":
        extension = MeasurementOnlyExtension(wwpgd_interval)
    elif extension_name in INTERVENTION_EXTENSIONS:
        extension = WWPGDExtension(cfg.wwpgd, wwpgd_interval, control=extension_name, scientific_seed=control_seed)
    else:
        extension = NoExtension()
    reader = (RandomWindowTokenReader(data.train, cfg.model.block_size, resolved_seeds["train_reader_seed"]) if cfg.train.training_sampling == "random_window" else NonRepeatingTokenReader(data.train, cfg.model.block_size))
    initial_minibatch_indices = _initial_minibatch_indices(data.train, cfg.model.block_size, cfg.train.batch_size, cfg.train.training_sampling, resolved_seeds["train_reader_seed"])
    validation_probe_hash = ""
    training_probe_hash = ""
    if data.data_manifest and data.data_manifest.get("storage_format") not in (None, "raw_memmap_v1"):
        raise RuntimeError("obsolete prepared-data format: rebuild with `wwgpt prepare-data` to create memmap token files")
    man = {
        "smoke_test": False,
        "valid_for_science": True,
        "level": level,
        "token_multiplier": token_multiplier,
        "seed": seed,
        "pair_id": pair_id,
        "optimizer": optimizer_name,
        "base_optimizer": base_optimizer,
        "extension": extension_name,
        "control_arm": extension_name if extension_name in CONTROL_EXTENSIONS else "",
        "control_arm_scientific_seed": control_seed if extension_name == "norm_matched_sham" else None,
        "arm_name": optimizer_name,
        "arm_display_name": ARM_DISPLAY[optimizer_name],
        "requested_tokens": budget_target_tokens,
        "target_train_tokens": target_tokens,
        "realized_tokens": realized_tokens,
        "realized_train_tokens": realized_tokens,
        "budget_derived_optimizer_steps": budget_derived_steps,
        "configured_max_steps": cfg.train.max_steps,
        "resolved_optimizer_steps": steps,
        "optimizer_steps": steps,
        "total_optimizer_steps": steps,
        "tokens_per_optimizer_step": tokens_per_step,
        "resolved_train_tokens": realized_tokens,
        "optimizer_step_limit_source": optimizer_step_limit_source,
        "budget_source": budget_source,
        "parameter_count_convention": cfg.parameter_count_convention,
        "parameter_count_used": parameter_count_used,
        "selected_parameter_count": parameter_count_used,
        "realized_tokens_per_selected_parameter": realized_tokens / max(parameter_count_used, 1),
        "sequence_count": realized_tokens // cfg.model.block_size,
        "dataset_name": data.data_manifest["dataset_name"],
        "dataset_config": data.data_manifest["dataset_config"],
        "dataset_revision": data.data_manifest["dataset_revision"],
        "tokenizer_hash": data.tokenizer_manifest["tokenizer_hash"],
        "data_hash": data.corpus_hash,
        "corpus_hash": data.corpus_hash,
        "initialization_hash": init_hash,
        "parameter_report": model.report_dict(),
        "model_config": asdict(cfg.model),
        "model_architecture_version": cfg.model.model_architecture_version,
        "model_config_hash": sha256_bytes(json.dumps(asdict(cfg.model), sort_keys=True).encode()),
        "optimizer_hyperparameters": asdict(cfg.train),
        "optimizer_fingerprint": json.loads(json.dumps(optimizer_fingerprint(bundle), default=str)),
        "extension_hyperparameters": asdict(cfg.wwpgd),
        "resolved_baseline_hyperparameters": resolved_baseline_hyperparameters(
            cfg,
            resolved_warmup_steps=resolved_warmup_steps,
            resolved_lr_decay_steps=resolved_lr_decay_steps,
            resolved_llrd_gamma=resolved_llrd_gamma,
        ),
        "training_schedule_hash": sha256_bytes(json.dumps({"seed": seed, "level": level, "token_multiplier": token_multiplier, "steps": steps, "batch": cfg.train.batch_size, "training_sampling": cfg.train.training_sampling}, sort_keys=True).encode()),
        "resolved_stochastic_seeds": resolved_seeds,
        "initial_minibatch_indices": initial_minibatch_indices,
        "training_sampling": cfg.train.training_sampling,
        "evaluation_sampling": cfg.train.evaluation_sampling,
        "evaluation_schedule_version": "random_per_eval_v1",
        "lr_schedule": cfg.train.lr_schedule,
        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,
        "layer_lr": cfg.train.layer_lr,
        "warmup_steps_requested": cfg.train.warmup_steps,
        "warmup_ratio": cfg.train.warmup_ratio,
        "resolved_warmup_steps": resolved_warmup_steps,
        "lr_decay_steps_requested": cfg.train.lr_decay_steps,
        "resolved_lr_decay_steps": resolved_lr_decay_steps,
        "min_lr_ratio": cfg.train.min_lr_ratio,
        "resolved_llrd_gamma": resolved_llrd_gamma,
        "llrd_min_multiplier": cfg.train.llrd_min_multiplier,
        "weight_decay": cfg.train.weight_decay,
        "grad_clip": cfg.train.grad_clip,
        "batch_size": cfg.train.batch_size,
        "gradient_accumulation": cfg.train.gradient_accumulation,
        "wwpgd_interval": wwpgd_interval,
        "projection_schedule_type": "optimizer_step_interval",
        "expected_projection_optimizer_steps": expected_projection_optimizer_steps,
        "wwpgd_adaptive_enabled": extension_name in INTERVENTION_EXTENSIONS and cfg.wwpgd.adaptive.mode != "uniform",
        "wwpgd_adaptive_mode": cfg.wwpgd.adaptive.mode,
        "wwpgd_adaptive_config": asdict(cfg.wwpgd.adaptive),
        "wwpgd_controller_version": CONTROLLER_VERSION,
        "wwpgd_override_precedence": PRECEDENCE,
        "wwpgd_maximum_blend_eta": cfg.wwpgd.blend_eta,
        "wwpgd_maximum_cayley_eta": cfg.wwpgd.cayley_eta,
        "wwpgd_trust_region": cfg.wwpgd.adaptive.max_relative_frobenius_change,
        "total_projection_events": len(expected_projection_optimizer_steps),
        "optimizer_step_count": 0,
        "device": selected_device_summary,
        "device_support": {"single_device_only": True, "distributed_training": False, "multi_gpu_or_tpu": "not claimed; no executable distributed smoke path is implemented"},
        "precision_policy": {k: v for k, v in precision_info.items() if k != "torch_dtype"},
        "wwpgd_call_count": 0,
        "projected_matrix_count": 0,
        "WeightWatcher version": "",
        "spectral estimator": "weightwatcher",
        "composite specification version": "raw_and_composite_v1",
        "estimated_flops": 6
        * GPT(cfg.model).parameter_report().total_parameters
        * int(data.data_manifest["realized_tokens"]),
        "spectral_estimator": "weightwatcher",
        "spectral_estimator_version": "",
        "wwpgd_implementation": "ww_pgd" if extension_name in INTERVENTION_EXTENSIONS else "none",
        "wwpgd_commit": WWPGD_COMMIT if extension_name in INTERVENTION_EXTENSIONS else "",
        "validation_probe_hash": validation_probe_hash,
        "training_probe_hash": training_probe_hash,
        "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "checkpoint_schema_version": 2,
        "immediate_projection_spectral": immediate_projection_spectral,
        "immediate_spectral_source": "weightwatcher_measured" if immediate_projection_spectral else "disabled",
        "weightwatcher_version": _ww_version(),
        "weightwatcher_configuration": {"detX": True, "randomize": False, "plot": False},
        "weightwatcher_diagnostic_configuration": {"detX": True, "randomize": True, "plot": False},
        "weightwatcher_diagnostic_outputs": {"per_layer_long_form": "spectral.csv", "run_level_aggregates": "weightwatcher_aggregates.csv"},
    }
    code_version = _repository_version()
    code_version.update({
        "weightwatcher_version": _ww_version(),
        "wwpgd_commit": WWPGD_COMMIT,
        "torch_version": torch.__version__,
        "optimizer_implementation_version": bundle.implementation_versions,
    })
    man.update(code_version)
    man.update(external_wwpgd_manifest_fields(extension_name in INTERVENTION_EXTENSIONS, cfg.wwpgd if extension_name in INTERVENTION_EXTENSIONS else None))
    if cached_mode:
        adaptive = cfg.wwpgd.adaptive
        man.update({
            "endpoint_measurement_source": adaptive.measurement_source,
            "endpoint_measurement_interval": endpoint_measurement_interval,
            "endpoint_apply_interval": adaptive.apply_interval,
            "expected_endpoint_measurement_steps": expected_endpoint_measurement_steps,
            "expected_fast_apply_steps": expected_fast_apply_steps,
            "expected_measurement_count": len(expected_endpoint_measurement_steps),
            "start_step": adaptive.start_step,
            "max_per_step_gain": adaptive.max_per_step_gain,
            "max_relative_frobenius_change_per_step": adaptive.max_relative_frobenius_change_per_step,
            "endpoint_stop_relative_distance": adaptive.endpoint_stop_relative_distance,
            "max_endpoint_age_steps": adaptive.max_endpoint_age_steps,
            "stale_distance_multiplier": adaptive.stale_distance_multiplier,
            "controller_version": CONTROLLER_VERSION,
            "adapter_mode": "cached_endpoint_relaxation_v1",
            "projection_schedule_type": "cached_endpoint_measurement_and_fast_apply",
        })
    cfgd_for_hash = json.loads(json.dumps(asdict(cfg)))
    man.update({
        "configuration_hash": stable_hash(cfgd_for_hash),
        "model_configuration_hash": stable_hash(cfgd_for_hash.get("model", {})),
        "training_configuration_hash": stable_hash(cfgd_for_hash.get("train", {})),
        "wwpgd_configuration_hash": stable_hash(cfgd_for_hash.get("wwpgd", {})),
    })
    expected_identity = {
        "pair_id": pair_id,
        "arm_name": optimizer_name,
        "seed": seed,
        "configuration_hash": man["configuration_hash"],
        "data_hash": data.corpus_hash,
        "tokenizer_hash": data.tokenizer_manifest["tokenizer_hash"],
        "initialization_hash": init_hash,
        "optimizer_fingerprint": man["optimizer_fingerprint"],
        "immediate_projection_spectral": immediate_projection_spectral,
        "resolved_optimizer_steps": steps,
    }
    compatibility = _compatibility(cfg, data, init_hash, validation_probe_hash, training_probe_hash)
    compatibility.update({"optimizer_name": optimizer_name, "optimizer_class": type(bundle.optimizers[0]).__name__, "immediate_projection_spectral": immediate_projection_spectral, "weightwatcher_configuration": {"detX": True, "randomize": False, "plot": False}, "seed": seed, "level": level, "token_multiplier": token_multiplier, "requested_tokens": budget_target_tokens, "realized_tokens": realized_tokens, "resolved_optimizer_steps": steps, "optimizer_step_limit_source": optimizer_step_limit_source, "optimizer_fingerprint": man["optimizer_fingerprint"], **code_version})
    resume_mismatches = {}
    if resume:
        action, selected, resume_mismatches = _select_resume_run(run_parent / optimizer_name, expected_identity, compatibility, allow_code_version_mismatch=allow_code_version_mismatch)
        if action == "complete":
            _log_train_progress(f"verified and skipped completed run pair={pair_id} optimizer={optimizer_name} seed={seed} output={selected}")
            return selected
        if action == "resume":
            run_dir = selected
            ckpt = run_dir / "checkpoints"
        else:
            run_dir = unique_dir(run_parent / optimizer_name, "run")
            ckpt = run_dir / "checkpoints"; ckpt.mkdir(parents=True, exist_ok=True)
            resume = False
    else:
        run_dir = unique_dir(run_parent / optimizer_name, "run")
        ckpt = run_dir / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
    cfgd = json.loads(json.dumps(asdict(cfg)))
    if not resume:
        write_json(run_dir / "environment.json", environment())
        (run_dir / "initialization_hash.txt").write_text(init_hash)
        write_json(run_dir / "manifest.json", man)
        write_json(run_dir / "data_manifest.json", data.data_manifest)
        write_json(run_dir / "tokenizer_manifest.json", data.tokenizer_manifest)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfgd))
        write_json(run_dir / "config.json", cfgd)
    metric_rows = []
    spectral_rows = []
    spectral_aggregate_rows = []
    composite_rows = []
    proj_rows = []
    controller_rows = []
    immediate_spectral_rows = []
    lr_rows = []
    alpha_rows = []
    best_validation_loss = float("inf")
    best_validation_step = 0
    latest_validation_loss = float("nan")
    completed_projection_event_indexes = []
    next_projection_event_index = 0
    elapsed_prior = 0.0
    optimizer_step_count = 0
    wwpgd_call_count = 0
    projected_matrix_count = 0
    _log_train_progress(
        f"starting run level={level} token_multiplier={token_multiplier} pair={pair_id} optimizer={optimizer_name} seed={seed} steps={steps} device={selected_device} output={run_dir}"
    )
    start_step = 1
    if resume:
        loaded = load_latest_checkpoint(run_dir)
        resume_mismatches = assert_checkpoint_compatible(loaded, compatibility, allow_code_version_mismatch=allow_code_version_mismatch)
        if resume_mismatches:
            audit = {"mismatches": resume_mismatches, "allowed": True, "audit_override": audit_override_code_version_mismatch, "publication_eligible": bool(audit_override_code_version_mismatch)}
            write_json(run_dir / f"code_version_mismatch_{int(time.time_ns())}.json", audit)
        model.load_state_dict(loaded["model_state_dict"])
        bundle.load_state_dict(loaded.get("optimizer_state_dict", loaded.get("base_optimizer_state_dict")))
        if "training_reader_state" in loaded and hasattr(reader, "load_state_dict"):
            reader.load_state_dict(loaded["training_reader_state"])
        else:
            reader.pos = int(loaded["training_reader_position"])
        metric_rows = list(loaded.get("metrics_rows", []))
        spectral_rows = list(loaded.get("periodic_weightwatcher_rows", []))
        alpha_rows = list(loaded.get("alpha_measurement_rows", []))
        spectral_aggregate_rows = list(loaded.get("periodic_weightwatcher_aggregate_rows", []))
        proj_rows = list(loaded.get("wwpgd_projection_rows", []))
        controller_rows = list(loaded.get("wwpgd_controller_rows", []))
        if hasattr(extension, "load_state_dict"):
            extension.load_state_dict(loaded.get("wwpgd_adaptive_controller_state", loaded.get("wwpgd_state", {})))
        immediate_spectral_rows = list(loaded.get("immediate_projection_weightwatcher_rows", []))
        lr_rows = list(loaded.get("lr_rows", []))
        composite_rows = list(loaded.get("composite_spectral_rows", []))
        best_validation_loss = float(loaded.get("best_validation_loss", best_validation_loss))
        best_validation_step = int(loaded.get("best_validation_step", best_validation_step))
        latest_validation_loss = float(loaded.get("latest_validation_loss", latest_validation_loss))
        completed_projection_event_indexes = list(loaded.get("completed_projection_event_indexes", []))
        next_projection_event_index = int(loaded.get("next_projection_event_index", len(completed_projection_event_indexes)))
        elapsed_prior = float(loaded.get("elapsed_training_time", 0.0))
        optimizer_step_count = int(loaded.get("optimizer_step_count", len(metric_rows)))
        wwpgd_call_count = int(loaded.get("wwpgd_call_count", len(completed_projection_event_indexes)))
        projected_matrix_count = int(loaded.get("projected_matrix_count", len(proj_rows)))
        restore_rng_state(loaded)
        start_step = int(loaded.get("next_step", int(loaded.get("current_step", 0)) + 1))
        _log_train_progress(f"resuming run pair={pair_id} optimizer={optimizer_name} seed={seed} from step={start_step} checkpoint={run_dir}")
    start = time.perf_counter()
    last_loss = latest_validation_loss if math.isfinite(latest_validation_loss) else 0.0
    ww_over = 0.0
    if not resume:
        optimizer_step_count = start_step - 1
        wwpgd_call_count = len(completed_projection_event_indexes)
        projected_matrix_count = len(proj_rows)
    for step in range(start_step, steps + 1):
        train_loss_value = 0.0
        for _ in range(cfg.train.gradient_accumulation):
            xb, yb = reader.next_batch(cfg.train.batch_size)
            x = torch.tensor(xb, device=selected_device)
            y = torch.tensor(yb, device=selected_device)
            with autocast_context(selected_device, precision):
                _, loss = model(x, y)
            assert loss is not None
            (loss / cfg.train.gradient_accumulation).backward()
            train_loss_value += float(loss.detach().cpu())
        grad_before_clip = _gradient_norm(model.parameters())
        if cfg.train.grad_clip > 0.0:
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            grad_after_clip = _gradient_norm(model.parameters())
        else:
            grad = grad_before_clip
            grad_after_clip = grad_before_clip
        lr_update_rows = apply_lr_schedule(bundle, step - 1, steps, resolved_warmup_steps, cfg.train)
        lr_rows.extend(lr_update_rows)
        logged_lr = float(lr_update_rows[0]["current_lr"]) if lr_update_rows else cfg.train.learning_rate
        for _opt in bundle.optimizers:
            optimizer_step(_opt, selected_device)
        synchronize_device(selected_device)
        optimizer_step_count = step
        loss = torch.tensor(train_loss_value / cfg.train.gradient_accumulation)
        ps = time.perf_counter()
        alpha_due = step % cfg.measurement.alpha_interval == 0 or step == steps
        pre_details, new_proj, new_controller = extension.after_optimizer_step(model=model, optimizer_step=step, total_optimizer_steps=steps, tokens_seen=step * tokens_per_step, collect_pre_details=(immediate_projection_spectral or alpha_due), seed=seed, pair_id=pair_id, base_optimizer=base_optimizer, arm_name=optimizer_name, measurement_interval=cfg.measurement.alpha_interval)
        cached_mode = extension_name == "wwpgd" and cfg.wwpgd.adaptive.apply_mode == "cached_endpoint_relaxation"
        if extension_name in INTERVENTION_EXTENSIONS and new_proj and not cached_mode:
            wwpgd_call_count += 1
        projected_matrix_count += sum(bool(r.get("changed", r.get("projected", True))) for r in new_proj)
        proj_rows.extend(new_proj)
        controller_rows.extend(new_controller)
        if new_proj and not cached_mode:
            event_idx = int(new_proj[0].get("projection_event", next_projection_event_index))
            if event_idx not in completed_projection_event_indexes:
                completed_projection_event_indexes.append(event_idx)
            next_projection_event_index = max(next_projection_event_index, event_idx + 1)
            if immediate_projection_spectral:
                post = measured_projection_spectral_rows(pre_details, model, step=step, tokens_seen=step * tokens_per_step, optimizer=optimizer_name, seed=seed, pair_id=pair_id, projection_event=event_idx, projection_rows=new_proj, target_alpha=cfg.wwpgd.target_alpha, phase="post")
                immediate_spectral_rows.extend(post)
        proj_time = time.perf_counter() - ps if new_proj else 0.0
        bundle.zero_grad()
        measurement_interval = cfg.measurement.alpha_interval
        measurement_due = (extension_name == "wwpgd" and cfg.wwpgd.adaptive.apply_mode == "cached_endpoint_relaxation"
                           and (step % measurement_interval == 0 or (cfg.wwpgd.adaptive.refresh_at_final_step and step == steps)))
        if step % (eval_interval or cfg.train.eval_interval) == 0 or step == steps or measurement_due or alpha_due:
            eval_index = len(metric_rows)
            was_training = model.training
            model.eval()
            if cfg.train.evaluation_sampling == "fixed_probe":
                train_x, train_y, training_probe_hash = fixed_probe(data.train[cfg.train.batch_size * cfg.model.block_size * 2:], cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
                val_x, val_y, validation_probe_hash = fixed_probe(data.val, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
            else:
                train_x, train_y, training_probe_hash = random_probe(data.train, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches, stable_seed(resolved_seeds["train_eval_probe_seed_base"], eval_index, "random_per_eval_v1"))
                val_x, val_y, validation_probe_hash = random_probe(data.val, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches, stable_seed(resolved_seeds["val_eval_probe_seed_base"], eval_index, "random_per_eval_v1"))
            diagnostic_test_metrics = None
            diagnostic_test_loss = float("nan")
            diagnostic_test_probe_hash = ""
            with torch.no_grad():
                tm, _ = _evaluate_probe_batches(model, train_x, train_y, selected_device)
                vm, validation_probe_loss = _evaluate_probe_batches(model, val_x, val_y, selected_device)
                if cfg.train.test_evaluation_mode == "diagnostic_periodic" and data.test is not None:
                    test_x, test_y, diagnostic_test_probe_hash = random_probe(
                        data.test,
                        cfg.model.block_size,
                        cfg.train.batch_size,
                        cfg.train.eval_batches,
                        stable_seed(resolved_seeds["val_eval_probe_seed_base"], eval_index, "diagnostic_test_random_per_eval_v1"),
                    )
                    diagnostic_test_metrics, diagnostic_test_loss = _evaluate_probe_batches(model, test_x, test_y, selected_device)
            model.train(was_training)
            elapsed = elapsed_prior + time.perf_counter() - start
            last_loss = validation_probe_loss
            latest_validation_loss = validation_probe_loss
            if validation_probe_loss < best_validation_loss:
                best_validation_loss = validation_probe_loss
                best_validation_step = step
                torch.save(model.state_dict(), ckpt / f"best_val_step_{step:06d}_{seed}.pt")
            metric_rows.append(
                {
                    "step": step,
                    "tokens_processed": step * tokens_per_step,
                    "elapsed_time": elapsed,
                    "optimizer_steps": optimizer_step_count,
                    "wall_clock_time": elapsed,
                    "learning_rate": logged_lr,
                    "gradient_norm": float(grad_before_clip.detach().cpu()),
                    "gradient_norm_before_clip": float(grad_before_clip.detach().cpu()),
                    "gradient_norm_after_clip": float(grad_after_clip.detach().cpu()),
                    "train_minibatch_loss": float(loss.detach().cpu()),
                    "train_loss": tm["loss"],
                    "train_cross_entropy": tm["loss"],
                    "validation_loss": vm["loss"],
                    "validation_cross_entropy": vm["loss"],
                    "val_loss": vm["loss"],
                    "test_loss": diagnostic_test_loss,
                    "test_cross_entropy": diagnostic_test_loss,
                    "train_perplexity": tm["perplexity"],
                    "validation_perplexity": vm["perplexity"],
                    "val_perplexity": vm["perplexity"],
                    "test_perplexity": diagnostic_test_metrics["perplexity"] if diagnostic_test_metrics else float("nan"),
                    "train_bits_per_token": tm["bits_per_token"],
                    "val_bits_per_token": vm["bits_per_token"],
                    "train_top1_accuracy": tm["top1_accuracy"],
                    "val_top1_accuracy": vm["top1_accuracy"],
                    "train_top5_accuracy": tm["top5_accuracy"],
                    "val_top5_accuracy": vm["top5_accuracy"],
                    "train_token_error": tm["token_error"],
                    "val_token_error": vm["token_error"],
                    "train_validation_gap": vm["loss"] - tm["loss"],
                    "train_test_gap": diagnostic_test_loss - tm["loss"] if diagnostic_test_metrics else float("nan"),
                    "generalization_gap": vm["loss"] - tm["loss"],
                    "evaluation_index": eval_index,
                    "evaluation_sampling": cfg.train.evaluation_sampling,
                    "train_eval_batch_hash": training_probe_hash,
                    "val_eval_batch_hash": validation_probe_hash,
                    "test_eval_batch_hash": diagnostic_test_probe_hash,
                    "evaluation_token_count": int(
                        cfg.train.eval_batches * cfg.train.batch_size * cfg.model.block_size
                    ),
                    "validation_probe_hash": validation_probe_hash,
                    "training_probe_hash": training_probe_hash,
                    "evaluation_batches": cfg.train.eval_batches,
                    "validation_document_count": data.data_manifest.get(
                        "validation_document_count", 0
                    ),
                    "tokens_per_second": (step * tokens_per_step) / max(elapsed, 1e-9),
                    "examples_per_second": (step * cfg.train.batch_size) / max(elapsed, 1e-9),
                    "weightwatcher_overhead": ww_over,
                    "projection_overhead": proj_time,
                    "peak_memory": float(
                        memory_stats(selected_device).get("max_allocated", 0.0)
                    ),
                }
            )
            if measurement_due:
                measurement_result = extension.after_metrics_measurement(
                    model=model, optimizer_step=step, total_optimizer_steps=steps,
                    tokens_seen=step * tokens_per_step, force=True,
                    measurement_interval=measurement_interval)
                measured_details = measurement_result.pre_projection_details
                controller_rows.extend(measurement_result.controller_rows)
                if measurement_result.stock_wwpgd_invoked:
                    wwpgd_call_count += 1
            if alpha_due:
                # Cached WW-PGD details and event-projection stock details are
                # authoritative.  Only arms without such a result make a call.
                alpha_details = measured_details if measurement_due else pre_details
                failure = ""
                if alpha_details is None:
                    saved_rng = rng_state()
                    was_training_for_alpha = model.training
                    try:
                        model.eval()
                        alpha_details = nonmutating_weightwatcher_details(model, randomize=False)
                    except Exception as exc:
                        failure = f"weightwatcher_error:{type(exc).__name__}:{exc}"
                    finally:
                        model.train(was_training_for_alpha)
                        restore_rng_state(saved_rng)
                alpha_rows.extend(alpha_measurement_rows(alpha_details, model, step=step,
                    tokens_seen=step * tokens_per_step, seed=seed, pair_id=pair_id,
                    base_optimizer=base_optimizer, extension=extension_name, arm_name=optimizer_name,
                    failure_reason=failure))
            ws = time.perf_counter()
            if step % cfg.measurement.trap_diagnostic_interval == 0 or step == steps:
                new_spectral_rows = spectral_summary(
                    model,
                    step=step,
                    tokens_seen=step * tokens_per_step,
                    optimizer=optimizer_name,
                    seed=seed,
                    pair_id=pair_id,
                    randomize=cfg.measurement.trap_randomize,
                )
                spectral_rows.extend(new_spectral_rows)
                spectral_aggregate_rows.extend(weightwatcher_run_aggregates(new_spectral_rows))
            ww_over += time.perf_counter() - ws
            _log_train_progress(
                f"progress pair={pair_id} optimizer={optimizer_name} seed={seed} step={step}/{steps} tokens={step * tokens_per_step}/{int(data.data_manifest['realized_tokens'])} train_loss={tm['loss']:.4f} val_loss={vm['loss']:.4f} elapsed_s={elapsed:.1f} tokens_per_s={(step * tokens_per_step) / max(elapsed, 1e-9):.1f}"
            )
        if cfg.composite_spectral_analysis_enabled and (step % (spectral_interval or cfg.train.spectral_interval) == 0 or step == steps):
            composite_rows.extend(composite_spectral_summary(model, step=step, tokens_seen=step * tokens_per_step, base_optimizer=base_optimizer, extension=extension_name, arm_name=optimizer_name, seed=seed, pair_id=pair_id))
        if step % (checkpoint_interval or cfg.train.checkpoint_interval) == 0:
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": bundle.state_dict(),
                "base_optimizer_state_dict": bundle.state_dict(),
                "scheduler_state_dict": None,
                "gradient_scaler_state_dict": None,
                "current_step": step,
                "next_step": step + 1,
                "optimizer_step_count": optimizer_step_count,
                "wwpgd_call_count": wwpgd_call_count,
                "projected_matrix_count": projected_matrix_count,
                "wwpgd_state": {"extension": extension_name, "call_count": wwpgd_call_count, "projected_matrix_count": projected_matrix_count, "completed_projection_event_indexes": list(completed_projection_event_indexes), "next_projection_event_index": next_projection_event_index, "wwpgd_interval": wwpgd_interval},
                "tokens_processed": step * tokens_per_step,
                "training_reader_position": reader.pos,
                "reader_position": reader.pos,
                "training_reader_state": reader.state_dict() if hasattr(reader, "state_dict") else {"pos": reader.pos},
                "seed": seed,
                **rng_state(),
                "device_type": selected_device.type,
                "precision_policy": precision or "torch_default",
                "gradient_accumulation_position": 0,
                "best_validation_loss": best_validation_loss,
                "best_validation_step": best_validation_step,
                "latest_validation_loss": latest_validation_loss,
                "completed_projection_event_indexes": completed_projection_event_indexes,
                "next_projection_event_index": next_projection_event_index,
                        "metrics_rows": metric_rows,
                "periodic_weightwatcher_rows": spectral_rows,
                "alpha_measurement_rows": alpha_rows,
                "periodic_weightwatcher_aggregate_rows": spectral_aggregate_rows,
                "wwpgd_projection_rows": proj_rows,
                "wwpgd_controller_rows": controller_rows,
                "wwpgd_adaptive_controller_state": extension.state_dict() if hasattr(extension, "state_dict") else {},
                "immediate_projection_weightwatcher_rows": immediate_spectral_rows,
                "lr_rows": lr_rows,
                "composite_spectral_rows": composite_rows,
                "elapsed_training_time": elapsed_prior + time.perf_counter() - start,
                "initialization_hash": init_hash,
                "resolved_stochastic_seeds": resolved_seeds,
                "compatibility": compatibility,
                "resolved_config": cfgd,
                "optimizer_fingerprint": man["optimizer_fingerprint"],
                "data_hash": data.corpus_hash,
                "tokenizer_hash": data.tokenizer_manifest["tokenizer_hash"],
                "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
                "lr_schedule": cfg.train.lr_schedule, "scheduler_implementation": SCHEDULER_IMPLEMENTATION, "layer_lr": cfg.train.layer_lr, "warmup_steps_requested": cfg.train.warmup_steps, "warmup_ratio": cfg.train.warmup_ratio, "resolved_warmup_steps": resolved_warmup_steps, "lr_decay_steps_requested": cfg.train.lr_decay_steps, "resolved_lr_decay_steps": resolved_lr_decay_steps, "min_lr_ratio": cfg.train.min_lr_ratio,
                "weightwatcher_version": _ww_version(), "weightwatcher_configuration": {"detX": True, "randomize": False, "plot": False}, "wwpgd_commit": WWPGD_COMMIT if extension_name in INTERVENTION_EXTENSIONS else "", "git_commit": man.get("git_commit", "unknown"), "optimizer_name": optimizer_name, "pair_id": pair_id, "level": level, "token_multiplier": token_multiplier, "realized_tokens": realized_tokens, "requested_tokens": target_tokens, "immediate_projection_spectral": immediate_projection_spectral, "run_directory": str(run_dir),
            }
            save_checkpoint(run_dir, state)
            _log_train_progress(
                f"checkpoint saved pair={pair_id} optimizer={optimizer_name} seed={seed} step={step}/{steps} dir={ckpt}"
            )
    final_elapsed = elapsed_prior + time.perf_counter() - start
    save_checkpoint(run_dir, {"model_state_dict": model.state_dict(), "optimizer_state_dict": bundle.state_dict(), "base_optimizer_state_dict": bundle.state_dict(), "scheduler_state_dict": None, "gradient_scaler_state_dict": None, "current_step": steps, "next_step": steps + 1, "optimizer_step_count": optimizer_step_count, "wwpgd_call_count": wwpgd_call_count, "projected_matrix_count": projected_matrix_count, "wwpgd_state": {"extension": extension_name, "call_count": wwpgd_call_count, "projected_matrix_count": projected_matrix_count, "completed_projection_event_indexes": list(completed_projection_event_indexes), "next_projection_event_index": next_projection_event_index, "wwpgd_interval": wwpgd_interval}, "tokens_processed": steps * tokens_per_step, "training_reader_position": reader.pos, "reader_position": reader.pos, "training_reader_state": reader.state_dict() if hasattr(reader, "state_dict") else {"pos": reader.pos}, "seed": seed, **rng_state(), "device_type": selected_device.type, "precision_policy": precision or "torch_default", "gradient_accumulation_position": 0, "best_validation_loss": best_validation_loss, "best_validation_step": best_validation_step, "latest_validation_loss": latest_validation_loss, "completed_projection_event_indexes": completed_projection_event_indexes, "next_projection_event_index": next_projection_event_index, "metrics_rows": metric_rows, "periodic_weightwatcher_rows": spectral_rows, "periodic_weightwatcher_aggregate_rows": spectral_aggregate_rows, "wwpgd_projection_rows": proj_rows, "wwpgd_controller_rows": controller_rows, "wwpgd_adaptive_controller_state": extension.state_dict() if hasattr(extension, "state_dict") else {}, "immediate_projection_weightwatcher_rows": immediate_spectral_rows, "lr_rows": lr_rows, "composite_spectral_rows": composite_rows, "elapsed_training_time": final_elapsed, "initialization_hash": init_hash, "resolved_stochastic_seeds": resolved_seeds, "compatibility": compatibility, "resolved_config": cfgd, "optimizer_fingerprint": man["optimizer_fingerprint"], "data_hash": data.corpus_hash, "tokenizer_hash": data.tokenizer_manifest["tokenizer_hash"], "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION, "lr_schedule": cfg.train.lr_schedule, "scheduler_implementation": SCHEDULER_IMPLEMENTATION, "layer_lr": cfg.train.layer_lr, "warmup_steps_requested": cfg.train.warmup_steps, "warmup_ratio": cfg.train.warmup_ratio, "resolved_warmup_steps": resolved_warmup_steps, "lr_decay_steps_requested": cfg.train.lr_decay_steps, "resolved_lr_decay_steps": resolved_lr_decay_steps, "min_lr_ratio": cfg.train.min_lr_ratio, "weightwatcher_version": _ww_version(), "weightwatcher_configuration": {"detX": True, "randomize": False, "plot": False}, "wwpgd_commit": WWPGD_COMMIT if extension_name in INTERVENTION_EXTENSIONS else "", "git_commit": man.get("git_commit", "unknown"), "optimizer_name": optimizer_name, "pair_id": pair_id, "level": level, "token_multiplier": token_multiplier, "realized_tokens": realized_tokens, "requested_tokens": budget_target_tokens, "budget_derived_optimizer_steps": budget_derived_steps, "configured_max_steps": cfg.train.max_steps, "resolved_optimizer_steps": steps, "tokens_per_optimizer_step": tokens_per_step, "resolved_train_tokens": realized_tokens, "optimizer_step_limit_source": optimizer_step_limit_source, "immediate_projection_spectral": immediate_projection_spectral, "run_directory": str(run_dir)})
    final_path = ckpt / f"final_step_{steps:06d}_{seed}.pt"
    torch.save(model.state_dict(), final_path)

    # Checkpoint comparisons never mutate metrics.csv; it is the live-model time series.
    if metric_rows:
        final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        train_x, train_y, final_train_hash = fixed_probe(data.train[cfg.train.batch_size * cfg.model.block_size * 2:], cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
        val_x, val_y, final_val_hash = fixed_probe(data.val, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            final_train_metrics, _ = _evaluate_probe_batches(model, train_x, train_y, selected_device)
            final_val_metrics, _ = _evaluate_probe_batches(model, val_x, val_y, selected_device)
        final_record = _checkpoint_metric_row(
            checkpoint_path=final_path, checkpoint_hash=_file_sha256(final_path), step=steps,
            selection_metric="final_training_step", train_metrics=final_train_metrics,
            validation_metrics=final_val_metrics, test_metrics=None,
            probe_hashes={"train_probe_hash": final_train_hash, "validation_probe_hash": final_val_hash, "test_probe_hash": ""})
        final_record["final_checkpoint_path"] = final_record["checkpoint_path"]
        final_record["final_checkpoint_hash"] = final_record["checkpoint_hash"]
        final_record["training_probe_hash"] = final_record["train_probe_hash"]
        write_json(run_dir / "final_checkpoint_metrics.json", final_record)

        # Selection depends solely on validation loss. All three selected metrics are
        # then recomputed together after loading that exact immutable artifact.
        selected_step = best_validation_step or steps
        selected_path = ckpt / f"best_val_step_{selected_step:06d}_{seed}.pt"
        if not selected_path.exists():
            selected_path = final_path
        model.load_state_dict(torch.load(selected_path, map_location=selected_device, weights_only=False))
        selected_train_x, selected_train_y, train_hash = fixed_probe(data.train[cfg.train.batch_size * cfg.model.block_size * 2:], cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
        selected_val_x, selected_val_y, val_hash = fixed_probe(data.val, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
        test_hash = ""
        selected_test_metrics = None
        with torch.no_grad():
            selected_train_metrics, _ = _evaluate_probe_batches(model, selected_train_x, selected_train_y, selected_device)
            selected_val_metrics, _ = _evaluate_probe_batches(model, selected_val_x, selected_val_y, selected_device)
            if data.test is not None:
                test_x, test_y, test_hash = fixed_probe(data.test, cfg.model.block_size, cfg.train.batch_size, cfg.train.eval_batches)
                selected_test_metrics, _ = _evaluate_probe_batches(model, test_x, test_y, selected_device)
        selected_record = _checkpoint_metric_row(
            checkpoint_path=selected_path, checkpoint_hash=_file_sha256(selected_path), step=selected_step,
            selection_metric="validation_loss", train_metrics=selected_train_metrics,
            validation_metrics=selected_val_metrics, test_metrics=selected_test_metrics,
            probe_hashes={"train_probe_hash": train_hash, "validation_probe_hash": val_hash, "test_probe_hash": test_hash})
        selected_record["selection_metric_value"] = best_validation_loss
        selected_record["training_probe_hash"] = selected_record["train_probe_hash"]
        write_json(run_dir / "selected_checkpoint_metrics.json", selected_record)
        _write_csv(run_dir / "selected_checkpoint_metrics.csv", [selected_record], overwrite=True)
        model.load_state_dict(final_state)
        model.train(was_training)
    _append_only_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "spectral.csv", spectral_rows, overwrite=resume)
    _write_csv(run_dir / "alpha_measurements.csv", alpha_rows, overwrite=resume)
    _write_csv(run_dir / "weightwatcher_aggregates.csv", spectral_aggregate_rows, overwrite=resume)
    if cfg.composite_spectral_analysis_enabled:
        _write_csv(run_dir / "composite_spectral.csv", composite_rows, overwrite=resume)
    _write_csv(run_dir / "lrs.csv", lr_rows, overwrite=resume)
    if extension_name in INTERVENTION_EXTENSIONS:
        write_csv_union_schema(run_dir / "wwpgd_projection.csv", proj_rows, empty_fields=[
            "optimizer_step", "layer_name", "changed", "requested_relative_frobenius_change",
            "applied_relative_frobenius_change", "controller_gain_requested", "controller_gain_applied",
            "action_type", "controller_version", "adapter_mode",
        ])
        write_csv_union_schema(run_dir / "wwpgd_controller.csv", controller_rows,
                               empty_fields=["optimizer_step", "layer_name", "action_type", "skip_reason", "changed"])
        write_csv_union_schema(run_dir / "wwpgd_endpoint_measurements.csv", getattr(extension, "measurement_rows", []),
                               empty_fields=["optimizer_step", "measurement_index", "layer_name", "cache_activated", "skip_reason", "action_type"])
        write_csv_union_schema(run_dir / "wwpgd_endpoint_relaxation.csv", getattr(extension, "relaxation_rows", []),
                               empty_fields=["optimizer_step", "layer_name", "changed", "converged", "invalidated", "invalidation_reason", "action_type"])
        if immediate_projection_spectral:
            _write_csv(run_dir / "wwpgd_projection_spectral.csv", immediate_spectral_rows, overwrite=resume)
    (run_dir / "events.jsonl").write_text(json.dumps({"event": "complete"}) + "\n")
    skip_counts={}
    for r in controller_rows:
        reason=str(r.get("skip_reason", ""))
        if reason: skip_counts[reason]=skip_counts.get(reason,0)+1
    common_complete={"step": steps, "final_val_loss": last_loss, "optimizer_step_count": optimizer_step_count,
                     "wwpgd_call_count": wwpgd_call_count, "projected_matrix_count": projected_matrix_count,
                     "budget_derived_optimizer_steps": budget_derived_steps, "configured_max_steps": cfg.train.max_steps,
                     "resolved_optimizer_steps": steps, "tokens_per_optimizer_step": tokens_per_step,
                     "resolved_train_tokens": realized_tokens, "optimizer_step_limit_source": optimizer_step_limit_source}
    if cached_mode:
        measurements = getattr(extension, "measurement_rows", [])
        relaxations = getattr(extension, "relaxation_rows", [])
        counters = getattr(extension, "counters", {})
        changed = [r for r in relaxations if r.get("changed")]
        requested_gains = [float(r["controller_gain_requested"]) for r in relaxations if r.get("controller_gain_requested") is not None]
        applied_gains = [float(r["controller_gain_applied"]) for r in relaxations if r.get("controller_gain_applied") is not None]
        applied_changes = [float(r["applied_relative_frobenius_change"]) for r in changed if r.get("applied_relative_frobenius_change") is not None]
        activations = [r for r in measurements if r.get("cache_activated")]
        common_complete.update({
            "completed_measurement_count": int(counters.get("measurement_count", 0)),
            "stock_wwpgd_invocation_count": int(counters.get("candidate_generation_count", 0)),
            "endpoint_activation_count": len(activations),
            "fast_control_step_count": int(counters.get("fast_control_step_count", 0)),
            "fast_changed_step_count": int(counters.get("changed_fast_control_step_count", 0)),
            "fast_layer_decision_count": len(relaxations), "fast_changed_layer_count": len(changed),
            "endpoint_convergence_count": int(counters.get("endpoint_convergence_count", 0)),
            "endpoint_invalidation_count": int(counters.get("endpoint_invalidation_count", 0)),
            "active_endpoint_count_at_finish": sum(ep.active for ep in getattr(extension, "endpoint_cache", {}).values()),
            "above_target_measurement_count": sum(r.get("alpha_side") == "above_target" for r in measurements),
            "below_target_measurement_count": sum(r.get("alpha_side") == "below_target" for r in measurements),
            "above_target_activation_count": sum(r.get("alpha_side") == "above_target" and r.get("cache_activated") for r in measurements),
            "below_target_activation_count": sum(r.get("alpha_side") == "below_target" and r.get("cache_activated") for r in measurements),
            "mean_controller_gain_requested": sum(requested_gains)/len(requested_gains) if requested_gains else 0.0,
            "max_controller_gain_requested": max(requested_gains, default=0.0),
            "mean_controller_gain_applied": sum(applied_gains)/len(applied_gains) if applied_gains else 0.0,
            "max_controller_gain_applied": max(applied_gains, default=0.0),
            "mean_applied_relative_frobenius_change": sum(applied_changes)/len(applied_changes) if applied_changes else 0.0,
            "max_applied_relative_frobenius_change": max(applied_changes, default=0.0),
            "expected_measurement_count": len(expected_endpoint_measurement_steps),
            "expected_endpoint_measurement_steps": expected_endpoint_measurement_steps,
            "expected_fast_apply_steps": expected_fast_apply_steps,
            "projection_schedule_type": "cached_endpoint_measurement_and_fast_apply",
        })
    else:
        applied=[float(r.get("combined_hardness_applied", 0.0) or 0.0) for r in controller_rows]
        rels=[float(r.get("relative_frobenius_change_applied", 0.0) or 0.0) for r in controller_rows if math.isfinite(float(r.get("relative_frobenius_change_applied", 0.0) or 0.0))]
        common_complete.update({"completed_projection_event_indexes": completed_projection_event_indexes,
            "next_projection_event_index": next_projection_event_index, "wwpgd_interval": wwpgd_interval,
            "expected_projection_optimizer_steps": expected_projection_optimizer_steps,
            "total_projection_events": len(expected_projection_optimizer_steps), "projection_schedule_type": "optimizer_step_interval",
            "total_layer_decisions": len(controller_rows), "total_projected_layers": sum(bool(r.get("projected")) for r in controller_rows),
            "total_skipped_layers": sum(not bool(r.get("projected")) for r in controller_rows), "controller_skip_counts": skip_counts,
            "mean_applied_hardness": sum(applied)/len(applied) if applied else 0.0, "max_applied_hardness": max(applied, default=0.0),
            "mean_relative_frobenius_change": sum(rels)/len(rels) if rels else 0.0, "maximum_relative_frobenius_change": max(rels, default=0.0)})
    write_json(run_dir / "run_complete.json", common_complete)
    _log_train_progress(
        f"completed run pair={pair_id} optimizer={optimizer_name} seed={seed} steps={steps} final_val_loss={last_loss:.4f} output={run_dir}"
    )
    return run_dir


CANONICAL_TRIAL_ARMS = ("adamw", "adamw_wwpgd", "muon", "muon_wwpgd", "stable_adamw", "stable_adamw_wwpgd")
CANONICAL_TRIAL_PAIRS = {"adamw": "adamw_wwpgd", "muon": "muon_wwpgd", "stable_adamw": "stable_adamw_wwpgd"}
CANONICAL_TRIAL_BASES = ("adamw", "muon", "stable_adamw")

def _trial_manifest(pair_id: str, level: int, token_multiplier: int, seed: int, cfg: ExperimentConfig, data, init_hash: str, analysis_plan_path: Path | None = None) -> dict:
    cfgd = json.loads(json.dumps(asdict(cfg)))
    report = GPT(cfg.model).parameter_report()
    from wwgpt.scaling import selected_parameter_count
    resolution = _optimizer_step_resolution(cfg, token_multiplier, selected_parameter_count(report, cfg.parameter_count_convention))
    shared = {
        "trial_id": pair_id, "seed": seed, "level": level, "token_multiplier": token_multiplier,
        "model_config": asdict(cfg.model), "model_configuration_hash": stable_hash(cfgd.get("model", {})),
        "data_manifest": data.data_manifest, "data_hash": data.corpus_hash,
        "tokenizer_manifest": data.tokenizer_manifest, "tokenizer_hash": data.tokenizer_manifest.get("tokenizer_hash"),
        "initialization_hash": init_hash, "train": asdict(cfg.train), "token_budget": {"realized_tokens": data.data_manifest.get("realized_tokens"), "token_multiplier": token_multiplier, **resolution},
    }
    arms = []
    for base in CANONICAL_TRIAL_BASES:
        for ext in ("none", "wwpgd"):
            arm = make_arm_name(base, ext)
            arms.append({"arm_name": arm, "base_optimizer": base, "extension": ext, "paired_with": CANONICAL_TRIAL_PAIRS.get(base) if ext == "none" else base, "learning_rate": cfg.train.learning_rate, "lr_schedule": cfg.train.lr_schedule, "scheduler_implementation": SCHEDULER_IMPLEMENTATION, "weight_decay": cfg.train.weight_decay, "initialization_hash": init_hash, "batch_order_seed": resolved_stochastic_seeds(seed, level, token_multiplier, optimizer_identity=base)["train_reader_seed"], "token_budget": shared["token_budget"]})
    manifest = {"scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION, "immutable": True, "trial_id": pair_id, "shared": shared, "arms": arms, "pairs": [{"baseline": b, "wwpgd": w} for b, w in CANONICAL_TRIAL_PAIRS.items()]}
    if analysis_plan_path is not None:
        from wwgpt.acceleration_analysis import plan_manifest
        manifest.update(plan_manifest(analysis_plan_path))
    return manifest

def run_canonical_trials(level: int, data_root: Path, results_root: Path, token_multiplier: int, seeds: list[int] | None = None, config_path: Path | None = None, device: str | None = None, ww_interval: int | None = None, eval_interval: int | None = None, checkpoint_interval: int | None = None, spectral_interval: int | None = None, precision: str | None = None, resume: bool = False, immediate_projection_spectral: bool = False, allow_code_version_mismatch: bool = False, analysis_plan_path: Path | None = None, audit_override_code_version_mismatch: bool = False) -> Path:
    from wwgpt.data import load_prepared_scientific_data
    cfg = load_config(config_path, level)
    data = load_prepared_scientific_data(data_root, level, token_multiplier)
    exp_root = results_root / "experiments" / f"level_{level:02d}" / f"multiplier_{token_multiplier}"
    exp_root.mkdir(parents=True, exist_ok=True)
    for seed in (seeds or cfg.seeds):
        existing_trials = sorted(exp_root.glob(f"trial_{seed}*")) if resume else []
        trial = existing_trials[0] if existing_trials else unique_dir(exp_root, f"trial_{seed}")
        trial_id = trial.name
        init_dir = trial / "initial_state"; init_dir.mkdir(parents=True, exist_ok=True)
        if resume and (init_dir / "model.pt").exists():
            init_state = torch.load(init_dir / "model.pt", map_location="cpu", weights_only=False); init_hash = (init_dir / "initialization_hash.txt").read_text().strip()
        else:
            torch.manual_seed(resolved_stochastic_seeds(seed, level, token_multiplier)["model_init_seed"]); init_model = GPT(cfg.model); init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}; init_hash = _state_hash(init_state); torch.save(init_state, init_dir / "model.pt"); (init_dir / "initialization_hash.txt").write_text(init_hash)
        if not (resume and (trial / "trial_manifest.json").exists()): write_json(trial / "trial_manifest.json", _trial_manifest(trial_id, level, token_multiplier, seed, cfg, data, init_hash, analysis_plan_path))
        for base in CANONICAL_TRIAL_BASES:
            for ext in ("none", "wwpgd"):
                arm_cfg = replace(cfg, wwpgd=replace(cfg.wwpgd, extension=ext, enabled=(ext == "wwpgd")))
                run_scientific_single(trial, make_arm_name(base, ext), seed, arm_cfg, data, trial_id, init_state, init_hash, level, token_multiplier, device, ww_interval, eval_interval, checkpoint_interval, spectral_interval, precision, resume, immediate_projection_spectral, allow_code_version_mismatch, audit_override_code_version_mismatch)
    return exp_root

def run_multiseed_scientific(
    level: int,
    data_root: Path,
    results_root: Path,
    token_multiplier: int,
    seeds: list[int] | None = None,
    config_path: Path | None = None,
    device: str | None = None,
    ww_interval: int | None = None,
    eval_interval: int | None = None,
    checkpoint_interval: int | None = None,
    spectral_interval: int | None = None,
    precision: str | None = None,
    resume: bool = False,
    optimizer: str = "adamw",
    extensions: list[str] | None = None,
    immediate_projection_spectral: bool = False,
    allow_code_version_mismatch: bool = False,
    audit_override_code_version_mismatch: bool = False,
) -> Path:
    from wwgpt.data import load_prepared_scientific_data

    cfg = load_config(config_path, level)
    data = load_prepared_scientific_data(data_root, level, token_multiplier)
    exp_root = (
        results_root / "experiments" / f"level_{level:02d}" / f"multiplier_{token_multiplier}"
    )
    exp_root.mkdir(parents=True, exist_ok=True)
    run_seeds = seeds or cfg.seeds
    _log_train_progress(
        f"starting multiseed level={level} token_multiplier={token_multiplier} seeds={','.join(str(s) for s in run_seeds)} results={exp_root}"
    )
    for seed_index, seed in enumerate(run_seeds, start=1):
        if resume:
            existing_pairs = sorted(
                [p for p in exp_root.glob(f"pair_{seed}*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            pair = existing_pairs[0] if existing_pairs else unique_dir(exp_root, f"pair_{seed}")
        else:
            pair = unique_dir(exp_root, f"pair_{seed}")
        pair_id = pair.name
        init_dir = pair / "initial_state"
        if resume and (init_dir / "model.pt").exists():
            init_state = torch.load(init_dir / "model.pt", map_location="cpu", weights_only=False)
            init_hash = (init_dir / "initialization_hash.txt").read_text().strip()
        else:
            init_seed = resolved_stochastic_seeds(seed, level, token_multiplier)["model_init_seed"]
            torch.manual_seed(init_seed)
            init_model = GPT(cfg.model)
            init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
            init_hash = _state_hash(init_state)
        _log_train_progress(
            f"starting seed {seed_index}/{len(run_seeds)} seed={seed} pair={pair_id}"
        )
        init_dir.mkdir(exist_ok=True)
        if not (init_dir / "model.pt").exists():
            torch.save(init_state, init_dir / "model.pt")
        if not (init_dir / "initialization_hash.txt").exists():
            (init_dir / "initialization_hash.txt").write_text(init_hash)
        if not (resume and (pair / "pair_manifest.json").exists()):
            write_json(
                pair / "pair_manifest.json",
                {
                "pair_id": pair_id,
                "seed": seed,
                "level": level,
                "token_multiplier": token_multiplier,
                "initialization_hash": init_hash,
                "resolved_stochastic_seeds": resolved_stochastic_seeds(seed, level, token_multiplier),
                "base_optimizer": optimizer,
                "extensions": extensions or ["none", "wwpgd"],
                "arms": [make_arm_name(optimizer, e) for e in (extensions or ["none", "wwpgd"])],
                },
            )
        for ext in (extensions or ["none", "wwpgd"]):
            arm_cfg = replace(cfg, wwpgd=replace(cfg.wwpgd, extension=ext, enabled=(ext in INTERVENTION_EXTENSIONS)))
            run_scientific_single(
                pair,
                optimizer,
                seed,
                arm_cfg,
                data,
                pair_id,
                init_state,
                init_hash,
                level,
                token_multiplier,
                device,
                ww_interval,
                eval_interval,
                checkpoint_interval,
                spectral_interval,
                precision,
                resume,
                immediate_projection_spectral,
                allow_code_version_mismatch,
                audit_override_code_version_mismatch,
            )
        _log_train_progress(
            f"completed seed {seed_index}/{len(run_seeds)} seed={seed} pair={pair_id}"
        )
    _log_train_progress(
        f"completed multiseed level={level} token_multiplier={token_multiplier} output={exp_root}"
    )
    return exp_root
