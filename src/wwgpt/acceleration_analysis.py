"""Paired, seed-level learning-curve acceleration analysis.

Evaluation events are repeated measurements, not experimental units.  This
module consequently emits one outcome per seed/base-optimizer pair and never
uses evaluation rows as independent samples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_analysis_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load a plan and return it with a stable hash of its exact frozen bytes."""
    path = Path(path)
    raw = path.read_bytes()
    plan = yaml.safe_load(raw) or {}
    if not isinstance(plan, dict):
        raise ValueError("analysis plan must be a YAML mapping")
    mode = plan.get("mode")
    if mode not in {"exploratory", "confirmatory"}:
        raise ValueError("analysis plan mode must be exploratory or confirmatory")
    if mode == "confirmatory" and (not plan.get("thresholds") or not plan.get("primary_outcomes")):
        raise ValueError("confirmatory plans require non-empty thresholds and primary_outcomes")
    return plan, hashlib.sha256(raw).hexdigest()


def plan_manifest(path: str | Path) -> dict[str, Any]:
    plan, digest = load_analysis_plan(path)
    return {"analysis_plan_path": str(Path(path)), "analysis_plan_sha256": digest,
            "analysis_mode": plan["mode"], "analysis_plan": plan}


def _curve(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {"tokens_processed": "tokens_seen", "val_loss": "validation_loss",
               "elapsed_time": "elapsed_seconds"}
    d = frame.rename(columns={k: v for k, v in aliases.items() if k in frame}).copy()
    required = {"tokens_seen", "validation_loss"}
    if not required <= set(d):
        raise ValueError(f"curve is missing columns: {sorted(required - set(d))}")
    for col in ("tokens_seen", "validation_loss", "step", "elapsed_seconds"):
        if col in d: d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["tokens_seen", "validation_loss"]).sort_values("tokens_seen")
    return d.drop_duplicates("tokens_seen", keep="last").reset_index(drop=True)


def threshold_crossing(curve: pd.DataFrame, threshold: float, *, sustained: bool = True) -> dict[str, Any]:
    """Find an interpolated crossing, requiring two consecutive low events if sustained."""
    d = _curve(curve); y = d.validation_loss.to_numpy(); x = d.tokens_seen.to_numpy()
    limit = len(d) - 1 if sustained else len(d)
    for i in range(limit):
        if y[i] <= threshold and (not sustained or y[i + 1] <= threshold):
            if i and y[i - 1] > threshold and y[i] != y[i - 1]:
                frac = (y[i - 1] - threshold) / (y[i - 1] - y[i])
                token = float(x[i - 1] + frac * (x[i] - x[i - 1]))
            else:
                token = float(x[i])
            result = {"reached": True, "tokens": token, "event_index": i,
                      "event_tokens": float(x[i]), "status": "reached"}
            for source, name in (("step", "optimizer_steps"), ("elapsed_seconds", "wall_clock_seconds")):
                if source in d:
                    result[name] = float(np.interp(token, x, d[source].to_numpy()))
            return result
    return {"reached": False, "tokens": None, "event_index": None,
            "event_tokens": None, "status": "not reached"}


def sustained_tokens_to_threshold(curve: pd.DataFrame, threshold: float) -> float | None:
    return threshold_crossing(curve, threshold, sustained=True)["tokens"]


def paired_auc(baseline: pd.DataFrame, wwpgd: pd.DataFrame) -> dict[str, Any]:
    a, b = _curve(baseline), _curve(wwpgd)
    lo = max(float(a.tokens_seen.min()), float(b.tokens_seen.min()))
    hi = min(float(a.tokens_seen.max()), float(b.tokens_seen.max()))
    if hi <= lo:
        return {"common_token_start": lo, "common_token_end": hi, "baseline_auc": None,
                "wwpgd_auc": None, "paired_auc_difference": None, "status": "no common interval"}
    grid = np.unique(np.r_[lo, hi, a.tokens_seen[(a.tokens_seen >= lo) & (a.tokens_seen <= hi)],
                           b.tokens_seen[(b.tokens_seen >= lo) & (b.tokens_seen <= hi)]])
    ay = np.interp(grid, a.tokens_seen, a.validation_loss)
    by = np.interp(grid, b.tokens_seen, b.validation_loss)
    aa, ba = float(np.trapezoid(ay, grid)), float(np.trapezoid(by, grid))
    return {"common_token_start": lo, "common_token_end": hi, "baseline_auc": aa,
            "wwpgd_auc": ba, "paired_auc_difference": aa - ba, "status": "observed common support"}


def _at_budget(curve: pd.DataFrame, budget: float) -> float | None:
    d = _curve(curve)
    if budget < d.tokens_seen.min() or budget > d.tokens_seen.max(): return None
    return float(np.interp(budget, d.tokens_seen, d.validation_loss))


def _exploratory_thresholds(pairs: list[dict[str, Any]]) -> list[float]:
    lows, highs = [], []
    for p in pairs:
        for arm in (p["baseline"], p["wwpgd"]):
            d = _curve(arm); lows.append(d.validation_loss.min()); highs.append(d.validation_loss.max())
    lo, hi = max(lows), min(highs)
    return [float(x) for x in np.linspace(lo, hi, 5)[1:-1]] if hi > lo else [float(lo)]


def analyze_acceleration_pairs(pairs: list[dict[str, Any]], output_dir: str | Path,
                               plan_path: str | Path) -> Path:
    """Analyze dictionaries containing seed, base_optimizer, baseline, and wwpgd curves."""
    plan, digest = load_analysis_plan(plan_path); mode = plan["mode"]
    thresholds = [float(x) for x in plan.get("thresholds", [])]
    if mode == "exploratory" and not thresholds: thresholds = _exploratory_thresholds(pairs)
    budgets = [float(x) for x in plan.get("fixed_token_budgets", [])]
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    accel, aucs, audits = [], [], []
    for p in pairs:
        seed, base = p.get("seed"), p.get("base_optimizer")
        a, w = _curve(p["baseline"]), _curve(p["wwpgd"])
        aucs.append({"seed": seed, "base_optimizer": base, "analysis_mode": mode,
                     "exploratory": mode == "exploratory", **paired_auc(a, w)})
        for threshold in thresholds:
            ac, wc = threshold_crossing(a, threshold), threshold_crossing(w, threshold)
            au, wu = threshold_crossing(a, threshold, sustained=False), threshold_crossing(w, threshold, sustained=False)
            saved = ac["tokens"] - wc["tokens"] if ac["reached"] and wc["reached"] else None
            ratio = ac["tokens"] / wc["tokens"] if saved is not None and wc["tokens"] else None
            row = {"seed": seed, "base_optimizer": base, "analysis_mode": mode,
                   "exploratory": mode == "exploratory", "threshold": threshold,
                   "baseline_sustained_tokens": ac["tokens"], "wwpgd_sustained_tokens": wc["tokens"],
                   "baseline_status": ac["status"], "wwpgd_status": wc["status"],
                   "tokens_saved": saved, "speedup_ratio": ratio,
                   "baseline_minimum_validation_loss": float(a.validation_loss.min()),
                   "wwpgd_minimum_validation_loss": float(w.validation_loss.min()),
                   "baseline_final_validation_loss": float(a.validation_loss.iloc[-1]),
                   "wwpgd_final_validation_loss": float(w.validation_loss.iloc[-1]),
                   "baseline_optimizer_steps_to_threshold": ac.get("optimizer_steps"),
                   "wwpgd_optimizer_steps_to_threshold": wc.get("optimizer_steps"),
                   "baseline_wall_clock_seconds_to_threshold": ac.get("wall_clock_seconds"),
                   "wwpgd_wall_clock_seconds_to_threshold": wc.get("wall_clock_seconds")}
            for budget in budgets:
                key = str(int(budget) if budget.is_integer() else budget)
                row[f"baseline_loss_at_{key}_tokens"] = _at_budget(a, budget)
                row[f"wwpgd_loss_at_{key}_tokens"] = _at_budget(w, budget)
            accel.append(row)
            for arm, sustained, first in (("baseline", ac, au), ("wwpgd", wc, wu)):
                audits.append({"seed": seed, "base_optimizer": base, "arm": arm, "threshold": threshold,
                               "analysis_mode": mode, "exploratory": mode == "exploratory",
                               "first_crossing_tokens": first["tokens"],
                               "first_crossing_status": first["status"], "sustained_crossing_tokens": sustained["tokens"],
                               "sustained_crossing_status": sustained["status"], "observed_token_min": float((a if arm == "baseline" else w).tokens_seen.min()),
                               "observed_token_max": float((a if arm == "baseline" else w).tokens_seen.max())})
    adf, udf = pd.DataFrame(accel), pd.DataFrame(aucs)
    adf.to_csv(out / "acceleration_by_seed.csv", index=False); udf.to_csv(out / "validation_auc_by_seed.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "threshold_crossing_audit.csv", index=False)
    group = ["base_optimizer", "threshold", "analysis_mode"]
    summary = adf.groupby(group, dropna=False).agg(seed_count=("seed", "size"), reached_pairs=("tokens_saved", "count"),
        mean_tokens_saved=("tokens_saved", "mean"), median_tokens_saved=("tokens_saved", "median"),
        mean_speedup_ratio=("speedup_ratio", "mean")).reset_index() if not adf.empty else pd.DataFrame()
    summary.to_csv(out / "acceleration_summary.csv", index=False)
    _plots(pairs, adf, out)
    (out / "analysis_plan_manifest.json").write_text(json.dumps({"analysis_plan_sha256": digest,
        "analysis_plan_path": str(Path(plan_path)), "analysis_mode": mode, "exploratory": mode == "exploratory",
        "thresholds_used": thresholds, "fixed_token_budgets": budgets, "plan": plan}, indent=2, sort_keys=True))
    return out


def _plots(pairs: list[dict[str, Any]], results: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in pairs:
        for arm, style in (("baseline", "--"), ("wwpgd", "-")):
            d = _curve(p[arm]); ax.plot(d.tokens_seen, d.validation_loss, style, alpha=.65,
                label=f"{p.get('base_optimizer')} {arm} seed {p.get('seed')}")
    ax.set(xlabel="Tokens", ylabel="Validation loss");
    if pairs: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "paired_learning_curves.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not results.empty:
        labels = [f"{b}/s{s}/L{t:g}" for b, s, t in zip(results.base_optimizer, results.seed, results.threshold)]
        ax.bar(labels, results.tokens_saved.fillna(0)); ax.tick_params(axis="x", rotation=70)
    ax.axhline(0, color="black", linewidth=.8); ax.set(ylabel="Tokens saved (positive = WW-PGD faster)")
    fig.tight_layout(); fig.savefig(out / "tokens_saved_by_seed.png", dpi=160); plt.close(fig)


def analyze_acceleration_results(results_root: str | Path, output_dir: str | Path,
                                 plan_path: str | Path) -> Path:
    from wwgpt.analysis import discover_canonical_runs, load_csv_file
    runs = discover_canonical_runs(Path(results_root), include_legacy=True); grouped = {}
    for r in runs:
        grouped.setdefault((r.get("seed"), r.get("base_optimizer")), {})[r.get("extension")] = r
    pairs = []
    for (seed, base), arms in grouped.items():
        if "none" in arms and "wwpgd" in arms:
            pairs.append({"seed": seed, "base_optimizer": base,
                "baseline": load_csv_file(Path(arms["none"]["run_dir"]) / "metrics.csv"),
                "wwpgd": load_csv_file(Path(arms["wwpgd"]["run_dir"]) / "metrics.csv")})
    return analyze_acceleration_pairs(pairs, output_dir, plan_path)
