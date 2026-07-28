"""Paired, seed-level learning-curve acceleration analysis.

Evaluation events are repeated measurements, not experimental units.  This
module consequently emits one outcome per seed/base-optimizer pair and never
uses evaluation rows as independent samples.
"""
from __future__ import annotations

import hashlib
import itertools
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
    planned_seeds = plan.get("confirmatory_paired_seeds")
    if mode == "confirmatory" and planned_seeds is not None and int(planned_seeds) < 10 \
            and not plan.get("paired_seed_count_justification"):
        raise ValueError("confirmatory plans require at least 10 paired seeds or an explicit justification")
    return plan, hashlib.sha256(raw).hexdigest()


def plan_manifest(path: str | Path) -> dict[str, Any]:
    plan, digest = load_analysis_plan(path)
    return {"analysis_plan_path": str(Path(path)), "analysis_plan_sha256": digest,
            "analysis_mode": plan["mode"],
            "confirmatory_paired_seeds": plan.get("confirmatory_paired_seeds"),
            "required_paired_seed_count": plan.get("confirmatory_paired_seeds"),
            "thresholds": plan.get("thresholds", []),
            "primary_outcomes": plan.get("primary_outcomes", []),
            "analysis_thresholds": plan.get("thresholds", []),
            "analysis_primary_outcomes": plan.get("primary_outcomes", []),
            "analysis_plan": plan}


def verify_analysis_eligibility(runs: list[dict[str, Any]], output_dir: str | Path,
                                plan_path: str | Path) -> dict[str, Any]:
    """Bind analysis to the frozen training plan and count complete seed pairs."""
    plan, digest = load_analysis_plan(plan_path)
    grouped: dict[tuple[str, Any], dict[str, dict[str, Any]]] = {}
    recorded: list[str | None] = []
    trial_recorded: list[str | None] = []
    for run in runs:
        manifest = run.get("manifest") or {}
        value = manifest.get("analysis_plan_sha256") or manifest.get("analysis_plan_hash")
        is_complete = bool(run.get("run_dir") and (Path(run["run_dir"]) / "run_complete.json").exists())
        if is_complete:
            recorded.append(str(value) if value else None)
            trial = manifest.get("trial_manifest")
            if isinstance(trial, dict):
                trial_hash = trial.get("analysis_plan_sha256") or trial.get("analysis_plan_hash")
                trial_recorded.append(str(trial_hash) if trial_hash else None)
        valid_for_science = run.get("valid_for_science", manifest.get("valid_for_science", True)) is True
        if is_complete and valid_for_science:
            grouped.setdefault((str(run.get("base_optimizer")), run.get("seed")), {})[str(run.get("extension"))] = run
    observed: dict[str, int] = {base: 0 for base, _seed in grouped}
    identity_fields = ("initialization_hash", "model_configuration_hash", "tokenizer_hash",
                       "data_hash", "validation_probe_hash", "training_probe_hash",
                       "base_optimizer_fingerprint", "token_multiplier", "level", "target_alpha")
    invalid_pairs: list[str] = []
    for (base, seed), arms in grouped.items():
        if {"none", "wwpgd"} <= set(arms):
            left, right = arms["none"].get("manifest", {}), arms["wwpgd"].get("manifest", {})
            mismatches = [key for key in identity_fields
                          if left.get(key) is not None and right.get(key) is not None
                          and left.get(key) != right.get(key)]
            if mismatches:
                invalid_pairs.append(f"{base}/seed={seed} identity mismatch: {', '.join(mismatches)}")
                continue
            if any((arm.get("manifest") or {}).get("analysis_plan_sha256") != digest for arm in arms.values()):
                invalid_pairs.append(f"{base}/seed={seed} does not carry the supplied frozen plan hash")
                continue
            observed[base] = observed.get(base, 0) + 1
    required = int(plan.get("confirmatory_paired_seeds") or 0)
    reasons: list[str] = []
    hash_status = "not_required"
    if plan["mode"] == "confirmatory":
        if not recorded or any(value is None for value in recorded):
            hash_status = "missing"
            reasons.append("every completed analyzed arm must record the confirmatory analysis plan hash")
        elif set(recorded) != {digest}:
            hash_status = "mismatch"
            reasons.append(f"supplied plan SHA-256 {digest} does not match recorded training hash(es): {sorted(set(recorded))}")
        else:
            hash_status = "match"
        if trial_recorded and (any(value is None for value in trial_recorded) or set(trial_recorded) != {digest}):
            reasons.append("canonical trial manifest hash does not match every arm and the supplied plan")
        reasons.extend(invalid_pairs)
        if not observed:
            reasons.append("no complete baseline/WW-PGD seed pairs were observed")
        for base, count in sorted(observed.items()):
            if count < required:
                reasons.append(f"{base} has {count} complete paired seeds; confirmatory plan requires {required}")
    elif not observed:
        reasons.append("exploratory analysis requires at least one complete baseline/WW-PGD seed pair")
    artifact = {"analysis_mode": plan["mode"], "required_paired_seeds": required,
                "observed_paired_seeds_by_optimizer": observed, "plan_hash_status": hash_status,
                "supplied_plan_sha256": digest, "recorded_plan_sha256": sorted({x for x in recorded if x}),
                "eligible": not reasons, "exclusion_reasons": reasons}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "analysis_eligibility.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    if reasons:
        raise RuntimeError("analysis is ineligible: " + "; ".join(reasons))
    return artifact


def _column_series(frame: pd.DataFrame, name: str) -> pd.Series | None:
    """Return one coalesced Series even when a label occurs more than once."""
    positions = [
        index for index, column in enumerate(frame.columns) if column == name
    ]
    if not positions:
        return None
    selected = frame.iloc[:, positions]
    result = selected.iloc[:, 0]
    for index in range(1, selected.shape[1]):
        result = result.combine_first(selected.iloc[:, index])
    return result


def _coalesce_aliases(
    frame: pd.DataFrame, aliases: dict[str, tuple[str, ...]]
) -> pd.DataFrame:
    """Canonicalize aliases without creating duplicate column labels."""
    result = frame.copy()
    for canonical, fallback_names in aliases.items():
        merged: pd.Series | None = None
        present: list[str] = []
        for name in (canonical, *fallback_names):
            values = _column_series(result, name)
            if values is None:
                continue
            present.append(name)
            merged = values if merged is None else merged.combine_first(values)
        if merged is None:
            continue
        result = result.drop(columns=list(dict.fromkeys(present)))
        result[canonical] = merged
    return result


def _curve(frame: pd.DataFrame) -> pd.DataFrame:
    d = _coalesce_aliases(
        frame,
        {
            "tokens_seen": ("tokens_processed",),
            "validation_loss": ("val_loss",),
            "elapsed_seconds": ("elapsed_time",),
        },
    )
    required = {"tokens_seen", "validation_loss"}
    if not required <= set(d):
        raise ValueError(f"curve is missing columns: {sorted(required - set(d))}")
    for col in (
        "tokens_seen",
        "validation_loss",
        "step",
        "elapsed_seconds",
        "base_optimizer_seconds",
        "total_elapsed_seconds",
    ):
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["tokens_seen", "validation_loss"]).sort_values(
        "tokens_seen"
    )
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
            for source, name in (("step", "optimizer_steps"),
                                 ("base_optimizer_seconds", "base_optimization_seconds"),
                                 ("total_elapsed_seconds", "total_wall_clock_seconds"),
                                 ("elapsed_seconds", "wall_clock_seconds")):
                if source in d:
                    result[name] = float(np.interp(token, x, d[source].to_numpy()))
            return result
    return {"reached": False, "tokens": None, "event_index": None,
            "event_tokens": None, "status": "not reached"}


def sustained_tokens_to_threshold(curve: pd.DataFrame, threshold: float) -> float | None:
    return threshold_crossing(curve, threshold, sustained=True)["tokens"]

def wall_clock_acceleration(baseline: dict[str, Any], intervention: dict[str, Any]) -> bool:
    """Only total elapsed time, including analysis/intervention, supports this claim."""
    left = baseline.get("total_wall_clock_seconds")
    right = intervention.get("total_wall_clock_seconds")
    return left is not None and right is not None and float(right) < float(left)


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


def _percentile_bootstrap(values: np.ndarray, *, samples: int = 10_000) -> tuple[float, float]:
    """Deterministic seed-level percentile bootstrap interval for the mean."""
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(20260725)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _sign_flip_pvalue(values: np.ndarray) -> tuple[float, int, str]:
    """Return an exact, two-sided paired sign-flip p-value when enumeration is practical."""
    nonzero = values[np.isfinite(values) & (values != 0)]
    n = len(nonzero)
    if not n:
        return 1.0, 0, "applicable; no nonzero effects"
    if n > 20:
        return np.nan, n, "not applicable; more than 20 nonzero pairs"
    observed = abs(float(nonzero.mean()))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        if abs(float(np.mean(nonzero * np.asarray(signs)))) >= observed - 1e-15:
            extreme += 1
    return extreme / (2 ** n), n, "applicable; exact enumeration"


def _paired_inference(effects: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    keys = ["base_optimizer", "metric", "threshold", "token_budget"]
    for key, group in effects.groupby(keys, dropna=False, sort=True):
        values = pd.to_numeric(group["paired_effect"], errors="coerce").dropna().to_numpy(float)
        low, high = _percentile_bootstrap(values)
        pvalue, nonzero, applicability = _sign_flip_pvalue(values)
        n = len(values)
        seeds_observed = group["seed"].nunique(dropna=True)
        rows.append(dict(zip(keys, key)) | {
            "analysis_mode": mode, "n_seeds_observed": seeds_observed, "n_complete_pairs": n,
            "individual_paired_effects": json.dumps(values.tolist()),
            "mean": float(np.mean(values)) if n else np.nan,
            "median": float(np.median(values)) if n else np.nan,
            "sample_standard_deviation": float(np.std(values, ddof=1)) if n >= 2 else np.nan,
            "standard_error": float(np.std(values, ddof=1) / np.sqrt(n)) if n >= 2 else np.nan,
            "percentile_bootstrap_ci_low": low, "percentile_bootstrap_ci_high": high,
            "bootstrap_resamples": 10_000, "exact_sign_flip_p_value_two_sided": pvalue,
            "sign_flip_nonzero_pairs": nonzero, "sign_flip_test_status": applicability,
            "power_label": "pilot, limited paired power" if seeds_observed == 5 else "",
            "interpretation_warning": "An interval excluding zero is not, by itself, a causal claim."
        })
    return pd.DataFrame(rows)


def analyze_acceleration_pairs(pairs: list[dict[str, Any]], output_dir: str | Path,
                               plan_path: str | Path) -> Path:
    """Analyze dictionaries containing seed, base_optimizer, baseline, and wwpgd curves."""
    plan, digest = load_analysis_plan(plan_path); mode = plan["mode"]
    thresholds = [float(x) for x in plan.get("thresholds", [])]
    if mode == "exploratory" and not thresholds: thresholds = _exploratory_thresholds(pairs)
    budgets = [float(x) for x in plan.get("fixed_token_budgets", [])]
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    accel, aucs, audits, effects, missing = [], [], [], [], []
    for p in pairs:
        seed, base = p.get("seed"), p.get("base_optimizer")
        if p.get("baseline") is None or p.get("wwpgd") is None:
            pattern = "both_arms_missing" if p.get("baseline") is None and p.get("wwpgd") is None else (
                "baseline_arm_missing" if p.get("baseline") is None else "wwpgd_arm_missing")
            for threshold in thresholds:
                missing.append({"seed": seed, "base_optimizer": base, "threshold": threshold,
                                "baseline_status": "arm missing" if p.get("baseline") is None else "arm observed",
                                "wwpgd_status": "arm missing" if p.get("wwpgd") is None else "arm observed",
                                "missingness_pattern": pattern, "complete_pair": False})
            continue
        a, w = _curve(p["baseline"]), _curve(p["wwpgd"])
        auc = {"seed": seed, "base_optimizer": base, "analysis_mode": mode,
               "exploratory": mode == "exploratory", **paired_auc(a, w)}
        aucs.append(auc)
        if "paired_validation_loss_auc" in plan.get("primary_outcomes", []):
            effects.append({"seed": seed, "base_optimizer": base, "metric": "paired_validation_loss_auc",
                            "threshold": np.nan, "token_budget": np.nan,
                            "paired_effect": auc["paired_auc_difference"], "effect_definition": "baseline minus wwpgd"})
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
            pattern = ("both_reached" if ac["reached"] and wc["reached"] else
                       "baseline_only_reached" if ac["reached"] else
                       "wwpgd_only_reached" if wc["reached"] else "neither_reached")
            missing.append({"seed": seed, "base_optimizer": base, "threshold": threshold,
                            "baseline_status": ac["status"], "wwpgd_status": wc["status"],
                            "missingness_pattern": pattern, "complete_pair": bool(ac["reached"] and wc["reached"])})
            primary = set(plan.get("primary_outcomes", []))
            if "sustained_tokens_to_threshold" in primary:
                effects.append({"seed": seed, "base_optimizer": base, "metric": "sustained_tokens_to_threshold",
                                "threshold": threshold, "token_budget": np.nan, "paired_effect": saved,
                                "effect_definition": "baseline tokens minus wwpgd tokens"})
            if "tokens_saved" in primary:
                effects.append({"seed": seed, "base_optimizer": base, "metric": "tokens_saved",
                                "threshold": threshold, "token_budget": np.nan, "paired_effect": saved,
                                "effect_definition": "baseline tokens minus wwpgd tokens"})
            if "speedup_ratio" in primary:
                effects.append({"seed": seed, "base_optimizer": base, "metric": "speedup_ratio",
                                "threshold": threshold, "token_budget": np.nan,
                                "paired_effect": ratio - 1 if ratio is not None else None,
                                "effect_definition": "baseline/wwpgd ratio minus null value 1"})
            # Fixed-budget outcomes do not vary by threshold; emit exactly one
            # seed-level effect rather than pseudoreplicating it across thresholds.
            if "validation_loss_at_fixed_token_budgets" in primary and threshold == thresholds[0]:
                for budget in budgets:
                    av, wv = _at_budget(a, budget), _at_budget(w, budget)
                    effects.append({"seed": seed, "base_optimizer": base,
                                    "metric": "validation_loss_at_fixed_token_budget", "threshold": np.nan,
                                    "token_budget": budget,
                                    "paired_effect": av - wv if av is not None and wv is not None else None,
                                    "effect_definition": "baseline loss minus wwpgd loss"})
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
    effect_columns = ["seed", "base_optimizer", "metric", "threshold", "token_budget",
                      "paired_effect", "effect_definition"]
    edf = pd.DataFrame(effects, columns=effect_columns)
    edf.to_csv(out / "paired_effects_by_seed.csv", index=False)
    _paired_inference(edf, mode).to_csv(out / "paired_effect_estimates.csv", index=False)
    pd.DataFrame(missing, columns=["seed", "base_optimizer", "threshold", "baseline_status",
        "wwpgd_status", "missingness_pattern", "complete_pair"]).to_csv(out / "missing_pair_audit.csv", index=False)
    warning = {
        "warning_code": "FIVE_PAIR_EXACT_SIGN_FLIP_LIMIT",
        "applies_when_nonzero_complete_pairs": 5,
        "label": "pilot, limited paired power",
        "test": "exact two-sided paired sign-flip randomization test",
        "can_attain_p_below_0_05": False,
        "minimum_attainable_two_sided_p_value": 2 / 32,
        "calculation": "2/32 = 0.0625",
        "experimental_unit": "seed",
        "independence_warning": "Layers, evaluations, tokens, and thresholds are not independent replicates.",
        "causal_interpretation_warning": "A confidence interval excluding zero does not automatically establish causality."
    }
    (out / "statistical_power_warning.json").write_text(json.dumps(warning, indent=2, sort_keys=True) + "\n")
    _plots(pairs, adf, out)
    (out / "analysis_plan_manifest.json").write_text(json.dumps({"analysis_plan_sha256": digest,
        "analysis_plan_path": str(Path(plan_path)), "analysis_mode": mode, "exploratory": mode == "exploratory",
        "thresholds_used": thresholds, "fixed_token_budgets": budgets, "plan": plan}, indent=2, sort_keys=True))
    return out


def _plots(pairs: list[dict[str, Any]], results: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in pairs:
        for arm, style in (("baseline", "--"), ("wwpgd", "-")):
            if p.get(arm) is None:
                continue
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
        pairs.append({"seed": seed, "base_optimizer": base,
            "baseline": load_csv_file(Path(arms["none"]["run_dir"]) / "metrics.csv") if "none" in arms else None,
            "wwpgd": load_csv_file(Path(arms["wwpgd"]["run_dir"]) / "metrics.csv") if "wwpgd" in arms else None})
    analyze_acceleration_pairs(pairs, output_dir, plan_path)
    analyze_paired_alpha_validation(runs, output_dir)
    return Path(output_dir)


def _read_optional(run_dir: Path, names: tuple[str, ...]) -> pd.DataFrame:
    for name in names:
        path = run_dir / name
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _backward_validation_join(alpha: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest validation event no later than each alpha observation.

    This deliberately uses a backward as-of join: interpolation and nearest-event
    joins can leak a future validation outcome into an earlier alpha observation.
    """
    a = alpha.copy()
    a["optimizer_step"] = pd.to_numeric(a["optimizer_step"], errors="coerce")
    m = _coalesce_aliases(
        metrics,
        {
            "validation_step": ("step", "optimizer_step"),
            "validation_loss": ("val_loss",),
        },
    )
    if not {"validation_step", "validation_loss"} <= set(m):
        a["validation_step"] = np.nan; a["validation_loss"] = np.nan
        return a
    m["validation_step"] = pd.to_numeric(m["validation_step"], errors="coerce")
    m["validation_loss"] = pd.to_numeric(m["validation_loss"], errors="coerce")
    m = m.dropna(subset=["validation_step", "validation_loss"]).sort_values("validation_step")
    a = a.dropna(subset=["optimizer_step"]).sort_values("optimizer_step")
    return pd.merge_asof(a, m[["validation_step", "validation_loss"]],
                         left_on="optimizer_step", right_on="validation_step",
                         direction="backward", allow_exact_matches=True)


def _correlations(seed: pd.DataFrame) -> pd.DataFrame:
    metrics = ["tokens_saved", "median_alpha_error_reduction",
               "fraction_layers_entering_target_band", "cumulative_applied_wwpgd_displacement"]
    if seed.empty:
        seed = pd.DataFrame(columns=metrics)
    rows = []
    for i, left in enumerate(metrics):
        for right in metrics[i + 1:]:
            d = seed[[left, right]].dropna()
            rows.append({"metric_x": left, "metric_y": right, "seed_count": len(d),
                         "pearson_association": d[left].corr(d[right], method="pearson") if len(d) >= 2 else np.nan,
                         "spearman_association": d[left].corr(d[right], method="spearman") if len(d) >= 2 else np.nan,
                         "interpretation": "cross-seed association; not causal evidence"})
    return pd.DataFrame(rows)


def analyze_paired_alpha_validation(runs: list[dict[str, Any]], output_dir: str | Path) -> None:
    """Relate paired seed trajectories in alpha distance to validation acceleration."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "alpha_summary_by_step.csv"
    columns = ["seed", "base_optimizer", "optimizer_step", "validation_step",
               "alpha_distance_baseline", "alpha_distance_wwpgd", "delta_alpha_distance",
               "validation_loss_baseline", "validation_loss_wwpgd", "delta_validation_loss",
               "temporal_relationship"]
    if not summary_path.exists() or pd.read_csv(summary_path).empty:
        pd.DataFrame(columns=columns).to_csv(out / "alpha_validation_alignment_by_seed.csv", index=False)
        pd.DataFrame().to_csv(out / "endpoint_event_study.csv", index=False)
        pd.DataFrame().to_csv(out / "acceleration_alpha_association.csv", index=False)
        fig, ax = plt.subplots(); ax.text(.5, .5, "No paired alpha trajectories", ha="center")
        fig.savefig(out / "paired_alpha_and_validation_plot.png", dpi=160); plt.close(fig)
        return
    alpha = pd.read_csv(summary_path)
    grouped: dict[tuple[Any, Any], dict[str, dict[str, Any]]] = {}
    for run in runs:
        ext = run.get("extension"); base = run.get("base_optimizer")
        if ext in {"none", "wwpgd"}:
            grouped.setdefault((run.get("seed"), base), {})[ext] = run
    trajectories, events, seed_rows = [], [], []
    acceleration = pd.read_csv(out / "acceleration_by_seed.csv") if (out / "acceleration_by_seed.csv").exists() else pd.DataFrame()
    for (seed, base), arms in grouped.items():
        if not {"none", "wwpgd"} <= set(arms): continue
        joined = {}
        for ext, suffix in (("none", "baseline"), ("wwpgd", "wwpgd")):
            run = arms[ext]; path = Path(run["run_dir"]); arm_name = run.get("arm_name") or run.get("optimizer_raw")
            aa = alpha[(alpha.seed == seed) & (alpha.arm_name.astype(str) == str(arm_name))]
            joined[suffix] = _backward_validation_join(aa, pd.read_csv(path / "metrics.csv"))
        p = joined["wwpgd"].merge(joined["baseline"], on="optimizer_step", suffixes=("_wwpgd", "_baseline"))
        if p.empty: continue
        p["seed"] = seed; p["base_optimizer"] = base
        p["alpha_distance_wwpgd"] = p["median_absolute_alpha_error_wwpgd"]
        p["alpha_distance_baseline"] = p["median_absolute_alpha_error_baseline"]
        p["delta_alpha_distance"] = p.alpha_distance_wwpgd - p.alpha_distance_baseline
        p["delta_validation_loss"] = p.validation_loss_wwpgd - p.validation_loss_baseline
        # Both arm joins must refer only to outcomes observed by the alpha step.
        p["validation_step"] = p[["validation_step_wwpgd", "validation_step_baseline"]].min(axis=1)
        alpha_first = p.loc[p.delta_alpha_distance.lt(0), "validation_step"].min()
        val_first = p.loc[p.delta_validation_loss.lt(0), "validation_step"].min()
        relation = ("alpha-distance reduction temporally precedes validation improvement" if alpha_first < val_first
                    else "alpha-distance reduction occurs during validation improvement" if alpha_first == val_first
                    else "alpha-distance reduction occurs only after validation improvement") if pd.notna(alpha_first) and pd.notna(val_first) else "relationship not observed"
        p["temporal_relationship"] = relation
        trajectories.append(p[columns])

        decisions = _read_optional(Path(arms["wwpgd"]["run_dir"]), ("wwpgd_endpoint_measurements.csv", "endpoint_measurements.csv", "adaptive_wwpgd_measurements.csv", "adaptive_wwpgd_decisions.csv"))
        active_step = np.nan
        if not decisions.empty and {"optimizer_step", "cache_activated"} <= set(decisions):
            active = decisions[decisions.cache_activated.astype(str).str.lower().isin(["true", "1"])]
            if len(active): active_step = pd.to_numeric(active.optimizer_step, errors="coerce").min()
        ordered = p.sort_values(["validation_step", "optimizer_step"]).drop_duplicates("validation_step", keep="last").reset_index(drop=True)
        if pd.notna(active_step) and len(ordered):
            after = np.flatnonzero(ordered.validation_step.to_numpy() >= active_step)
            if len(after):
                origin = int(after[0])
                for offset, label in [(-1, "one evaluation before activation"), (0, "activation"), (1, "one evaluation after activation"), (2, "two evaluations after activation"), (3, "three evaluations after activation")]:
                    if 0 <= origin + offset < len(ordered):
                        row = ordered.iloc[origin + offset]
                        events.append({"seed": seed, "base_optimizer": base, "event_time": offset,
                                       "event_label": label, "first_active_endpoint_step": active_step,
                                       "evaluation_step": row.validation_step, "delta_alpha_distance": row.delta_alpha_distance,
                                       "delta_validation_loss": row.delta_validation_loss,
                                       "observed": True, "interpretation": relation + "; associated with, not causal evidence"})
                    else:
                        events.append({"seed": seed, "base_optimizer": base, "event_time": offset,
                                       "event_label": label, "first_active_endpoint_step": active_step,
                                       "evaluation_step": np.nan, "delta_alpha_distance": np.nan,
                                       "delta_validation_loss": np.nan, "observed": False,
                                       "interpretation": "evaluation interval outside observed trajectory"})
        projections = _read_optional(Path(arms["wwpgd"]["run_dir"]), ("wwpgd_projection.csv", "adaptive_wwpgd_applications.csv"))
        displacement_col = next((c for c in ("applied_relative_frobenius_change", "relative_frobenius_change_applied",
                                              "relative_frobenius_change", "relative_frobenius_weight_change")
                                 if c in projections), None)
        displacement = pd.to_numeric(projections[displacement_col], errors="coerce").abs().sum() if displacement_col else 0.0
        ordered_p = p.sort_values("optimizer_step")
        first, last = ordered_p.iloc[0], ordered_p.iloc[-1]
        entering = max(0.0, float(last.fraction_inside_configured_target_deadband_wwpgd - first.fraction_inside_configured_target_deadband_wwpgd))
        ar = acceleration[(acceleration.seed == seed) & (acceleration.base_optimizer == base)] if not acceleration.empty else pd.DataFrame()
        seed_rows.append({"seed": seed, "base_optimizer": base,
                          "tokens_saved": pd.to_numeric(ar.get("tokens_saved"), errors="coerce").median() if len(ar) else np.nan,
                          "median_alpha_error_reduction": float(first.alpha_distance_wwpgd - last.alpha_distance_wwpgd),
                          "fraction_layers_entering_target_band": entering,
                          "cumulative_applied_wwpgd_displacement": displacement})
    aligned = pd.concat(trajectories, ignore_index=True) if trajectories else pd.DataFrame(columns=columns)
    aligned.to_csv(out / "alpha_validation_alignment_by_seed.csv", index=False)
    pd.DataFrame(events).to_csv(out / "endpoint_event_study.csv", index=False)
    _correlations(pd.DataFrame(seed_rows)).to_csv(out / "acceleration_alpha_association.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for (seed, base), d in aligned.groupby(["seed", "base_optimizer"]):
        label=f"{base}, seed {seed}"; axes[0].plot(d.optimizer_step, d.delta_alpha_distance, marker="o", label=label)
        axes[1].plot(d.optimizer_step, d.delta_validation_loss, marker="o", label=label)
    axes[0].axhline(0, color="black", lw=.8); axes[1].axhline(0, color="black", lw=.8)
    axes[0].set_ylabel("Delta alpha distance"); axes[1].set(ylabel="Delta validation loss", xlabel="Optimizer step")
    if len(aligned): axes[0].legend(fontsize=7)
    fig.suptitle("Paired per-seed alpha and validation trajectories (association, not causation)")
    fig.tight_layout(); fig.savefig(out / "paired_alpha_and_validation_plot.png", dpi=160); plt.close(fig)
