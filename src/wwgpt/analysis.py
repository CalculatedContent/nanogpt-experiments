from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

OPTIMIZER_LABELS = {
    "adamw": "AdamW",
    "wwpgd": "AdamW + WW-PGD",
    "adamw_wwpgd": "AdamW+WW-PGD",
    "muon": "Muon",
    "muon_wwpgd": "Muon+WW-PGD",
    "stableadamw": "StableAdamW",
    "stableadamw_wwpgd": "StableAdamW+WW-PGD",
    "stable_adamw": "StableAdamW",
    "stable_adamw_wwpgd": "StableAdamW+WW-PGD",
}
PROJECTED_LAYER_PATTERNS = (".attn.key", ".attn.query", ".attn.value", ".attn.proj", ".mlp.0", ".mlp.2")
SCHEMA_V3_ARMS = ("adamw", "adamw_wwpgd", "muon", "muon_wwpgd", "stableadamw", "stableadamw_wwpgd")
TRIAL_CANONICAL_ARMS = ("adamw", "adamw_wwpgd", "muon", "muon_wwpgd", "stable_adamw", "stable_adamw_wwpgd")
TRIAL_CANONICAL_PAIRS = {"adamw": "adamw_wwpgd", "muon": "muon_wwpgd", "stable_adamw": "stable_adamw_wwpgd"}
BASE_PAIRS = {"adamw": "adamw_wwpgd", "muon": "muon_wwpgd", "stableadamw": "stableadamw_wwpgd", "stable_adamw": "stable_adamw_wwpgd"}
PAIR_MATCH_FIELDS = ("initialization_hash", "tokenizer_hash", "validation_probe_hash", "training_probe_hash", "realized_tokens")

@dataclass(frozen=True)
class RunRecord:
    pair_id: str
    pair_dir: Path
    run_dir: Path
    optimizer_raw: str
    optimizer_family: str
    optimizer_label: str
    seed: int | None
    manifest: dict[str, Any]
    complete: bool
    legacy: bool
    valid_for_science: bool

@dataclass(frozen=True)
class PairCandidate:
    pair_id: str
    pair_dir: Path
    seed: int | None
    runs: dict[str, RunRecord]
    valid: bool
    exclusion_reason: str
    newest_mtime: float

# manifest loading
def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path and path.exists() else {}

def load_csv_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path and path.exists() else pd.DataFrame()

def resolve_experiment_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if any(root.glob("pair_*")) or any(root.glob("trial_*")):
        return root
    conventional = root / "experiments" / "level_00" / "multiplier_20"
    if conventional.exists() and (any(conventional.glob("pair_*")) or any(conventional.glob("trial_*"))):
        return conventional.resolve()
    return root

def _manifest_value(man: dict[str, Any], key: str) -> Any:
    if key in man:
        return man[key]
    for sub in ("shared", "config", "model_config", "data", "training", "parameter_report", "token_budget"):
        val = man.get(sub)
        if isinstance(val, dict) and key in val:
            return val[key]
    return None

def normalize_optimizer(raw: str, include_legacy: bool = False) -> dict[str, Any]:
    raw = str(raw).lower()
    if raw in {"adamw", "muon", "stableadamw", "stable_adamw"}:
        fam, legacy = raw, False
    elif raw == "adamw_wwpgd_reference":
        fam, legacy = "wwpgd", False
    elif raw in {"adamw_wwpgd", "muon_wwpgd", "stableadamw_wwpgd", "stable_adamw_wwpgd"}:
        fam, legacy = raw, False
    else:
        fam, legacy = raw, True
    allowed = raw in {"adamw", "adamw_wwpgd_reference", "muon", "muon_wwpgd", "stableadamw", "stableadamw_wwpgd", "stable_adamw", "stable_adamw_wwpgd"} or (include_legacy and raw == "adamw_wwpgd")
    return {"optimizer_raw": raw, "optimizer_family": fam, "optimizer_label": OPTIMIZER_LABELS.get(fam, raw), "legacy_optimizer": legacy, "allowed_by_default": allowed}

def _run_mtime(run: Path) -> float:
    try:
        return max([run.stat().st_mtime] + [p.stat().st_mtime for p in run.glob("*")])
    except Exception:
        return 0.0

# metric loading
def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for src, dst in {"tokens_processed": "tokens_seen", "val_loss": "validation_loss", "elapsed_time": "elapsed_seconds", "projection_overhead": "projection_seconds"}.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    return out

def normalize_projection_records(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for src, dst in {"actual_step": "step", "actual_tokens_seen": "tokens_seen", "TraceLog_before": "trace_log_before", "TraceLog_after": "trace_log_after"}.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    if {"trace_log_after", "trace_log_before"}.issubset(out.columns):
        out["trace_log_delta"] = pd.to_numeric(out.trace_log_after, errors="coerce") - pd.to_numeric(out.trace_log_before, errors="coerce")
    return out

def is_projected_transformer_matrix(name: str) -> bool:
    return str(name).startswith("blocks.") and any(p in str(name) for p in PROJECTED_LAYER_PATTERNS)

def normalize_spectral_records(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "tokens_processed" in out.columns and "tokens_seen" not in out.columns:
        out["tokens_seen"] = out["tokens_processed"]
    if "layer_name" not in out.columns:
        for c in ("longname", "name"):
            if c in out.columns:
                out["layer_name"] = out[c]
                break
    if "spectral_estimator" not in out.columns:
        out["spectral_estimator"] = pd.NA
    if "valid_weightwatcher" not in out.columns:
        out["valid_weightwatcher"] = out["spectral_estimator"].astype(str).str.lower().eq("weightwatcher")
    if "layer_group" not in out.columns and "layer_name" in out.columns:
        ln = out["layer_name"].astype(str)
        out["layer_group"] = np.select([ln.str.contains("wte|wpe"), ln.str.contains("lm_head"), ln.apply(is_projected_transformer_matrix)], ["embedding", "lm_head", "projected_transformer_matrix"], default="other")
    return out

def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    d = {"run_dir": run_dir, "manifest": read_json_file(run_dir / "manifest.json"), "manifest.json": read_json_file(run_dir / "manifest.json"), "complete": read_json_file(run_dir / "run_complete.json"), "run_complete.json": read_json_file(run_dir / "run_complete.json")}
    d["metrics"] = normalize_metrics(load_csv_file(run_dir / "metrics.csv")); d["metrics.csv"] = d["metrics"]
    d["spectral"] = normalize_spectral_records(load_csv_file(run_dir / "spectral.csv")); d["spectral.csv"] = d["spectral"]
    d["projection"] = normalize_projection_records(load_csv_file(run_dir / "wwpgd_projection.csv")); d["wwpgd_projection.csv"] = d["projection"]
    return d

# trial validation
def _arm_base_ext(arm: str, man: dict[str, Any] | None = None) -> tuple[str, str]:
    man = man or {}
    base = str(man.get("base_optimizer") or "").lower()
    ext = str(man.get("extension") or "").lower()
    if base and ext:
        return base.replace("stable_adamw", "stableadamw"), ext
    raw = arm.replace("stable_adamw", "stableadamw")
    return raw.removesuffix("_wwpgd"), "wwpgd" if raw.endswith("_wwpgd") else "none"

def _build_record(pair_id: str, pair_dir: Path, run_dir: Path, arm: str, man: dict[str, Any], valid: bool) -> RunRecord:
    base, ext = _arm_base_ext(arm, man)
    family = f"{base}_wwpgd" if ext == "wwpgd" else base
    if arm == "adamw_wwpgd_reference":
        family = "wwpgd"
    seed = _manifest_value(man, "seed")
    return RunRecord(pair_id, pair_dir, run_dir, arm, family, OPTIMIZER_LABELS.get(family, family), int(seed) if seed is not None else None, man, (run_dir / "run_complete.json").exists(), int(man.get("scientific_schema_version") or 0) < 2, valid)

def _latest_completed_run_for_compat(optimizer_dir: Path) -> Path | None:
    runs = sorted([p for p in optimizer_dir.glob("run_*") if p.is_dir()], key=lambda p: (_run_mtime(p), p.name), reverse=True)
    complete = [p for p in runs if (p / "manifest.json").exists() and (p / "metrics.csv").exists() and (p / "run_complete.json").exists()]
    return complete[0] if complete else (runs[0] if runs else None)

def discover_trial_manifests(results_root: Path) -> list[dict[str, Any]]:
    root = resolve_experiment_root(results_root)
    out = []
    for trial in sorted([p for p in root.glob("trial_*") if p.is_dir()] if root.exists() else []):
        man = read_json_file(trial / "trial_manifest.json")
        if not man:
            continue
        arms = [a.get("arm_name") for a in man.get("arms", [])]
        valid = tuple(arms) == TRIAL_CANONICAL_ARMS and man.get("pairs") == [{"baseline": b, "wwpgd": w} for b, w in TRIAL_CANONICAL_PAIRS.items()]
        out.append({"trial_id": man.get("trial_id", trial.name), "trial_dir": trial, "manifest": man, "valid": valid, "exclusion_reason": "" if valid else "incomplete canonical trial manifest"})
    return out

def _validate_pair_records(records: dict[str, RunRecord], bases: Iterable[str]) -> tuple[bool, str]:
    reasons = []
    for base in bases:
        ww = BASE_PAIRS[base]
        a, w = records.get(base), records.get(ww) or (records.get("wwpgd") if base == "adamw" else None)
        if not (a and w):
            reasons.append(f"missing {base} within-optimizer pair"); continue
        for r in (a, w):
            for fn in ("manifest.json", "metrics.csv", "run_complete.json"):
                if not (r.run_dir / fn).exists(): reasons.append(f"{r.optimizer_raw} missing {fn}")
            if not r.complete: reasons.append(f"{r.optimizer_raw} incomplete")
            if not r.valid_for_science: reasons.append(f"{r.optimizer_raw} invalid schema/profile")
        if a.seed != w.seed: reasons.append(f"{base} seed mismatch")
        for key in PAIR_MATCH_FIELDS:
            av, wv = _manifest_value(a.manifest, key), _manifest_value(w.manifest, key)
            if av is not None and wv is not None and av != wv: reasons.append(f"{base} {key} mismatch")
        ad = _manifest_value(a.manifest, "data_hash") or _manifest_value(a.manifest, "corpus_hash")
        wd = _manifest_value(w.manifest, "data_hash") or _manifest_value(w.manifest, "corpus_hash")
        if ad is not None and wd is not None and ad != wd: reasons.append(f"{base} data_hash/corpus_hash mismatch")
    return not reasons, "; ".join(dict.fromkeys(reasons))

def discover_pair_candidates(results_root: Path, include_legacy: bool = False) -> list[PairCandidate]:
    root = resolve_experiment_root(results_root); out = []
    if not root.exists(): return out
    for pair in sorted([p for p in root.glob("pair_*") if p.is_dir()]):
        records: dict[str, RunRecord] = {}
        for arm in list(SCHEMA_V3_ARMS) + ["stable_adamw", "stable_adamw_wwpgd", "adamw_wwpgd_reference"] + (["adamw_wwpgd"] if include_legacy else []):
            rd = _latest_completed_run_for_compat(pair / arm)
            if not rd: continue
            man = read_json_file(rd / "manifest.json")
            raw = str(man.get("optimizer") or man.get("arm_name") or arm).lower()
            schema = int(man.get("scientific_schema_version") or 0)
            norm = normalize_optimizer(raw, include_legacy)
            valid = bool(man.get("valid_for_science", True) is True and ((schema >= 3 and norm["optimizer_family"] in set(SCHEMA_V3_ARMS) | {"stable_adamw", "stable_adamw_wwpgd"}) or (schema >= 2 and norm["optimizer_family"] in {"adamw", "wwpgd"})))
            rec = _build_record(pair.name, pair, rd, raw, man, valid)
            records[rec.optimizer_family] = rec
        bases = [b for b in ("adamw", "muon", "stableadamw", "stable_adamw") if records.get(b) or records.get(BASE_PAIRS[b])]
        if not bases and (records.get("adamw") or records.get("wwpgd")): bases = ["adamw"]
        valid, reason = _validate_pair_records(records, bases) if bases else (False, "no complete within-optimizer canonical arm pair")
        mt = max([_run_mtime(r.run_dir) for r in records.values()] or [pair.stat().st_mtime])
        seed = next((r.seed for r in records.values() if r.seed is not None), None)
        out.append(PairCandidate(pair.name, pair, seed, records, valid, reason, mt))
    return out

def select_canonical_pairs(candidates: list[PairCandidate]) -> tuple[list[PairCandidate], pd.DataFrame]:
    selected, rows, by_seed = [], [], {}
    for c in candidates:
        by_seed.setdefault(c.seed, []).append(c)
    for seed, cs in by_seed.items():
        val = sorted([c for c in cs if c.valid], key=lambda c: (c.newest_mtime, c.pair_id), reverse=True)
        chosen = val[0] if val else None
        if chosen: selected.append(chosen)
        for c in cs:
            rows.append({"pair_id": c.pair_id, "pair_dir": str(c.pair_dir), "seed": c.seed, "status": "selected" if c is chosen else "excluded", "exclusion_reason": "" if c is chosen else (c.exclusion_reason or "duplicate older complete pair"), "valid_complete_pair": c.valid, "newest_mtime": c.newest_mtime})
    return sorted(selected, key=lambda c: (c.seed or -1)), pd.DataFrame(rows)

def _row_from_record(r: RunRecord) -> dict[str, Any]:
    return {"pair_id": r.pair_id, "pair_dir": r.pair_dir, "run_dir": r.run_dir, "seed": r.seed, "optimizer_raw": r.optimizer_raw, "optimizer_family": r.optimizer_family, "optimizer_label": r.optimizer_label, "base_optimizer": _arm_base_ext(r.optimizer_family, r.manifest)[0], "extension": _arm_base_ext(r.optimizer_family, r.manifest)[1], "manifest": r.manifest, "complete": r.complete, "valid_for_science": r.valid_for_science}

def discover_canonical_runs(results_root: Path, include_legacy: bool = False) -> list[dict[str, Any]]:
    trials = discover_trial_manifests(results_root)
    if trials:
        rows = []
        for t in trials:
            if not t["valid"]: continue
            man, shared = t["manifest"], t["manifest"].get("shared", {})
            for arm in man.get("arms", []):
                arm_name = arm["arm_name"]
                rd = _latest_completed_run_for_compat(t["trial_dir"] / arm_name) or (t["trial_dir"] / arm_name)
                rows.append(_row_from_record(_build_record(man["trial_id"], t["trial_dir"], rd, arm_name, {**shared, **arm, "scientific_schema_version": man.get("scientific_schema_version"), "trial_manifest": man}, t["valid"])))
        return rows
    pairs, _ = select_canonical_pairs(discover_pair_candidates(results_root, include_legacy))
    rows = []
    for c in pairs:
        for arm in ("adamw", "wwpgd", "adamw_wwpgd", "muon", "muon_wwpgd", "stableadamw", "stableadamw_wwpgd", "stable_adamw", "stable_adamw_wwpgd"):
            if arm in c.runs:
                rows.append(_row_from_record(c.runs[arm]))
    return rows

# aggregation / paired statistics
def student_t_summary(values: Iterable[float], confidence: float = .95) -> dict[str, float | int]:
    s = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna(); n = len(s)
    mean = float(s.mean()) if n else math.nan; sd = float(s.std(ddof=1)) if n > 1 else 0.0; se = sd / math.sqrt(n) if n else math.nan
    half = float(stats.t.ppf((1 + confidence) / 2, n - 1) * se) if n > 1 else 0.0
    return {"n": int(n), "mean": mean, "std": sd, "sample_std": sd, "se": se, "standard_error": se, "ci_low": mean - half if n else math.nan, "ci_high": mean + half if n else math.nan, "ci95_low": mean - half if n else math.nan, "ci95_high": mean + half if n else math.nan, "median": float(s.median()) if n else math.nan}

def summary(s: pd.Series) -> dict[str, float | int]:
    return student_t_summary(s)

def terminal_results(runs: list[dict[str, Any]], metric: str = "validation_loss") -> pd.DataFrame:
    rows = []
    for r in runs:
        if not r.get("run_dir"): continue
        m = (r.get("artifacts") or {}).get("metrics") if isinstance(r.get("artifacts"), dict) else load_run_artifacts(Path(r["run_dir"]))["metrics"]
        if metric not in m: continue
        vals = pd.to_numeric(m.sort_values("tokens_seen" if "tokens_seen" in m else "step")[metric], errors="coerce").dropna()
        if vals.empty: continue
        rows.append({"pair_id": r.get("pair_id"), "seed": r.get("seed"), "optimizer_family": r.get("optimizer_family") or normalize_optimizer(r.get("optimizer_raw") or r.get("optimizer", ""), True)["optimizer_family"], "final": float(vals.iloc[-1]), "minimum": float(vals.min())})
    d = pd.DataFrame(rows)
    if d.empty: return d
    p = d.pivot_table(index=["pair_id", "seed"], columns="optimizer_family", values=["final", "minimum"], aggfunc="first"); p.columns = [f"{fam}_{met}_{metric}" for met, fam in p.columns]; p = p.reset_index()
    for base, ww in [("adamw", "wwpgd"), ("adamw", "adamw_wwpgd"), ("muon", "muon_wwpgd"), ("stableadamw", "stableadamw_wwpgd"), ("stable_adamw", "stable_adamw_wwpgd")]:
        a, w = f"{base}_final_{metric}", f"{ww}_final_{metric}"
        if {a, w}.issubset(p.columns): p[f"{ww}_minus_{base}_{metric}"] = p[w] - p[a]
    if metric == "validation_loss":
        if f"wwpgd_minus_adamw_{metric}" in p.columns:
            p["wwpgd_minus_adamw_final_validation_loss"] = p[f"wwpgd_minus_adamw_{metric}"]
        if f"adamw_wwpgd_minus_adamw_{metric}" in p.columns:
            p["wwpgd_minus_adamw_final_validation_loss"] = p[f"adamw_wwpgd_minus_adamw_{metric}"]
    return p

def paired_extension_effects(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    keys = [c for c in ["experiment_profile", "scientific_schema_version", "level", "token_multiplier", "base_optimizer", "seed"] if c in df.columns]
    if "base_optimizer" not in keys or "extension" not in df.columns: return pd.DataFrame()
    p = df.pivot_table(index=keys, columns="extension", values=metric, aggfunc="first").reset_index()
    if {"none", "wwpgd"}.issubset(p.columns):
        p[f"wwpgd_minus_none_{metric}"] = p["wwpgd"] - p["none"]
        p["paired_comparison"] = p["base_optimizer"].map({"adamw": "AdamW+WW-PGD - AdamW", "muon": "Muon+WW-PGD - Muon", "stableadamw": "StableAdamW+WW-PGD - StableAdamW", "stable_adamw": "StableAdamW+WW-PGD - StableAdamW"})
    return p

def paired_effect_estimates(paired: pd.DataFrame, metric: str) -> pd.DataFrame:
    col = f"wwpgd_minus_none_{metric}"
    rows = []
    if col not in paired: return pd.DataFrame()
    for base, g in paired.groupby("base_optimizer", dropna=False):
        s = student_t_summary(g[col])
        rows.append({"base_optimizer": base, "metric": metric, "paired_difference_column": col, "paired_effect_mean": s["mean"], "paired_effect_std": s["std"], "paired_effect_ci_low": s["ci_low"], "paired_effect_ci_high": s["ci_high"], "n_pairs": s["n"]})
    return pd.DataFrame(rows)

# plotting inputs
def align_curves(curves: list[pd.DataFrame], x_col: str, y_col: str, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    clean = []
    for df in curves:
        if x_col in df and y_col in df:
            d = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(x_col).drop_duplicates(x_col)
            if len(d) >= 2: clean.append(d)
    if not clean: return np.array([]), np.empty((0, 0))
    lo, hi = max(d[x_col].min() for d in clean), min(d[x_col].max() for d in clean)
    if not np.isfinite(lo) or hi <= lo: return np.array([]), np.empty((0, 0))
    grid = np.linspace(lo, hi, points)
    return grid, np.vstack([np.interp(grid, d[x_col], d[y_col]) for d in clean])

def paired_curve_differences(pairs: list[tuple[pd.DataFrame, pd.DataFrame]], x_col: str, y_col: str, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    grids, diffs = [], []
    for a, w in pairs:
        g, v = align_curves([a, w], x_col, y_col, points)
        if v.shape[0] == 2: grids.append(g); diffs.append(v[1] - v[0])
    if not diffs: return np.array([]), np.empty((0, 0))
    lo, hi = max(g.min() for g in grids), min(g.max() for g in grids); grid = np.linspace(lo, hi, points)
    return grid, np.vstack([np.interp(grid, g, d) for g, d in zip(grids, diffs)])

# report export
def build_run_inventory(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        art = load_run_artifacts(Path(r["run_dir"])); m = art["metrics"]; man = r.get("manifest") or art["manifest"]
        final = m.sort_values("tokens_seen" if "tokens_seen" in m.columns else "step").tail(1) if not m.empty else pd.DataFrame()
        rows.append({"seed": r.get("seed"), "pair_id": r.get("pair_id"), "optimizer_raw": r.get("optimizer_raw"), "optimizer_family": r.get("optimizer_family"), "base_optimizer": r.get("base_optimizer"), "extension": r.get("extension"), "run_dir": str(r.get("run_dir")), "final_validation_loss": final["validation_loss"].iloc[0] if len(final) and "validation_loss" in final else np.nan, "minimum_validation_loss": pd.to_numeric(m.get("validation_loss"), errors="coerce").min() if "validation_loss" in m else np.nan, "final_test_loss": final["test_loss"].iloc[0] if len(final) and "test_loss" in final else np.nan, "realized_tokens": _manifest_value(man, "realized_tokens"), "scientific_schema_version": man.get("scientific_schema_version")})
    return pd.DataFrame(rows)

def build_pair_audit(candidates: list[PairCandidate]) -> pd.DataFrame:
    return select_canonical_pairs(candidates)[1]

def analyze_results(results_root: Path) -> Path:
    out = Path(results_root) / "analysis"; out.mkdir(parents=True, exist_ok=True)
    runs = discover_canonical_runs(results_root, include_legacy=True)
    inv = build_run_inventory(runs) if runs else pd.DataFrame(); inv.to_csv(out / "runs_manifest.csv", index=False)
    terminal_results(runs, "validation_loss").to_csv(out / "paired_validation_metric_differences.csv", index=False)
    terminal_results(runs, "test_loss").to_csv(out / "paired_test_metric_differences.csv", index=False)
    if not inv.empty and "final_validation_loss" in inv:
        paired = paired_extension_effects(inv.rename(columns={"final_validation_loss": "loss"}).dropna(subset=["loss"]), "loss")
        paired.to_csv(out / "paired_validation_effects_by_seed.csv", index=False)
        paired_effect_estimates(paired, "loss").to_csv(out / "paired_validation_effect_estimates.csv", index=False)
    pd.DataFrame([{"status": "not_fit", "note": "no significance claim; no hypothesis test implemented"}]).to_csv(out / "scaling_fit_results.csv", index=False)
    (out / "analysis_manifest.json").write_text(json.dumps({"source": str(results_root), "completed_runs": len(runs)}))
    return out


def audit_spectral_validity(spectral: pd.DataFrame) -> pd.DataFrame:
    """Return a machine-readable row-level validity audit for WeightWatcher science rows."""
    rows: list[dict[str, Any]] = []
    required = ["step", "tokens_seen", "alpha", "spectral_estimator", "valid_weightwatcher", "valid_for_science", "scientific_schema_version"]
    finite_fields = ["step", "tokens_seen", "alpha"]
    optional_finite_fields = ["D", "num_evals", "spectral_norm", "stable_rank"]
    schemas = set(pd.to_numeric(spectral.get("scientific_schema_version", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique()) if not spectral.empty else set()
    incompatible_schema_pool = len(schemas) > 1
    for i, row in spectral.reset_index(drop=True).iterrows():
        reasons: list[str] = []
        for col in required:
            if col not in spectral.columns or pd.isna(row.get(col)):
                reasons.append(f"missing_{col}")
        estimator = str(row.get("spectral_estimator", "")).lower()
        if estimator != "weightwatcher":
            reasons.append("spectral_estimator_not_weightwatcher")
        valid_weightwatcher = str(row.get("valid_weightwatcher", False)).lower() in {"true", "1"}
        valid_for_science = str(row.get("valid_for_science", False)).lower() in {"true", "1"}
        if not valid_weightwatcher:
            reasons.append("valid_weightwatcher_false")
        if not valid_for_science:
            reasons.append("valid_for_science_false")
        schema = pd.to_numeric(pd.Series([row.get("scientific_schema_version")]), errors="coerce").iloc[0] if "scientific_schema_version" in spectral.columns else np.nan
        if not np.isfinite(schema) or schema < 2:
            reasons.append("unsupported_scientific_schema_version")
        if incompatible_schema_pool:
            reasons.append("incompatible_schema_pool")
        for col in finite_fields + [c for c in optional_finite_fields if c in spectral.columns]:
            val = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if not np.isfinite(val):
                reasons.append(f"nonfinite_{col}")
        rows.append({"row_index": i, "seed": row.get("seed"), "pair_id": row.get("pair_id"), "optimizer_family": row.get("optimizer_family"), "step": row.get("step"), "tokens_seen": row.get("tokens_seen"), "layer_name": row.get("layer_name"), "spectral_estimator": row.get("spectral_estimator"), "scientific_schema_version": row.get("scientific_schema_version"), "valid_for_science": row.get("valid_for_science"), "valid_weightwatcher": row.get("valid_weightwatcher"), "valid_for_weightwatcher_science": not reasons, "invalid_reasons": ";".join(dict.fromkeys(reasons))})
    return pd.DataFrame(rows)

# compatibility helpers
def discover_pair_directories(results_root: Path) -> list[Path]:
    return sorted([p for p in Path(results_root).iterdir() if p.is_dir() and p.name.startswith("pair_")]) if Path(results_root).exists() else []

def select_valid_run_directory(optimizer_dir: Path) -> tuple[Path | None, str]:
    rd = _latest_completed_run_for_compat(optimizer_dir)
    if rd and (rd / "run_complete.json").exists(): return rd, "selected most recent completed valid run"
    return (rd, "no valid run found; selected newest directory for audit") if rd else (None, "no run found")

def discover_experiment_runs(results_root: Path) -> list[dict[str, Any]]:
    rows = []
    for pair in discover_pair_directories(results_root):
        opts = ["adamw", "adamw_wwpgd"] + (["adamw_wwpgd_reference"] if (pair / "adamw_wwpgd_reference").exists() else [])
        for opt in opts:
            rd, note = select_valid_run_directory(pair / opt); art = load_run_artifacts(rd) if rd else {"files_loaded": []}; man = art.get("manifest", {})
            seed = man.get("seed")
            if seed is None:
                m = re.search(r"pair_(\d+)", pair.name); seed = int(m.group(1)) if m else None
            norm = normalize_optimizer(opt, include_legacy=True)
            rows.append({"pair_id": pair.name, "pair_dir": pair, "optimizer": opt, "optimizer_raw": opt, "optimizer_family": norm["optimizer_family"], "optimizer_label": norm["optimizer_label"], "run_dir": rd, "selection_note": note, "seed": seed, "artifacts": art})
    return rows

def vocab_size_from_artifacts(artifacts: dict[str, Any]) -> int | None:
    return _manifest_value(artifacts.get("manifest") or artifacts.get("manifest.json") or {}, "vocab_size")

def add_generalization_measures(metrics: pd.DataFrame, vocab_size: int | None = None) -> pd.DataFrame:
    out = normalize_metrics(metrics)
    if "validation_loss" in out and "val_loss" not in out: out["val_loss"] = out["validation_loss"]
    for split in ("train", "val"):
        loss = f"{split}_loss"
        if loss in out:
            if f"{split}_perplexity" not in out:
                out[f"{split}_perplexity"] = np.exp(pd.to_numeric(out[loss], errors="coerce").clip(upper=20))
            if f"{split}_bits_per_token" not in out:
                out[f"{split}_bits_per_token"] = pd.to_numeric(out[loss], errors="coerce") / np.log(2)
            if vocab_size and vocab_size > 1 and f"{split}_token_prediction_capacity" not in out:
                out[f"{split}_token_prediction_capacity"] = 1 - out[f"{split}_bits_per_token"] / np.log2(vocab_size)
    if {"val_loss", "train_loss"}.issubset(out.columns) and "generalization_gap" not in out: out["generalization_gap"] = out["val_loss"] - out["train_loss"]
    if {"val_perplexity", "train_perplexity"}.issubset(out.columns): out["perplexity_gap"] = out["val_perplexity"] - out["train_perplexity"]; out["perplexity_ratio"] = out["val_perplexity"] / out["train_perplexity"]
    if {"val_token_prediction_capacity", "train_token_prediction_capacity"}.issubset(out.columns): out["capacity_generalization_gap"] = out["train_token_prediction_capacity"] - out["val_token_prediction_capacity"]
    return out

def completed_runs(root: Path, scientific_only: bool = True) -> list[Path]:
    out = []
    for p in Path(root).rglob("run_complete.json"):
        man = read_json_file(p.parent / "manifest.json")
        if scientific_only and man.get("valid_for_science", True) is not True: continue
        out.append(p.parent)
    return out

def discover_scaling_runs(root: Path, include_legacy: bool = False) -> pd.DataFrame:
    rows = []
    for exp in sorted(Path(root).glob("experiments/level_*/multiplier_*")):
        for r in discover_canonical_runs(exp, include_legacy):
            man = r["manifest"]
            rows.append({"level": _infer_level(Path(r["run_dir"])), "token_multiplier": _infer_multiplier(Path(r["run_dir"])), "seed": r["seed"], "pair_id": r["pair_id"], "optimizer_family": r["optimizer_family"], "optimizer_raw": r["optimizer_raw"], "run_dir": str(r["run_dir"]), "parameter_count": _manifest_value(man, "total_parameters") or _manifest_value(man, "parameter_count"), "non_embedding_parameters": _manifest_value(man, "non_embedding_parameters"), "realized_tokens": _manifest_value(man, "realized_tokens") or _manifest_value(man, "requested_tokens"), "estimated_flops": _manifest_value(man, "estimated_flops")})
    return pd.DataFrame(rows).drop_duplicates(subset=["run_dir"]) if rows else pd.DataFrame()

def _infer_level(p: Path) -> str:
    return next((part for part in p.parts if part.startswith("level_")), "unknown")

def _infer_multiplier(p: Path) -> str:
    return next((part for part in p.parts if part.startswith("multiplier_")), "unknown")

def scaling_design_points(run_inventory: pd.DataFrame) -> pd.DataFrame:
    if run_inventory.empty: return pd.DataFrame()
    rows = []
    for keys, g in run_inventory.groupby(["level", "token_multiplier", "optimizer_family", "parameter_count", "realized_tokens"], dropna=False):
        vals = []
        for rd in g["run_dir"]:
            m = normalize_metrics(load_csv_file(Path(rd) / "metrics.csv"))
            if "validation_loss" in m and not m.empty: vals.append(pd.to_numeric(m.sort_values("tokens_seen" if "tokens_seen" in m else "step")["validation_loss"], errors="coerce").dropna().iloc[-1])
        summ = student_t_summary(vals)
        rows.append(dict(zip(["level", "token_multiplier", "optimizer_family", "parameter_count", "realized_tokens"], keys), seed_count=len(vals), mean_terminal_validation_loss=summ["mean"], loss_std=summ["std"], estimated_flops=pd.to_numeric(g.get("estimated_flops"), errors="coerce").mean()))
    return pd.DataFrame(rows)

def scaling_readiness(design: pd.DataFrame) -> pd.DataFrame:
    if design.empty: return pd.DataFrame([{"ready": False, "reason": "no canonical scientific design points discovered", "needed": "completed explicit-manifest paired experiments"}])
    nN = design["parameter_count"].nunique(dropna=True); nD = design["realized_tokens"].nunique(dropna=True); pts = len(design[["level", "token_multiplier", "parameter_count", "realized_tokens"]].drop_duplicates())
    ready = nN >= 2 and nD >= 2 and pts >= 4
    return pd.DataFrame([{"ready": ready, "reason": "ready for nonlinear scaling fit" if ready else f"insufficient grid: {nN} parameter count(s), {nD} token budget(s), {pts} design point(s)", "needed": "" if ready else "add model levels and token multipliers with completed paired seeds", "parameter_counts": nN, "token_budgets": nD, "design_points": pts}])
