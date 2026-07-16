from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from wwgpt.utils import unique_dir, write_json


def summary(s: pd.Series) -> dict[str, float | int]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(len(s)); sd = float(s.std(ddof=1)) if n > 1 else 0.0; se = sd / (n**0.5) if n else 0.0
    ci = float(stats.t.ppf(0.975, n - 1) * se) if n > 1 else 0.0
    return {"n": n, "mean": float(s.mean()) if n else float("nan"), "sample_std": sd, "standard_error": se, "ci95_low": float(s.mean()-ci) if n else float("nan"), "ci95_high": float(s.mean()+ci) if n else float("nan"), "median": float(s.median()) if n else float("nan"), "iqr": float(s.quantile(.75)-s.quantile(.25)) if n else float("nan"), "min": float(s.min()) if n else float("nan"), "max": float(s.max()) if n else float("nan")}


def errorbar_indices(n_points: int, max_points: int = 24) -> np.ndarray:
    """Return evenly spaced indices for readable error bars on dense notebook plots."""
    if n_points <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, n_points - 1, min(max_points, n_points), dtype=int))


def mean_ci95(values: pd.Series) -> tuple[float, float, int]:
    """Return mean, 95% confidence-interval half-width, and sample size."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    n = int(len(v))
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(v.iloc[0]), 0.0, 1
    se = float(v.std(ddof=1) / np.sqrt(n))
    return float(v.mean()), float(stats.t.ppf(0.975, n - 1) * se), n


def collect_metrics(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Return one normalized metrics table with run metadata attached."""
    frames = []
    for run in runs:
        metrics = normalize_metrics(run.get("artifacts", {}).get("metrics.csv", pd.DataFrame()))
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics["optimizer"] = run.get("optimizer")
        metrics["seed"] = run.get("seed")
        metrics["pair_id"] = run.get("pair_id")
        metrics["run_dir"] = str(run.get("run_dir"))
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_metric_curve(metrics: pd.DataFrame, y_col: str, output_path: Path, *, title: str | None = None, ylabel: str | None = None, x_col: str | None = None) -> Path | None:
    """Plot per-run curves plus optimizer means with 95% CI error bars."""
    if metrics.empty or y_col not in metrics.columns or "optimizer" not in metrics.columns:
        return None
    x_col = x_col or ("tokens_seen" if "tokens_seen" in metrics.columns else "step")
    if x_col not in metrics.columns:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    colors = {"adamw": "tab:blue", "adamw_wwpgd": "tab:orange"}
    for optimizer, opt_df in metrics.groupby("optimizer"):
        curves = []
        color = colors.get(str(optimizer), None)
        for _, run_df in opt_df.groupby("run_dir"):
            d = run_df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(x_col)
            if len(d) < 2:
                continue
            plt.plot(d[x_col], d[y_col], color=color, alpha=0.25, linewidth=1)
            curves.append(d)
        grid, vals = align_curves(curves, x_col, y_col)
        if len(grid):
            mean = vals.mean(axis=0)
            band = np.zeros_like(mean)
            if vals.shape[0] > 1:
                band = stats.t.ppf(0.975, vals.shape[0] - 1) * vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0])
            plt.plot(grid, mean, color=color, linewidth=2.5, label=f"{optimizer} mean (n={vals.shape[0]})")
            idx = errorbar_indices(len(grid))
            plt.errorbar(grid[idx], mean[idx], yerr=band[idx], fmt="o", color=color, markersize=3, linewidth=1, capsize=3, label=f"{optimizer} mean ± 95% CI")
    plt.xlabel(x_col.replace("_", " "))
    plt.ylabel(ylabel or y_col.replace("_", " "))
    plt.title(title or f"{(ylabel or y_col).replace('_', ' ').title()} by optimizer")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def read_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON object from *path* with a clear error for malformed files."""
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"failed to read JSON file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}, found {type(data).__name__}")
    return data


def read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file emitted by training events and return object records."""
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            rows.append(obj)
    except Exception as exc:
        raise ValueError(f"failed to read JSONL file {path}: {exc}") from exc
    return rows


def load_csv_file(path: Path) -> pd.DataFrame:
    """Load a repository-emitted CSV file, returning an empty frame for absent optional files."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"failed to read CSV file {path}: {exc}") from exc


def discover_pair_directories(results_root: Path) -> list[Path]:
    """Return sorted pair_* directories directly under a multiseed result root."""
    if not results_root.exists():
        return []
    return sorted([p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("pair_")])


def _run_sort_key(path: Path) -> tuple[bool, float, str]:
    complete = (path / "run_complete.json").exists()
    mtimes = [f.stat().st_mtime for f in path.glob("*") if f.exists()]
    return (complete, max(mtimes) if mtimes else path.stat().st_mtime, path.name)


def select_valid_run_directory(optimizer_dir: Path) -> tuple[Path | None, str]:
    """Select the most recent valid run_* directory from an optimizer directory."""
    if not optimizer_dir.exists():
        return None, "optimizer directory missing"
    candidates = sorted([p for p in optimizer_dir.iterdir() if p.is_dir() and p.name.startswith("run_")], key=_run_sort_key, reverse=True)
    if not candidates:
        return None, "no run_* directories found"
    valid = [p for p in candidates if (p / "manifest.json").exists() and (p / "metrics.csv").exists()]
    complete = [p for p in valid if (p / "run_complete.json").exists()]
    if complete:
        return complete[0], "selected most recent completed valid run"
    if valid:
        return valid[0], "selected most recent valid but incomplete run"
    return candidates[0], "no valid run found; selected newest directory for audit"


def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    """Load standard files written by wwgpt training without assuming optional files exist."""
    artifacts: dict[str, Any] = {"run_dir": run_dir, "files_loaded": []}
    for name in ["manifest.json", "run_complete.json", "environment.json", "data_manifest.json", "tokenizer_manifest.json"]:
        path = run_dir / name
        if path.exists():
            artifacts[name] = read_json_file(path); artifacts["files_loaded"].append(path)
    init = run_dir / "initialization_hash.txt"
    if init.exists():
        artifacts["initialization_hash.txt"] = init.read_text().strip(); artifacts["files_loaded"].append(init)
    for name in ["metrics.csv", "spectral.csv", "wwpgd_projection.csv"]:
        path = run_dir / name
        if path.exists():
            artifacts[name] = load_csv_file(path); artifacts["files_loaded"].append(path)
    events = run_dir / "events.jsonl"
    if events.exists():
        artifacts["events.jsonl"] = read_jsonl_file(events); artifacts["files_loaded"].append(events)
    return artifacts


def extract_seed(pair_dir: Path, artifacts: dict[str, Any]) -> int | None:
    """Extract the seed from run metadata, pair metadata, or finally pair_<seed> directory names."""
    man = artifacts.get("manifest.json") or {}
    if "seed" in man:
        return int(man["seed"])
    pm = pair_dir / "pair_manifest.json"
    if pm.exists():
        data = read_json_file(pm)
        if "seed" in data:
            return int(data["seed"])
    m = re.match(r"pair_(\d+)", pair_dir.name)
    return int(m.group(1)) if m else None


def discover_experiment_runs(results_root: Path) -> list[dict[str, Any]]:
    """Discover AdamW and AdamW+WW-PGD arms for each pair directory and load artifacts."""
    rows: list[dict[str, Any]] = []
    for pair_dir in discover_pair_directories(results_root):
        for optimizer in ["adamw", "adamw_wwpgd"]:
            run_dir, note = select_valid_run_directory(pair_dir / optimizer)
            artifacts = load_run_artifacts(run_dir) if run_dir else {"files_loaded": []}
            rows.append({"pair_id": pair_dir.name, "pair_dir": pair_dir, "optimizer": optimizer, "run_dir": run_dir, "selection_note": note, "seed": extract_seed(pair_dir, artifacts) if run_dir else None, "artifacts": artifacts})
    return rows


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return metrics with notebook-friendly aliases while preserving original columns."""
    out = df.copy()
    aliases = {"tokens_processed": "tokens_seen", "val_loss": "validation_loss", "elapsed_time": "elapsed_seconds", "projection_overhead": "projection_seconds"}
    for src, dst in aliases.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    return out


def terminal_results(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Construct paired terminal validation-loss rows from discovered runs."""
    finals = []
    for r in runs:
        m = normalize_metrics(r.get("artifacts", {}).get("metrics.csv", pd.DataFrame()))
        if m.empty or "validation_loss" not in m:
            continue
        sort_col = "tokens_seen" if "tokens_seen" in m else "step"
        m = m.sort_values(sort_col)
        last = m.dropna(subset=["validation_loss"]).tail(1)
        if last.empty:
            continue
        finals.append({"pair_id": r["pair_id"], "seed": r["seed"], "optimizer": r["optimizer"], "final_validation_loss": float(last["validation_loss"].iloc[0]), "minimum_validation_loss": float(pd.to_numeric(m["validation_loss"], errors="coerce").min())})
    df = pd.DataFrame(finals)
    if df.empty:
        return pd.DataFrame()
    p = df.pivot_table(index=["pair_id", "seed"], columns="optimizer", values=["final_validation_loss", "minimum_validation_loss"], aggfunc="first")
    p.columns = [f"{opt}_{metric}" for metric, opt in p.columns]
    p = p.reset_index()
    if {"adamw_wwpgd_final_validation_loss", "adamw_final_validation_loss"}.issubset(p.columns):
        p["wwpgd_minus_adamw_final_validation_loss"] = p["adamw_wwpgd_final_validation_loss"] - p["adamw_final_validation_loss"]
        p["adamw_minus_wwpgd_improvement"] = -p["wwpgd_minus_adamw_final_validation_loss"]
    return p


def align_curves(curves: list[pd.DataFrame], x_col: str, y_col: str, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate curves to their shared x range without extrapolation."""
    clean = []
    for df in curves:
        if x_col not in df or y_col not in df:
            continue
        d = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(x_col).drop_duplicates(x_col)
        if len(d) >= 2:
            clean.append(d)
    if not clean:
        return np.array([]), np.empty((0, 0))
    lo = max(float(d[x_col].min()) for d in clean); hi = min(float(d[x_col].max()) for d in clean)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([]), np.empty((0, 0))
    grid = np.linspace(lo, hi, points)
    vals = np.vstack([np.interp(grid, d[x_col].to_numpy(float), d[y_col].to_numpy(float)) for d in clean])
    return grid, vals


def paired_curve_differences(pairs: list[tuple[pd.DataFrame, pd.DataFrame]], x_col: str, y_col: str, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Align paired curves and return WW-PGD minus AdamW differences on a common grid."""
    per_pair = []
    grids = []
    for adamw, wwpgd in pairs:
        grid, vals = align_curves([adamw, wwpgd], x_col, y_col, points)
        if vals.shape[0] == 2:
            grids.append(grid); per_pair.append(vals[1] - vals[0])
    if not per_pair:
        return np.array([]), np.empty((0, 0))
    lo = max(g.min() for g in grids); hi = min(g.max() for g in grids)
    if hi <= lo:
        return np.array([]), np.empty((0, 0))
    grid = np.linspace(lo, hi, points)
    vals = np.vstack([np.interp(grid, g, d) for g, d in zip(grids, per_pair)])
    return grid, vals


def completed_runs(root: Path, scientific_only: bool = True) -> list[Path]:
    runs = []
    for p in root.rglob("run_complete.json"):
        run = p.parent
        try:
            man = json.loads((run / "manifest.json").read_text())
        except Exception:
            continue
        if scientific_only and (man.get("smoke_test") is True or man.get("valid_for_science") is not True):
            continue
        runs.append(run)
    return runs


def analyze_results(results_root: Path) -> Path:
    out = unique_dir(results_root / "analysis", "analysis")
    runs=[]; metrics=[]; spectral=[]; projections=[]
    for run in completed_runs(results_root, scientific_only=True):
        man=json.loads((run/"manifest.json").read_text())
        runs.append({"run_dir": str(run), **man})
        m=pd.read_csv(run/"metrics.csv"); m["run_dir"]=str(run); m["optimizer"]=man["optimizer"]; m["seed"]=man["seed"]; m["pair_id"]=man["pair_id"]; metrics.append(m)
        s=pd.read_csv(run/"spectral.csv"); s["run_dir"]=str(run); spectral.append(s)
        p=run/"wwpgd_projection.csv"
        if p.exists(): projections.append(pd.read_csv(p))
    pd.DataFrame(runs).to_csv(out/"runs_manifest.csv", index=False)
    rdf=pd.DataFrame(runs)
    paired = rdf.groupby("pair_id").filter(lambda g: {"adamw", "adamw_wwpgd"}.issubset(set(g["optimizer"]))) if not rdf.empty else rdf
    paired.to_csv(out/"paired_runs_manifest.csv", index=False)
    if not rdf.empty:
        dup = rdf.groupby(["optimizer", "seed"]).size().reset_index(name="run_count")
        rdf.groupby(["optimizer"])["seed"].nunique().reset_index(name="seed_count").merge(dup, how="left").to_csv(out/"seed_counts.csv", index=False)
    else:
        pd.DataFrame(columns=["optimizer", "seed_count"]).to_csv(out/"seed_counts.csv", index=False)
    if metrics:
        mf=pd.concat(metrics); final=mf.sort_values("step").groupby("run_dir").tail(1)
        rows=[]
        for opt,g in mf.groupby("optimizer"):
            for col in g.select_dtypes("number").columns:
                rows.append({"optimizer":opt,"metric":col,**summary(g[col])})
        pd.DataFrame(rows).to_csv(out/"metrics_errorbars.csv", index=False)
        rows=[]
        for opt,g in final.groupby("optimizer"):
            for col in g.select_dtypes("number").columns:
                rows.append({"optimizer":opt,"metric":col,**summary(g[col])})
        pd.DataFrame(rows).to_csv(out/"final_metrics_errorbars.csv", index=False)
        diffs=[]
        for metric in [c for c in final.select_dtypes("number").columns if c not in {"seed"}]:
            pivot=final.pivot_table(index="pair_id", columns="optimizer", values=metric, aggfunc="first")
            if not {"adamw","adamw_wwpgd"}.issubset(pivot.columns):
                continue
            d=(pivot["adamw_wwpgd"]-pivot["adamw"]).dropna()
            t=stats.ttest_1samp(d,0.0) if len(d)>1 else None
            t_stat = float(t.statistic) if t is not None else float("nan")
            t_p = float(t.pvalue) if t is not None else float("nan")
            ci = float(stats.t.ppf(0.975, len(d)-1) * d.sem()) if len(d)>1 else 0.0
            diffs.append({"metric":metric,"mean_paired_difference":float(d.mean()),"std_paired_difference":float(d.std(ddof=1)) if len(d)>1 else 0.0,"standard_error":float(d.sem()) if len(d)>1 else 0.0,"ci95_low":float(d.mean()-ci),"ci95_high":float(d.mean()+ci),"paired_t_statistic":t_stat,"paired_t_pvalue":t_p,"wilcoxon_pvalue":float(stats.wilcoxon(d).pvalue) if len(d)>1 and (d!=0).any() else float("nan"),"effect_size":float(d.mean()/d.std(ddof=1)) if len(d)>1 and d.std(ddof=1) else float("nan"),"complete_pairs":int(len(d))})
        pd.DataFrame(diffs).to_csv(out/"paired_metric_differences.csv", index=False)
        plot_metric_curve(normalize_metrics(mf), "validation_loss", out / "plots" / "validation_loss.png", title="Validation loss by optimizer", ylabel="validation loss")
    if spectral:
        sf=pd.concat(spectral)
        rows=[]
        for keys,g in sf.groupby(["optimizer","step","layer_name"]):
            rows.append({"optimizer":keys[0],"step":keys[1],"layer_name":keys[2],"metric":"alpha",**summary(g["alpha"])})
        pd.DataFrame(rows).to_csv(out/"weightwatcher_layer_errorbars.csv", index=False)
        model_alpha=sf.groupby(["run_dir","optimizer","seed","pair_id","step"])["alpha"].mean().reset_index()
        rows=[]
        for keys,g in model_alpha.groupby(["optimizer","step"]): rows.append({"optimizer":keys[0],"step":keys[1],"metric":"alpha",**summary(g["alpha"])})
        pd.DataFrame(rows).to_csv(out/"weightwatcher_model_errorbars.csv", index=False)
        sf.assign(alpha_distance=(sf["alpha"]-2).abs()).to_csv(out/"weightwatcher_target_distance.csv", index=False)
    if projections:
        pf=pd.concat(projections); rows=[]
        for col in pf.select_dtypes("number").columns: rows.append({"metric":col,**summary(pf[col])})
        pd.DataFrame(rows).to_csv(out/"wwpgd_projection_errorbars.csv", index=False)
    else: (out/"wwpgd_projection_errorbars.csv").write_text("")
    for name in ["scaling_plan.csv","scaling_fit_results.csv"]:
        with (out/name).open("w", newline="") as f: csv.writer(f).writerow(["status","note"]); csv.writer(f).writerow(["not_fit","insufficient smoke design"])
    write_json(out/"analysis_manifest.json", {"source": str(results_root), "completed_runs": len(runs), "limited_power": True})
    return out
