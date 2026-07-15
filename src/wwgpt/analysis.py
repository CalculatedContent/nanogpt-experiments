from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from wwgpt.utils import unique_dir, write_json


def summary(s: pd.Series) -> dict[str, float | int]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(len(s)); sd = float(s.std(ddof=1)) if n > 1 else 0.0; se = sd / (n**0.5) if n else 0.0
    ci = float(stats.t.ppf(0.975, n - 1) * se) if n > 1 else 0.0
    return {"n": n, "mean": float(s.mean()) if n else float("nan"), "sample_std": sd, "standard_error": se, "ci95_low": float(s.mean()-ci) if n else float("nan"), "ci95_high": float(s.mean()+ci) if n else float("nan"), "median": float(s.median()) if n else float("nan"), "iqr": float(s.quantile(.75)-s.quantile(.25)) if n else float("nan"), "min": float(s.min()) if n else float("nan"), "max": float(s.max()) if n else float("nan")}


def completed_runs(root: Path) -> list[Path]:
    return [p.parent for p in root.rglob("run_complete.json")]


def analyze_results(results_root: Path) -> Path:
    out = unique_dir(results_root / "analysis", "analysis")
    runs=[]; metrics=[]; spectral=[]; projections=[]
    for run in completed_runs(results_root):
        man=json.loads((run/"manifest.json").read_text())
        runs.append({"run_dir": str(run), **man})
        m=pd.read_csv(run/"metrics.csv"); m["run_dir"]=str(run); m["optimizer"]=man["optimizer"]; m["seed"]=man["seed"]; m["pair_id"]=man["pair_id"]; metrics.append(m)
        s=pd.read_csv(run/"spectral.csv"); s["run_dir"]=str(run); spectral.append(s)
        p=run/"wwpgd_projection.csv"
        if p.exists(): projections.append(pd.read_csv(p))
    pd.DataFrame(runs).to_csv(out/"runs_manifest.csv", index=False)
    rdf=pd.DataFrame(runs)
    rdf.to_csv(out/"paired_runs_manifest.csv", index=False)
    rdf.groupby(["optimizer"])["seed"].nunique().reset_index(name="seed_count").to_csv(out/"seed_counts.csv", index=False)
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
        pivot=final.pivot_table(index="seed", columns="optimizer", values="val_loss", aggfunc="first")
        if {"adamw","adamw_wwpgd"}.issubset(pivot.columns):
            d=pivot["adamw_wwpgd"]-pivot["adamw"]
            t=stats.ttest_1samp(d,0.0) if len(d)>1 else None
            t_stat = float(t.statistic) if t is not None else float("nan")
            t_p = float(t.pvalue) if t is not None else float("nan")
            diffs.append({"metric":"val_loss","mean_paired_difference":float(d.mean()),"std_paired_difference":float(d.std(ddof=1)) if len(d)>1 else 0.0,"standard_error":float(d.sem()) if len(d)>1 else 0.0,"paired_t_statistic":t_stat,"paired_t_pvalue":t_p,"wilcoxon_pvalue":float(stats.wilcoxon(d).pvalue) if len(d)>1 and (d!=0).any() else float("nan"),"effect_size":float(d.mean()/d.std(ddof=1)) if len(d)>1 and d.std(ddof=1) else float("nan"),"complete_pairs":int(len(d))})
        pd.DataFrame(diffs).to_csv(out/"paired_metric_differences.csv", index=False)
        plt.figure();
        for opt,g in mf.groupby("optimizer"): plt.plot(g.groupby("step")["val_loss"].mean(), label=opt)
        plt.legend(); plt.ylabel("validation loss"); plt.xlabel("step"); (out/"plots").mkdir(); plt.savefig(out/"plots"/"validation_loss.png"); plt.close()
    if spectral:
        sf=pd.concat(spectral); sf.to_csv(out/"weightwatcher_layer_errorbars.csv", index=False)
        sf.groupby(["optimizer","step"])["alpha"].mean().reset_index().to_csv(out/"weightwatcher_model_errorbars.csv", index=False)
        sf.assign(alpha_distance=(sf["alpha"]-2).abs()).to_csv(out/"weightwatcher_target_distance.csv", index=False)
    if projections: pd.concat(projections).to_csv(out/"wwpgd_projection_errorbars.csv", index=False)
    else: (out/"wwpgd_projection_errorbars.csv").write_text("")
    for name in ["scaling_plan.csv","scaling_fit_results.csv"]:
        with (out/name).open("w", newline="") as f: csv.writer(f).writerow(["status","note"]); csv.writer(f).writerow(["not_fit","insufficient smoke design"])
    write_json(out/"analysis_manifest.json", {"source": str(results_root), "completed_runs": len(runs), "limited_power": True})
    return out
