from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from wwgpt.analysis import OPTIMIZER_LABELS, add_generalization_measures, normalize_metrics, normalize_spectral_records, paired_extension_effects, paired_effect_estimates, vocab_size_from_artifacts

BAND_DEFINITIONS = {
    "mean_std": "Mean plus/minus one sample standard deviation across independent seeds at a compatible token budget.",
    "mean_ci95": "Mean plus/minus a two-sided 95% Student-t confidence interval across independent seeds at a compatible token budget.",
}
DEFAULT_FIGURES = (
    "train_loss",
    "validation_loss",
    "final_test_loss",
    "perplexity",
    "generalization_gaps",
    "token_step_progress",
    "per_layer_alpha",
    "alpha_trajectories",
    "correlation_trap_metrics",
    "paired_wwpgd_effects",
)

@dataclass(frozen=True)
class PublicationPlotConfig:
    band: str = "mean_std"
    dpi: int = 300
    file_format: str = "csv"
    curve_points: int = 200
    style: str = "default"

    def __post_init__(self):
        if self.band not in BAND_DEFINITIONS:
            raise ValueError(f"unknown band {self.band!r}; choose one of {sorted(BAND_DEFINITIONS)}")
        if self.dpi < 300:
            raise ValueError("publication PNG dpi must be 300 or greater")
        if self.file_format not in {"csv", "parquet"}:
            raise ValueError("source data format must be csv or parquet")

def optimizer_order(df: pd.DataFrame) -> list[str]:
    if "base_optimizer" in df.columns and "extension" in df.columns:
        bases = sorted(str(x) for x in df["base_optimizer"].dropna().unique())
        order = []
        for base in bases:
            for ext in ("none", "wwpgd"):
                fam = f"{base}_wwpgd" if ext == "wwpgd" else base
                if ((df["base_optimizer"].astype(str) == base) & (df["extension"].astype(str) == ext)).any():
                    order.append(fam)
        return order
    return sorted(str(x) for x in df.get("optimizer_family", pd.Series(dtype=str)).dropna().unique())

def optimizer_label(family: str, base_optimizer: str | None = None, extension: str | None = None) -> str:
    if base_optimizer and extension:
        base_label = OPTIMIZER_LABELS.get(str(base_optimizer), str(base_optimizer))
        return f"{base_label} + WW-PGD" if str(extension) == "wwpgd" else base_label
    return OPTIMIZER_LABELS.get(str(family), str(family))

def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def save_publication_figure(fig, source_df: pd.DataFrame, analysis_dir: Path, figure_name: str, metadata: dict[str, Any], config: PublicationPlotConfig | None = None):
    config = config or PublicationPlotConfig()
    analysis_dir = Path(analysis_dir); analysis_dir.mkdir(parents=True, exist_ok=True)
    png = analysis_dir / f"{figure_name}.png"; pdf = analysis_dir / f"{figure_name}.pdf"
    data = analysis_dir / f"{figure_name}_data.{config.file_format}"; meta = analysis_dir / f"{figure_name}_metadata.json"
    fig.savefig(png, dpi=config.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if config.file_format == "parquet":
        source_df.to_parquet(data, index=False)
    else:
        source_df.to_csv(data, index=False)
    md = {**metadata, "band": config.band, "band_definition": BAND_DEFINITIONS[config.band], "png_dpi": config.dpi, "vector_format": "pdf", "generated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_commit": _commit(), "figure_generation_code_version": "publication_plots_v2", "source_data": str(data)}
    meta.write_text(json.dumps(md, indent=2, sort_keys=True) + "\n")
    return {"png": png, "pdf": pdf, "data": data, "csv": data if config.file_format == "csv" else None, "metadata": meta}

def student_t_ci_by_seed(df: pd.DataFrame, metric: str, group_cols: list[str]):
    from wwgpt.analysis import student_t_summary
    agg = df.groupby(group_cols + ["seed"], dropna=False)[metric].mean().reset_index()
    rows = []
    for keys, g in agg.groupby(group_cols, dropna=False):
        s = student_t_summary(g[metric])
        if not isinstance(keys, tuple): keys = (keys,)
        rows.append(dict(zip(group_cols, keys), mean=s["mean"], ci95=(s["ci95_high"] - s["mean"]), seed_count=s["n"], ci_method="Student-t across independent seeds" if s["n"] > 1 else "unavailable: fewer than two independent seeds"))
    return pd.DataFrame(rows)

def prepare_metric_source(runs: list[dict[str, Any]], metric: str) -> pd.DataFrame:
    rows = []
    for r in runs:
        art = r.get("artifacts") or {}
        m = art.get("metrics") if "metrics" in art else art.get("metrics.csv")
        if m is None and r.get("run_dir"):
            from wwgpt.analysis import load_run_artifacts
            art = load_run_artifacts(Path(r["run_dir"])); m = art["metrics"]
        if m is None or len(m) == 0: continue
        vocab = vocab_size_from_artifacts(art)
        d = add_generalization_measures(normalize_metrics(m), vocab_size=vocab)
        if metric not in d.columns: continue
        x = "tokens_seen" if "tokens_seen" in d.columns else "step"
        keep = [x, "step", metric] if x != "step" and "step" in d.columns else [x, metric]
        for _, row in d[keep].dropna(subset=[metric]).iterrows():
            rows.append({"seed": r.get("seed"), "pair_id": r.get("pair_id"), "optimizer_family": r.get("optimizer_family"), "base_optimizer": r.get("base_optimizer"), "extension": r.get("extension"), "x_axis": x, "x_value": row[x], "step": row.get("step", np.nan), "metric": metric, "value": row[metric]})
    return pd.DataFrame(rows)

def aggregate_compatible_tokens(source: pd.DataFrame, config: PublicationPlotConfig) -> pd.DataFrame:
    rows = []
    if source.empty: return source
    group_cols = ["optimizer_family", "base_optimizer", "extension", "x_axis", "x_value", "metric"]
    for keys, g in source.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(g["value"], errors="coerce").dropna()
        if vals.empty: continue
        mean = vals.mean(); sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        if config.band == "mean_ci95" and len(vals) > 1:
            from scipy import stats
            half = stats.t.ppf(0.975, len(vals)-1) * sd / np.sqrt(len(vals))
        elif config.band == "mean_ci95":
            half = 0.0
        else:
            half = sd
        rows.append(dict(zip(group_cols, keys), mean=mean, band_low=mean-half, band_high=mean+half, seed_count=len(vals), band_definition=BAND_DEFINITIONS[config.band]))
    return pd.DataFrame(rows)

def plot_metric(runs: list[dict[str, Any]], metric: str, out_dir: Path, figure_name: str | None = None, ylabel: str | None = None, config: PublicationPlotConfig | None = None):
    config = config or PublicationPlotConfig()
    import matplotlib.pyplot as plt
    src = prepare_metric_source(runs, metric)
    agg = aggregate_compatible_tokens(src, config)
    fig, ax = plt.subplots(figsize=(7, 4))
    for fam in optimizer_order(src):
        g = src[src["optimizer_family"] == fam]
        label = optimizer_label(fam, g["base_optimizer"].dropna().iloc[0] if g["base_optimizer"].notna().any() else None, g["extension"].dropna().iloc[0] if g["extension"].notna().any() else None)
        for _, s in g.groupby("seed", dropna=False):
            s = s.sort_values("x_value"); ax.plot(s["x_value"], s["value"], alpha=0.22, lw=1)
        a = agg[agg["optimizer_family"] == fam].sort_values("x_value")
        if not a.empty:
            ax.plot(a["x_value"], a["mean"], lw=2.8, label=label)
            ax.fill_between(a["x_value"].to_numpy(float), a["band_low"].to_numpy(float), a["band_high"].to_numpy(float), alpha=0.14)
    ax.set_xlabel("Tokens seen" if src.get("x_axis", pd.Series(["tokens_seen"])).iloc[0] == "tokens_seen" else "Optimizer step")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title()); ax.legend(); ax.grid(alpha=.2)
    combined = pd.concat([src.assign(row_type="seed"), agg.assign(row_type="aggregate")], ignore_index=True, sort=False)
    return save_publication_figure(fig, combined, out_dir, figure_name or metric, {"figure": figure_name or metric, "metric": metric, "individual_seeds": "light lines", "aggregate_trends": "prominent mean lines"}, config)

def build_all_publication_figures(runs: list[dict[str, Any]], out_dir: Path, config: PublicationPlotConfig | None = None) -> dict[str, dict[str, Path]]:
    config = config or PublicationPlotConfig(); outputs = {}
    metric_specs = {
        "train_loss": ("train_loss", "Train loss"),
        "validation_loss": ("validation_loss", "Validation loss"),
        "perplexity": ("val_perplexity", "Validation perplexity"),
        "generalization_gaps": ("generalization_gap", "Validation - train loss"),
        "token_step_progress": ("tokens_per_second", "Tokens per second"),
    }
    for name, (metric, ylabel) in metric_specs.items():
        outputs[name] = plot_metric(runs, metric, out_dir, name, ylabel, config)
    outputs["final_test_loss"] = plot_metric(runs, "test_loss", out_dir, "final_test_loss", "Final test loss", config)
    # Spectral figure families use available WeightWatcher columns without assuming layer names.
    outputs.update(_plot_spectral_families(runs, out_dir, config))
    outputs["paired_wwpgd_effects"] = _plot_paired_effects(runs, out_dir, config)
    return outputs

def _spectral_source(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows=[]
    for r in runs:
        art=r.get("artifacts") or {}; s=art.get("spectral") if "spectral" in art else art.get("spectral.csv")
        if s is None: continue
        d=normalize_spectral_records(s)
        for _, row in d.iterrows():
            rows.append({**row.to_dict(), "seed": r.get("seed"), "pair_id": r.get("pair_id"), "optimizer_family": r.get("optimizer_family"), "base_optimizer": r.get("base_optimizer"), "extension": r.get("extension")})
    return pd.DataFrame(rows)

def _plot_spectral_families(runs, out_dir, config):
    import matplotlib.pyplot as plt
    src=_spectral_source(runs); outs={}
    specs={"per_layer_alpha": ("layer_name", "alpha", "Layer", "Alpha"), "alpha_trajectories": ("tokens_seen", "alpha", "Tokens seen", "Alpha"), "correlation_trap_metrics": ("tokens_seen", "detX_num", "Tokens seen", "detX count")}
    for name,(x,y,xlab,ylab) in specs.items():
        fig, ax=plt.subplots(figsize=(7,4)); plot=src.dropna(subset=[c for c in [x,y] if c in src.columns]) if not src.empty and {x,y}.issubset(src.columns) else pd.DataFrame(columns=[x,y])
        if not plot.empty:
            for fam in optimizer_order(plot):
                g=plot[plot["optimizer_family"]==fam]
                if x == "layer_name":
                    a=g.groupby(x)[y].mean().reset_index(); ax.plot(a[x].astype(str), a[y], marker="o", lw=2.5, label=optimizer_label(fam))
                else:
                    for _, s in g.groupby("seed", dropna=False): ax.plot(s.sort_values(x)[x], s.sort_values(x)[y], alpha=.22, lw=1)
                    a=g.groupby(x)[y].mean().reset_index(); ax.plot(a[x], a[y], lw=2.8, label=optimizer_label(fam))
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.tick_params(axis='x', labelrotation=45); ax.legend(); ax.grid(alpha=.2)
        outs[name]=save_publication_figure(fig, plot, out_dir, name, {"figure": name, "band_definition": BAND_DEFINITIONS[config.band]}, config)
    return outs

def _plot_paired_effects(runs, out_dir, config):
    import matplotlib.pyplot as plt
    finals=[]
    for metric in ["validation_loss", "test_loss", "val_perplexity", "generalization_gap"]:
        src=prepare_metric_source(runs, metric)
        if src.empty: continue
        idx=src.sort_values("x_value").groupby(["pair_id","seed","base_optimizer","extension"], dropna=False).tail(1)
        p=paired_extension_effects(idx.rename(columns={"value": metric}), metric)
        if not p.empty: finals.append(p.assign(metric=metric))
    data=pd.concat(finals, ignore_index=True, sort=False) if finals else pd.DataFrame()
    fig, ax=plt.subplots(figsize=(7,4))
    effects=[]
    if not data.empty:
        for metric,g in data.groupby("metric"):
            est=paired_effect_estimates(g, metric)
            effects.append(est)
        e=pd.concat(effects, ignore_index=True, sort=False) if effects else pd.DataFrame()
        if not e.empty:
            labels=[f"{optimizer_label(b)}\n{m}" for b,m in zip(e["base_optimizer"], e["metric"])]
            ax.bar(range(len(e)), e["paired_effect_mean"]); ax.axhline(0,color="black",lw=1); ax.set_xticks(range(len(e)), labels, rotation=45, ha="right")
    ax.set_ylabel("WW-PGD minus baseline (paired final metric)"); ax.grid(axis="y", alpha=.2)
    return save_publication_figure(fig, data if not data.empty else pd.DataFrame(), out_dir, "paired_wwpgd_effects", {"figure":"paired_wwpgd_effects", "paired_by":"seed and base_optimizer"}, config)
