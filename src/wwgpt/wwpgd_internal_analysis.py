"""Descriptive analysis for diagnostics emitted by the stock WW-PGD calculation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _numeric(frame: pd.DataFrame, names: list[str]) -> None:
    for name in names:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")


def generate_wwpgd_internal_analysis(run_dir: Path, output_dir: Path | None = None) -> dict[str, Path]:
    """Generate tables and figures without reconstructing spectra or invoking WeightWatcher."""
    run_dir = Path(run_dir)
    out = Path(output_dir or run_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = run_dir / "wwpgd_internal_diagnostics.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    data = pd.read_csv(source)
    numeric = ["optimizer_step", "k_pl", "k_detx", "k_star", "selected_tail_size",
               "trace_log_retraction_residual", "trace_log_retraction_absolute_error",
               "trace_log_final_blend_delta_from_original", "cayley_low_clip_count",
               "cayley_high_clip_count", "candidate_relative_frobenius_change"]
    _numeric(data, numeric)
    keys = [c for c in ("seed", "layer_name", "matrix_type", "block") if c in data]
    measures = [c for c in numeric if c in data and c != "optimizer_step"]
    by_layer = data.groupby(keys, dropna=False)[measures].agg(["count", "mean", "max"]).reset_index()
    step_keys = [c for c in ("seed", "optimizer_step") if c in data]
    by_step = data.groupby(step_keys, dropna=False)[measures].mean().reset_index()
    trace = data[[c for c in ("seed", "optimizer_step", "layer_name", "trace_log_retraction_residual",
                              "trace_log_retraction_tolerance", "trace_log_retraction_pass",
                              "trace_log_final_blend_delta_from_original") if c in data]].copy()
    midpoint = data[[c for c in ("seed", "optimizer_step", "layer_name", "k_pl", "k_detx",
                                 "k_star", "selected_tail_size") if c in data]].copy()
    clipping = data.groupby(step_keys, dropna=False)[[c for c in ("cayley_low_clip_count", "cayley_high_clip_count") if c in data]].sum().reset_index()
    tables = {"wwpgd_internal_diagnostics_by_layer.csv": by_layer,
              "wwpgd_internal_diagnostics_by_step.csv": by_step,
              "wwpgd_trace_log_audit.csv": trace,
              "wwpgd_tail_midpoint_trajectory.csv": midpoint,
              "wwpgd_cayley_clipping_summary.csv": clipping}
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        paths[name] = out / name; frame.to_csv(paths[name], index=False)

    plots = {
        "wwpgd_midpoint_by_step.png": (midpoint, ["k_pl", "k_detx", "k_star"]),
        "wwpgd_selected_tail_size_by_step.png": (midpoint, ["selected_tail_size"]),
        "wwpgd_trace_log_retraction_error.png": (trace, ["trace_log_retraction_residual", "trace_log_final_blend_delta_from_original"]),
        "wwpgd_cayley_clipping_by_step.png": (clipping, ["cayley_low_clip_count", "cayley_high_clip_count"]),
        "wwpgd_candidate_relative_change_by_step.png": (data, ["candidate_relative_frobenius_change"]),
    }
    for name, (frame, columns) in plots.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        groups = frame.groupby([c for c in ("seed", "layer_name") if c in frame], dropna=False) if any(c in frame for c in ("seed", "layer_name")) else [("all", frame)]
        for group, part in groups:
            for column in columns:
                if column in part:
                    ax.plot(part.get("optimizer_step", part.index), part[column], marker=".", label=f"{group}:{column}")
        ax.set_xlabel("optimizer step"); ax.set_ylabel("diagnostic value"); ax.legend(fontsize="x-small")
        fig.tight_layout(); paths[name] = out / name; fig.savefig(paths[name], dpi=150); plt.close(fig)
    return paths
