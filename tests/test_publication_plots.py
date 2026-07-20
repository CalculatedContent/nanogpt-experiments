from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wwgpt.analysis import load_run_artifacts
from wwgpt.publication_plots import (
    BAND_DEFINITIONS,
    PublicationPlotConfig,
    aggregate_compatible_tokens,
    build_all_publication_figures,
    prepare_metric_source,
)


def _run(root: Path, base: str, ext: str, seed: int, loss_offset: float):
    fam = f"{base}_wwpgd" if ext == "wwpgd" else base
    rd = root / f"seed_{seed}_{fam}"
    rd.mkdir(parents=True)
    man = {"scientific_schema_version": 3, "valid_for_science": True, "seed": seed, "pair_id": f"pair_{seed}", "base_optimizer": base, "extension": ext, "optimizer": fam, "parameter_report": {"vocab_size": 32}}
    (rd / "manifest.json").write_text(json.dumps(man))
    pd.DataFrame({
        "step": [1, 2, 3],
        "tokens_processed": [100, 200, 300],
        "train_loss": [3.0, 2.5, 2.0 + loss_offset],
        "val_loss": [3.2, 2.7, 2.2 + loss_offset],
        "test_loss": [3.3, 2.8, 2.3 + loss_offset],
        "tokens_per_second": [1000, 1100, 1200],
    }).to_csv(rd / "metrics.csv", index=False)
    pd.DataFrame({
        "layer_name": ["blocks.0.attn.c_attn", "blocks.0.mlp.0"],
        "tokens_seen": [100, 300],
        "alpha": [2.0 + loss_offset, 2.2 + loss_offset],
        "detX_num": [3, 4],
        "spectral_estimator": ["weightwatcher", "weightwatcher"],
    }).to_csv(rd / "spectral.csv", index=False)
    (rd / "run_complete.json").write_text(json.dumps({"step": 3}))
    return {"run_dir": rd, "artifacts": load_run_artifacts(rd), "seed": seed, "pair_id": f"pair_{seed}", "optimizer_family": fam, "base_optimizer": base, "extension": ext}


def _runs(tmp_path: Path):
    rows = []
    for seed, seed_offset in [(1, 0.0), (2, 0.1)]:
        for base in ["adamw", "muon", "stableadamw"]:
            rows.append(_run(tmp_path, base, "none", seed, seed_offset))
            rows.append(_run(tmp_path, base, "wwpgd", seed, seed_offset - 0.2))
    return rows


def test_publication_plot_exports_png_pdf_source_data_and_metadata(tmp_path: Path):
    out = tmp_path / "figures"
    outputs = build_all_publication_figures(_runs(tmp_path / "runs"), out, PublicationPlotConfig(dpi=300))
    expected = {"train_loss", "validation_loss", "final_test_loss", "perplexity", "generalization_gaps", "token_step_progress", "per_layer_alpha", "alpha_trajectories", "correlation_trap_metrics", "paired_wwpgd_effects"}
    assert expected.issubset(outputs)
    for name in expected:
        paths = outputs[name]
        assert paths["png"].exists() and paths["png"].stat().st_size > 0
        assert paths["pdf"].exists() and paths["pdf"].stat().st_size > 0
        assert paths["data"].exists() and paths["data"].stat().st_size > 0
        metadata = json.loads(paths["metadata"].read_text())
        assert metadata["png_dpi"] >= 300
        assert metadata["vector_format"] == "pdf"
        assert metadata["band_definition"] in BAND_DEFINITIONS.values()


def test_source_data_keeps_individual_seeds_and_aggregates_only_matching_tokens(tmp_path: Path):
    runs = _runs(tmp_path / "runs")
    source = prepare_metric_source(runs, "validation_loss")
    aggregate = aggregate_compatible_tokens(source, PublicationPlotConfig(band="mean_std"))
    assert {"seed", "optimizer_family", "x_value", "value"}.issubset(source.columns)
    assert aggregate["band_definition"].eq(BAND_DEFINITIONS["mean_std"]).all()
    # No smoothing or interpolation across incompatible budgets: aggregates exist only at plotted x values.
    assert set(aggregate["x_value"]).issubset(set(source["x_value"]))
    assert aggregate.groupby(["optimizer_family", "x_value"])["seed_count"].max().min() == 2


def test_spectral_exports_include_seed_rows_and_aggregate_bands(tmp_path: Path):
    out = tmp_path / "figures"
    outputs = build_all_publication_figures(_runs(tmp_path / "runs"), out, PublicationPlotConfig(band="mean_std"))
    alpha_data = pd.read_csv(outputs["alpha_trajectories"]["data"])
    assert {"seed", "row_type", "band_low", "band_high", "band_definition"}.issubset(alpha_data.columns)
    assert {"seed", "aggregate"}.issubset(set(alpha_data["row_type"].dropna()))
    aggregates = alpha_data[alpha_data["row_type"].eq("aggregate")]
    assert aggregates["band_definition"].eq(BAND_DEFINITIONS["mean_std"]).all()
    assert aggregates.groupby(["optimizer_family", "tokens_seen"])["seed_count"].max().min() == 2


def test_paired_effect_exports_all_available_base_optimizers_without_notebook_hardcoding(tmp_path: Path):
    out = tmp_path / "figures"
    outputs = build_all_publication_figures(_runs(tmp_path / "runs"), out, PublicationPlotConfig())
    paired = pd.read_csv(outputs["paired_wwpgd_effects"]["data"])
    plotted = paired[paired["row_type"].eq("plotted_effect")]
    assert set(plotted["base_optimizer"].dropna()) == {"adamw", "muon", "stableadamw"}
    assert set(plotted["metric"].dropna()).issuperset({"validation_loss", "test_loss", "val_perplexity", "generalization_gap"})
