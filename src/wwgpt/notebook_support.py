"""Schema-v3 notebook I/O and analysis helpers.

This module never mutates experiment run directories.  Notebook outputs are
written only beneath ``NotebookParameters.output_root``.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRUTHY = {"1", "true", "yes", "on", "y"}
FALSY = {"0", "false", "no", "off", "n"}
ACCURACY_METRICS = {
    "train_accuracy",
    "train_top1_accuracy",
    "train_next_token_accuracy",
    "validation_accuracy",
    "validation_top1_accuracy",
    "validation_next_token_accuracy",
    "test_accuracy",
    "test_top1_accuracy",
    "test_next_token_accuracy",
}


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUTHY:
        return True
    if text in FALSY:
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


@dataclass(frozen=True)
class NotebookParameters:
    results_root: Path
    output_root: Path
    analysis_plan: Path | None = None
    profile: str = ""
    level: int | None = None
    token_multiplier: int | None = None
    base_optimizer: str = "adamw"
    strict: bool = False
    run_analysis: bool = False
    reuse_existing_analysis: bool = True
    figure_format: str = "png"
    random_seed: int = 1729

    def summary(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


def _choice(injected: Any, env: str, default: Any) -> Any:
    return injected if injected is not None and injected != "" else os.getenv(env, default)


def resolve_notebook_parameters(parameters: dict[str, Any] | None = None) -> NotebookParameters:
    """Resolve nonempty Papermill values, then environment, then defaults."""
    supplied = parameters or {}
    results_root = _choice(supplied.get("RESULTS_ROOT"), "WWGPT_RESULTS_ROOT", "")
    if not results_root:
        raise ValueError("RESULTS_ROOT or WWGPT_RESULTS_ROOT is required")
    output_root = _choice(
        supplied.get("OUTPUT_ROOT"),
        "WWGPT_NOTEBOOK_OUTPUT_DIR",
        "notebook-output",
    )
    plan = _choice(supplied.get("ANALYSIS_PLAN"), "WWGPT_ANALYSIS_PLAN", "")
    level = _choice(supplied.get("LEVEL"), "WWGPT_LEVEL", None)
    multiplier = _choice(
        supplied.get("TOKEN_MULTIPLIER"),
        "WWGPT_TOKEN_MULTIPLIER",
        None,
    )
    return NotebookParameters(
        results_root=Path(results_root).expanduser().resolve(),
        output_root=Path(output_root).expanduser().resolve(),
        analysis_plan=Path(plan).expanduser().resolve() if plan else None,
        profile=str(_choice(supplied.get("PROFILE"), "WWGPT_PROFILE", "")),
        level=int(level) if level not in (None, "") else None,
        token_multiplier=int(multiplier) if multiplier not in (None, "") else None,
        base_optimizer=normalize_arm(
            str(_choice(supplied.get("BASE_OPTIMIZER"), "WWGPT_BASE_OPTIMIZER", "adamw"))
        ).removesuffix("_wwpgd"),
        strict=parse_bool(_choice(supplied.get("STRICT"), "WWGPT_NOTEBOOK_STRICT", False)),
        run_analysis=parse_bool(
            _choice(supplied.get("RUN_ANALYSIS"), "WWGPT_RUN_ANALYSIS", False)
        ),
        reuse_existing_analysis=parse_bool(
            _choice(
                supplied.get("REUSE_EXISTING_ANALYSIS"),
                "WWGPT_REUSE_ANALYSIS",
                True,
            ),
            True,
        ),
        figure_format=str(
            _choice(supplied.get("FIGURE_FORMAT"), "WWGPT_FIGURE_FORMAT", "png")
        ).lstrip("."),
        random_seed=int(
            _choice(supplied.get("RANDOM_SEED"), "WWGPT_NOTEBOOK_RANDOM_SEED", 1729)
        ),
    )


def validate_paths(params: NotebookParameters) -> None:
    if not params.results_root.is_dir():
        raise FileNotFoundError(f"results root does not exist: {params.results_root}")
    if params.analysis_plan and not params.analysis_plan.is_file():
        raise FileNotFoundError(f"analysis plan does not exist: {params.analysis_plan}")
    params.output_root.mkdir(parents=True, exist_ok=True)
    for name in ("tables", "figures", "logs", "executed"):
        (params.output_root / name).mkdir(exist_ok=True)


def normalize_arm(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    return normalized.replace("stable_adamw", "stableadamw").replace(
        "_wwpgd_reference", "_wwpgd"
    )


def discover_layouts(root: Path) -> list[Path]:
    layouts: set[Path] = set()
    for manifest_name in ("pair_manifest.json", "trial_manifest.json"):
        layouts.update(path.parent for path in Path(root).rglob(manifest_name))
    return sorted(layouts)


def _latest_complete(arm_dir: Path) -> Path | None:
    if not arm_dir.is_dir():
        return None
    candidates = [arm_dir] + sorted(
        (path for path in arm_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    return next(
        (
            path
            for path in candidates
            if (path / "run_complete.json").is_file()
            and (path / "manifest.json").is_file()
        ),
        None,
    )


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def discover_completed_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for layout in discover_layouts(Path(root)):
        layout_manifest_path = layout / (
            "pair_manifest.json" if (layout / "pair_manifest.json").is_file() else "trial_manifest.json"
        )
        layout_manifest = load_json(layout_manifest_path) or {}
        layout_seed = layout_manifest.get("seed", layout_manifest.get("scientific_seed"))
        for arm_dir in sorted(path for path in layout.iterdir() if path.is_dir()):
            if arm_dir.name in {"initial_state", "analysis", "figures"}:
                continue
            run = _latest_complete(arm_dir)
            if run is None:
                continue
            manifest = load_json(run / "manifest.json") or {}
            arm = normalize_arm(
                str(manifest.get("arm_name") or manifest.get("optimizer") or arm_dir.name)
            )
            rows.append(
                {
                    "layout": layout.name,
                    "layout_path": layout,
                    "layout_manifest_path": layout_manifest_path,
                    "seed": manifest.get("seed", layout_seed),
                    "arm": arm,
                    "base_optimizer": normalize_arm(
                        str(manifest.get("base_optimizer") or arm.removesuffix("_wwpgd"))
                    ),
                    "extension": str(
                        manifest.get("extension")
                        or ("wwpgd" if arm.endswith("_wwpgd") else "none")
                    ),
                    "run_dir": run,
                    "manifest": manifest,
                    "level": manifest.get("level", layout_manifest.get("level")),
                    "token_multiplier": manifest.get(
                        "token_multiplier", layout_manifest.get("token_multiplier")
                    ),
                }
            )
    return pd.DataFrame(rows)


def filter_runs(
    runs: pd.DataFrame,
    *,
    level: int | None = None,
    token_multiplier: int | None = None,
) -> pd.DataFrame:
    selected = runs.copy()
    if level is not None and "level" in selected:
        selected = selected[pd.to_numeric(selected["level"], errors="coerce").eq(level)]
    if token_multiplier is not None and "token_multiplier" in selected:
        selected = selected[
            pd.to_numeric(selected["token_multiplier"], errors="coerce").eq(token_multiplier)
        ]
    return selected.reset_index(drop=True)


def pair_arms(runs: pd.DataFrame, base_optimizer: str = "adamw") -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame(
            columns=["layout", "seed", "baseline_run", "wwpgd_run"]
        )
    base = normalize_arm(base_optimizer).removesuffix("_wwpgd")
    rows: list[dict[str, Any]] = []
    for (layout, seed), group in runs.groupby(["layout", "seed"], dropna=False):
        by_arm = {normalize_arm(arm): path for arm, path in zip(group.arm, group.run_dir)}
        wwpgd_arm = f"{base}_wwpgd"
        if base in by_arm and wwpgd_arm in by_arm:
            rows.append(
                {
                    "layout": layout,
                    "seed": seed,
                    "baseline_arm": base,
                    "wwpgd_arm": wwpgd_arm,
                    "baseline_run": by_arm[base],
                    "wwpgd_run": by_arm[wwpgd_arm],
                }
            )
    return pd.DataFrame(rows)


def normalize_metrics(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    normalized = frame.copy()
    aliases = {
        "tokens_processed": "tokens_seen",
        "val_loss": "validation_loss",
        "val_perplexity": "validation_perplexity",
        "val_top1_accuracy": "validation_top1_accuracy",
        "train_top1_accuracy": "train_next_token_accuracy",
        "validation_top1_accuracy": "validation_next_token_accuracy",
        "test_top1_accuracy": "test_next_token_accuracy",
        "elapsed_time": "elapsed_seconds",
    }
    for source, destination in aliases.items():
        if source in normalized and destination not in normalized:
            normalized[destination] = normalized[source]
    if "validation_top1_accuracy" not in normalized and "validation_next_token_accuracy" in normalized:
        normalized["validation_top1_accuracy"] = normalized["validation_next_token_accuracy"]
    if "test_top1_accuracy" not in normalized and "test_next_token_accuracy" in normalized:
        normalized["test_top1_accuracy"] = normalized["test_next_token_accuracy"]
    return normalized


def normalize_selected_checkpoint_metrics(frame: pd.DataFrame | None) -> pd.DataFrame:
    normalized = normalize_metrics(frame)
    if normalized.empty:
        return normalized
    aliases = {
        "selected_step": "selected_checkpoint_step",
        "validation_accuracy": "validation_top1_accuracy",
        "test_accuracy": "test_top1_accuracy",
        "train_accuracy": "train_top1_accuracy",
        "train_test_gap": "train_test_loss_gap",
        "train_validation_gap": "train_validation_loss_gap",
    }
    for source, destination in aliases.items():
        if source in normalized and destination not in normalized:
            normalized[destination] = normalized[source]
    for split in ("train", "validation", "test"):
        top1 = f"{split}_top1_accuracy"
        named = f"{split}_next_token_accuracy"
        if top1 in normalized and named not in normalized:
            normalized[named] = normalized[top1]
    if {"test_perplexity", "train_perplexity"}.issubset(normalized.columns):
        normalized["train_test_perplexity_gap"] = (
            pd.to_numeric(normalized["test_perplexity"], errors="coerce")
            - pd.to_numeric(normalized["train_perplexity"], errors="coerce")
        )
    if {"validation_perplexity", "train_perplexity"}.issubset(normalized.columns):
        normalized["train_validation_perplexity_gap"] = (
            pd.to_numeric(normalized["validation_perplexity"], errors="coerce")
            - pd.to_numeric(normalized["train_perplexity"], errors="coerce")
        )
    return normalized


def load_metrics(run: Path) -> pd.DataFrame:
    return normalize_metrics(load_csv(Path(run) / "metrics.csv"))


def load_alpha_measurements(run: Path) -> pd.DataFrame | None:
    return load_csv(Path(run) / "alpha_measurements.csv")


def load_weightwatcher_aggregates(run: Path) -> pd.DataFrame | None:
    return load_csv(Path(run) / "weightwatcher_aggregates.csv")


def load_wwpgd_internal_diagnostics(run: Path) -> pd.DataFrame | None:
    return load_csv(Path(run) / "wwpgd_internal_diagnostics.csv")


def load_wwpgd_artifact(run: Path, name: str) -> pd.DataFrame | None:
    return load_csv(Path(run) / name)


def load_selected_checkpoint_metrics(run: Path) -> pd.DataFrame | None:
    record = load_json(Path(run) / "selected_checkpoint_metrics.json")
    if record is not None:
        return normalize_selected_checkpoint_metrics(pd.json_normalize(record))
    return normalize_selected_checkpoint_metrics(
        load_csv(Path(run) / "selected_checkpoint_metrics.csv")
    )


def filter_scientific_alpha(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    selected = frame.copy()

    def truth(column: str) -> pd.Series:
        if column not in selected:
            return pd.Series(False, index=selected.index)
        return selected[column].astype(str).str.lower().isin(TRUTHY)

    valid = (
        truth("projected")
        & truth("included_in_projected_alpha_summary")
        & truth("valid_for_science")
    )
    alpha = pd.to_numeric(selected.get("alpha"), errors="coerce")
    xmin = pd.to_numeric(selected.get("xmin"), errors="coerce")
    reason = selected.get(
        "validity_exclusion_reason", pd.Series("", index=selected.index)
    ).fillna("").astype(str).str.strip()
    return selected.loc[
        valid & np.isfinite(alpha) & np.isfinite(xmin) & (xmin > 0) & reason.eq("")
    ].copy()


def paired_effect(metric: str, baseline: float, wwpgd: float) -> float:
    """Return WWPGD minus baseline.

    Negative is favorable for losses and perplexities.  Positive is favorable
    for next-token accuracy metrics.
    """
    return float(wwpgd) - float(baseline)


def effect_direction(metric: str) -> str:
    return "positive_is_better" if metric in ACCURACY_METRICS or "accuracy" in metric else "negative_is_better"


def package_provenance() -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    for distribution, module in (("weightwatcher", "weightwatcher"), ("ww-pgd", "ww_pgd")):
        spec = importlib.util.find_spec(module)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        record: dict[str, Any] = {
            "version": version,
            "module_path": spec.origin if spec else None,
        }
        try:
            direct_url = importlib.metadata.distribution(distribution).read_text("direct_url.json")
            if direct_url:
                record["direct_url"] = json.loads(direct_url)
        except importlib.metadata.PackageNotFoundError:
            pass
        provenance[module] = record
    return provenance


def required_probe_tokens(manifest: dict[str, Any]) -> int | None:
    train = manifest.get("optimizer_hyperparameters") or manifest.get("train") or {}
    model = manifest.get("model_config") or {}
    try:
        eval_batches = int(train.get("eval_batches", manifest.get("eval_batches")))
        batch_size = int(train.get("batch_size", manifest.get("batch_size")))
        block_size = int(model.get("block_size", manifest.get("block_size")))
    except (TypeError, ValueError):
        return None
    return eval_batches * batch_size * block_size + 1


def split_capacity(run: Path) -> dict[str, Any]:
    manifest = load_json(Path(run) / "manifest.json") or {}
    data_manifest = load_json(Path(run) / "data_manifest.json") or {}
    required = required_probe_tokens(manifest)
    validation_tokens = data_manifest.get("validation_tokens")
    test_tokens = data_manifest.get("test_tokens")
    return {
        "required_probe_tokens": required,
        "validation_tokens": validation_tokens,
        "test_tokens": test_tokens,
        "validation_capacity_ok": required is not None
        and validation_tokens is not None
        and int(validation_tokens) >= required,
        "test_capacity_ok": required is not None
        and test_tokens is not None
        and int(test_tokens) >= required,
    }


def write_table(params: NotebookParameters, name: str, frame: pd.DataFrame) -> Path:
    path = params.output_root / "tables" / name
    frame.to_csv(path, index=False)
    return path


def write_figure(params: NotebookParameters, name: str, figure) -> Path:
    path = params.output_root / "figures" / f"{name}.{params.figure_format}"
    figure.savefig(path, bbox_inches="tight")
    return path


def optional_artifacts(run: Path, names: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "present": (Path(run) / name).is_file(),
                "warning": "" if (Path(run) / name).is_file() else "optional artifact missing",
            }
            for name in names
        ]
    )
