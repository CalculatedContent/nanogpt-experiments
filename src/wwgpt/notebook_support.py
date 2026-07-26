"""Schema-v3 notebook I/O.  This module never mutates experiment directories."""
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


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
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
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def _choice(injected: Any, env: str, default: Any) -> Any:
    return injected if injected is not None and injected != "" else os.getenv(env, default)


def resolve_notebook_parameters(parameters: dict[str, Any] | None = None) -> NotebookParameters:
    """Resolve nonempty Papermill values, then environment, then defaults."""
    p = parameters or {}
    rr = _choice(p.get("RESULTS_ROOT"), "WWGPT_RESULTS_ROOT", "")
    if not rr:
        raise ValueError("RESULTS_ROOT or WWGPT_RESULTS_ROOT is required")
    out = _choice(p.get("OUTPUT_ROOT"), "WWGPT_NOTEBOOK_OUTPUT_DIR", "notebook-output")
    plan = _choice(p.get("ANALYSIS_PLAN"), "WWGPT_ANALYSIS_PLAN", "")
    level = _choice(p.get("LEVEL"), "WWGPT_LEVEL", None)
    mult = _choice(p.get("TOKEN_MULTIPLIER"), "WWGPT_TOKEN_MULTIPLIER", None)
    return NotebookParameters(
        results_root=Path(rr).expanduser().resolve(), output_root=Path(out).expanduser().resolve(),
        analysis_plan=Path(plan).expanduser().resolve() if plan else None,
        profile=str(_choice(p.get("PROFILE"), "WWGPT_PROFILE", "")),
        level=int(level) if level not in (None, "") else None,
        token_multiplier=int(mult) if mult not in (None, "") else None,
        base_optimizer=normalize_arm(str(_choice(p.get("BASE_OPTIMIZER"), "WWGPT_BASE_OPTIMIZER", "adamw"))).replace("_wwpgd", ""),
        strict=parse_bool(_choice(p.get("STRICT"), "WWGPT_NOTEBOOK_STRICT", False)),
        run_analysis=parse_bool(_choice(p.get("RUN_ANALYSIS"), "WWGPT_RUN_ANALYSIS", False)),
        reuse_existing_analysis=parse_bool(_choice(p.get("REUSE_EXISTING_ANALYSIS"), "WWGPT_REUSE_ANALYSIS", True), True),
        figure_format=str(_choice(p.get("FIGURE_FORMAT"), "WWGPT_FIGURE_FORMAT", "png")).lstrip("."),
        random_seed=int(p.get("RANDOM_SEED", 1729)),
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
    n = name.strip().lower().replace("-", "_")
    n = n.replace("stable_adamw", "stableadamw").replace("_wwpgd_reference", "_wwpgd")
    return n


def discover_layouts(root: Path) -> list[Path]:
    return sorted({p.parent for pattern in ("pair_*/pair_manifest.json", "trial_*/trial_manifest.json") for p in root.rglob(pattern)})


def _latest_complete(arm_dir: Path) -> Path | None:
    candidates = [arm_dir] + sorted((p for p in arm_dir.iterdir() if p.is_dir()), reverse=True) if arm_dir.is_dir() else []
    return next((p for p in candidates if (p / "run_complete.json").is_file()), None)


def discover_completed_runs(root: Path) -> pd.DataFrame:
    rows = []
    for layout in discover_layouts(root):
        mf = layout / ("pair_manifest.json" if layout.name.startswith("pair_") else "trial_manifest.json")
        manifest = json.loads(mf.read_text())
        seed = manifest.get("seed", manifest.get("scientific_seed"))
        for arm_dir in (p for p in layout.iterdir() if p.is_dir()):
            run = _latest_complete(arm_dir)
            if run:
                rm = load_json(run / "manifest.json") or {}
                rows.append({"layout": layout.name, "layout_path": layout, "seed": rm.get("seed", seed),
                             "arm": normalize_arm(arm_dir.name), "run_dir": run, "manifest": rm,
                             "level": rm.get("level"), "token_multiplier": rm.get("token_multiplier")})
    return pd.DataFrame(rows)


def pair_arms(runs: pd.DataFrame, base_optimizer: str = "adamw") -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame(columns=["layout", "seed", "baseline_run", "wwpgd_run"])
    base = normalize_arm(base_optimizer).replace("_wwpgd", "")
    rows = []
    for (layout, seed), group in runs.groupby(["layout", "seed"], dropna=False):
        by_arm = {a: p for a, p in zip(group.arm, group.run_dir)}
        if base in by_arm and f"{base}_wwpgd" in by_arm:
            rows.append({"layout": layout, "seed": seed, "baseline_arm": base, "wwpgd_arm": f"{base}_wwpgd",
                         "baseline_run": by_arm[base], "wwpgd_run": by_arm[f"{base}_wwpgd"]})
    return pd.DataFrame(rows)


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def load_metrics(run: Path): return load_csv(run / "metrics.csv")
def load_alpha_measurements(run: Path): return load_csv(run / "alpha_measurements.csv")
def load_weightwatcher_aggregates(run: Path): return load_csv(run / "weightwatcher_aggregates.csv")
def load_wwpgd_internal_diagnostics(run: Path): return load_csv(run / "wwpgd_internal_diagnostics.csv")
def load_wwpgd_artifact(run: Path, name: str): return load_csv(run / name)


def load_selected_checkpoint_metrics(run: Path) -> pd.DataFrame | None:
    js = load_json(run / "selected_checkpoint_metrics.json")
    if js is not None:
        return pd.json_normalize(js)
    return load_csv(run / "selected_checkpoint_metrics.csv")


def filter_scientific_alpha(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None: return pd.DataFrame()
    x = frame.copy()
    def truth(col): return x[col].astype(str).str.lower().isin({"true", "1"}) if col in x else pd.Series(False, index=x.index)
    valid = truth("projected") & truth("included_in_projected_alpha_summary") & truth("valid_for_science")
    alpha = pd.to_numeric(x.get("alpha"), errors="coerce"); xmin = pd.to_numeric(x.get("xmin"), errors="coerce")
    reason = x.get("validity_exclusion_reason", pd.Series("", index=x.index)).fillna("").astype(str).str.strip()
    return x.loc[valid & np.isfinite(alpha) & np.isfinite(xmin) & (xmin > 0) & reason.eq("")].copy()


def paired_effect(metric: str, baseline: float, wwpgd: float) -> float:
    """WWPGD minus baseline for both losses/perplexities and next-token accuracy."""
    return float(wwpgd) - float(baseline)


def package_provenance() -> dict[str, Any]:
    out = {}
    for dist, module in (("weightwatcher", "weightwatcher"), ("ww-pgd", "ww_pgd")):
        spec = importlib.util.find_spec(module)
        try: version = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError: version = "not-installed"
        out[module] = {"version": version, "module_path": spec.origin if spec else None}
    return out


def write_table(params: NotebookParameters, name: str, frame: pd.DataFrame) -> Path:
    path = params.output_root / "tables" / name
    frame.to_csv(path, index=False); return path


def write_figure(params: NotebookParameters, name: str, figure) -> Path:
    path = params.output_root / "figures" / f"{name}.{params.figure_format}"
    figure.savefig(path, bbox_inches="tight"); return path


def optional_artifacts(run: Path, names: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame([{"artifact": n, "present": (run / n).is_file(),
                          "warning": "" if (run / n).is_file() else "optional artifact missing"} for n in names])
