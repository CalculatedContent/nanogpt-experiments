from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    matches = text.count(old)
    if matches != 1:
        raise RuntimeError(f"expected one match in {path}, found {matches}")
    target.write_text(text.replace(old, new, 1))


replace_once(
    "src/wwgpt/cli.py",
    '''    from wwgpt.adaptive_wwpgd import resolve_endpoint_measurement_interval
    endpoint_interval = resolve_endpoint_measurement_interval(
        cfg.wwpgd.adaptive, getattr(args, "eval_interval", None) or cfg.train.eval_interval
    )
    endpoint_steps = list(range(endpoint_interval, budget["resolved_optimizer_steps"] + 1, endpoint_interval))
    if cfg.wwpgd.adaptive.refresh_at_final_step and budget["resolved_optimizer_steps"] not in endpoint_steps:
        endpoint_steps.append(budget["resolved_optimizer_steps"])
''',
    '''    from wwgpt.adaptive_wwpgd import validate_adaptive_level_schedule
    cached_endpoint_mode = (
        cfg.wwpgd.enabled
        and cfg.wwpgd.adaptive.apply_mode == "cached_endpoint_relaxation"
    )
    endpoint_interval = (
        int(cfg.measurement.alpha_interval)
        if cached_endpoint_mode
        else _effective_ww_interval(cfg, _resolve_ww_interval_aliases(args))
    )
    adaptive_schedule = (
        validate_adaptive_level_schedule(
            cfg.wwpgd.adaptive,
            budget["resolved_optimizer_steps"],
            endpoint_interval,
        )
        if cached_endpoint_mode
        else {}
    )
    endpoint_steps = list(adaptive_schedule.get("measurement_steps", []))
''',
)
replace_once(
    "src/wwgpt/cli.py",
    '''        "wwpgd_adaptive": asdict(cfg.wwpgd.adaptive),
        "endpoint_measurement_source": cfg.wwpgd.adaptive.measurement_source,
''',
    '''        "wwpgd_adaptive": asdict(cfg.wwpgd.adaptive),
        "wwpgd_adaptive_schedule": adaptive_schedule,
        "endpoint_measurement_source": (
            "measurement.alpha_interval"
            if cached_endpoint_mode
            else cfg.wwpgd.adaptive.measurement_source
        ),
''',
)

replace_once(
    "src/wwgpt/data.py",
    '''def _log_prepare_progress(message: str) -> None:
    print(f"[wwgpt prepare-data] {message}", file=sys.stderr, flush=True)


def prepare_scientific_data''',
    '''def _log_prepare_progress(message: str) -> None:
    print(f"[wwgpt prepare-data] {message}", file=sys.stderr, flush=True)


def required_evaluation_tokens(cfg: ExperimentConfig) -> int:
    """Tokens required to materialize the complete fixed evaluation probe."""
    return int(cfg.train.eval_batches * cfg.train.batch_size * cfg.model.block_size + 1)


def evaluation_probe_capacity(token_count: int, cfg: ExperimentConfig) -> int:
    """Number of complete evaluation batches supported by a token split."""
    tokens_per_batch = int(cfg.train.batch_size * cfg.model.block_size)
    return max(0, (int(token_count) - 1) // max(tokens_per_batch, 1))


def validate_evaluation_capacity(
    data_manifest: dict[str, object], cfg: ExperimentConfig
) -> None:
    """Fail before training when validation or test probes cannot be built."""
    required = required_evaluation_tokens(cfg)
    for split, key in (("validation", "validation_tokens"), ("test", "test_tokens")):
        value = data_manifest.get(key)
        if value is None:
            raise RuntimeError(
                f"prepared data manifest is missing {key}; rebuild with `wwgpt prepare-data`"
            )
        count = int(value)
        if count < required:
            raise RuntimeError(
                f"insufficient {split} tokens for configured evaluation: "
                f"{count} < {required}; rebuild with `wwgpt prepare-data`"
            )


def prepare_scientific_data''',
)
replace_once(
    "src/wwgpt/data.py",
    '''    realized = budget.realized_tokens
    needed_train = realized + 1
    prep = unique_dir''',
    '''    realized = budget.realized_tokens
    needed_train = realized + 1
    required_probe_tokens = required_evaluation_tokens(cfg)
    required_validation_tokens = max(int(min_validation_tokens), required_probe_tokens)
    required_test_tokens = required_probe_tokens
    prep = unique_dir''',
)
replace_once(
    "src/wwgpt/data.py",
    '''    _log_prepare_progress(f"starting level={level} token_multiplier={token_multiplier} requested_tokens={requested} realized_tokens={realized} output={prep}")''',
    '''    _log_prepare_progress(
        f"starting level={level} token_multiplier={token_multiplier} "
        f"requested_tokens={requested} realized_tokens={realized} "
        f"required_validation_tokens={required_validation_tokens} "
        f"required_test_tokens={required_test_tokens} output={prep}"
    )''',
)
replace_once(
    "src/wwgpt/data.py",
    '''val_tokens={writers['val'].count}/{min_validation_tokens} elapsed_s={elapsed:.1f}''',
    '''val_tokens={writers['val'].count}/{required_validation_tokens} test_tokens={writers['test'].count}/{required_test_tokens} elapsed_s={elapsed:.1f}''',
)
replace_once(
    "src/wwgpt/data.py",
    '''        if writers and writers["train"].count >= needed_train and writers["val"].count >= min_validation_tokens and writers["test"].count > 0:
''',
    '''        if (
            writers
            and writers["train"].count >= needed_train
            and writers["val"].count >= required_validation_tokens
            and writers["test"].count >= required_test_tokens
        ):
''',
)
replace_once(
    "src/wwgpt/data.py",
    '''    if writers["val"].count < 1:
        raise ValueError("insufficient validation tokens")
    if writers["test"].count < 1:
        raise ValueError("insufficient test tokens")
''',
    '''    if writers["val"].count < required_validation_tokens:
        raise ValueError(
            f"insufficient validation tokens for configured evaluation: "
            f"{writers['val'].count} < {required_validation_tokens}"
        )
    if writers["test"].count < required_test_tokens:
        raise ValueError(
            f"insufficient test tokens for configured evaluation: "
            f"{writers['test'].count} < {required_test_tokens}"
        )
''',
)
replace_once(
    "src/wwgpt/data.py",
    '''"min_validation_tokens": min_validation_tokens, "requested_tokens": requested''',
    '''"min_validation_tokens": min_validation_tokens, "required_validation_tokens": required_validation_tokens, "required_test_tokens": required_test_tokens, "validation_probe_capacity": evaluation_probe_capacity(writers["val"].count, cfg), "test_probe_capacity": evaluation_probe_capacity(writers["test"].count, cfg), "requested_tokens": requested''',
)
replace_once(
    "src/wwgpt/data.py",
    '''        _validate_memmap_manifest(prep, dm)
        dtype = str(dm["dtype"]); splits = dm["splits"]
''',
    '''        _validate_memmap_manifest(prep, dm)
        validate_evaluation_capacity(dm, cfg)
        dtype = str(dm["dtype"]); splits = dm["splits"]
''',
)

replace_once(
    "src/wwgpt/analysis.py",
    '''    from wwgpt.alpha_analysis import analyze_alpha_trajectories
    analyze_alpha_trajectories(runs, out)
    if analysis_plan is not None:
''',
    '''    from wwgpt.alpha_analysis import analyze_alpha_trajectories
    analyze_alpha_trajectories(runs, out)
    from wwgpt.cross_level_analysis import analyze_cross_level_effects
    analyze_cross_level_effects(results_root, out, figures_dir=out / "figures")
    if analysis_plan is not None:
''',
)

notebook_path = ROOT / "notebooks/04_scaling_laws.ipynb"
notebook = json.loads(notebook_path.read_text())
parameters = notebook["cells"][1]
notebook["cells"] = [
    notebook["cells"][0],
    parameters,
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "crosslevel-setup",
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import json\n",
            "from wwgpt.notebook_support import resolve_notebook_parameters, validate_paths, package_provenance\n",
            "from wwgpt.cross_level_analysis import analyze_cross_level_effects\n",
            "P = resolve_notebook_parameters(globals())\n",
            "validate_paths(P)\n",
            "print('Resolved notebook parameters:', json.dumps(P.summary(), indent=2))\n",
            "print('Package provenance:', json.dumps(package_provenance(), indent=2))\n",
            "print('LEVEL is intentionally not used as a filter in this cross-level notebook.')\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "crosslevel-analysis",
        "metadata": {},
        "outputs": [],
        "source": [
            "outputs = analyze_cross_level_effects(P.results_root, P.output_root / 'tables', figures_dir=P.output_root / 'figures')\n",
            "readiness = outputs['cross_level_scaling_readiness.csv']\n",
            "paired = outputs['cross_level_paired_effects_by_seed.csv']\n",
            "summary = outputs['cross_level_paired_effect_summary.csv']\n",
            "trends = outputs['cross_level_model_size_trends.csv']\n",
            "display(readiness)\n",
            "display(paired.sort_values(['metric', 'level', 'seed']) if not paired.empty else paired)\n",
            "display(summary.sort_values(['metric', 'level']) if not summary.empty else summary)\n",
            "display(trends.sort_values(['metric', 'token_multiplier']) if not trends.empty else trends)\n",
        ],
    },
    {
        "cell_type": "markdown",
        "id": "crosslevel-interpretation",
        "metadata": {},
        "source": [
            "## Interpretation\n",
            "Effects are WWPGD minus baseline within seed and base optimizer. Negative is better for loss, perplexity, and gaps; positive is better for accuracy. Level trends are descriptive. A scaling-law claim requires multiple token multipliers and enough complete paired seeds.\n",
        ],
    },
]
notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")

replace_once(
    ".github/workflows/ci.yml",
    '''            python -m json.tool "artifacts/resolved-pilot-manifests/level${level}.json" >/dev/null
          done
''',
    '''            python -m json.tool "artifacts/resolved-pilot-manifests/level${level}.json" >/dev/null
          done
          python - <<'PY'
          import json
          from pathlib import Path

          expected = {
              0: (25, 75, 0.02),
              1: (250, 500, 0.002049),
              2: (1000, 3000, 0.000511),
          }
          for level, (interval, first_active, gain) in expected.items():
              payload = json.loads(Path(f"artifacts/resolved-pilot-manifests/level{level}.json").read_text())
              schedule = payload["wwpgd_adaptive_schedule"]
              assert payload["endpoint_measurement_interval"] == interval
              assert payload["endpoint_measurement_source"] == "measurement.alpha_interval"
              assert schedule["first_possible_active_endpoint_step"] == first_active
              assert abs(schedule["effective_base_gain"] - gain) <= max(1e-6, gain * 0.01)
              assert schedule["worst_case_endpoint_fraction_per_refresh"] <= 0.40 + 1e-12
          PY
''',
)

print("Applied deterministic Level 0-2 workflow fixes")
