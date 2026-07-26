from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def write(path: str, text: str, *, executable: bool = False) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    if executable:
        target.chmod(0o755)


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:100]!r}")
    write(path, text.replace(old, new, 1))


# ---------------------------------------------------------------------------
# Configuration: explicit LR scaling and candidate-device policy.
# ---------------------------------------------------------------------------
replace_once(
    "src/wwgpt/config.py",
    'VALID_EXTENSIONS = {"none", "wwpgd", "measurement_only", "norm_matched_sham", "delayed_onset"}\n',
    'VALID_EXTENSIONS = {"none", "wwpgd", "measurement_only", "norm_matched_sham", "delayed_onset"}\n'
    'VALID_LR_SCALE_RULES = {"fixed", "linear_batch", "sqrt_batch"}\n'
    'VALID_WWPGD_CANDIDATE_DEVICES = {"auto", "live", "cpu"}\n'
    'VALID_MATRIX_LR_ROLES = {\n'
    '    "token_embedding", "position_embedding", "attention_key", "attention_query",\n'
    '    "attention_value", "attention_projection", "mlp_input", "mlp_output",\n'
    '    "block_layernorm", "final_layernorm", "lm_head", "other", "block_other",\n'
    '}\n',
)
replace_once(
    "src/wwgpt/config.py",
    '    matrix_lr_multipliers: dict[str, float] = field(default_factory=dict)\n    muon_learning_rate: float = 2e-2\n',
    '    matrix_lr_multipliers: dict[str, float] = field(default_factory=dict)\n'
    '    # Standard batch-size scaling heuristics. The default preserves the\n'
    '    # configured learning rate exactly. No automatic width scaling is applied.\n'
    '    lr_scale_rule: str = "fixed"\n'
    '    lr_reference_tokens_per_step: int = 4096\n'
    '    muon_learning_rate: float = 2e-2\n',
)
replace_once(
    "src/wwgpt/config.py",
    '        if self.layer_lr not in {"flat", "llrd", "manual"}:\n            raise ValueError(f"unknown layer_lr {self.layer_lr}")\n',
    '        if self.layer_lr not in {"flat", "llrd", "manual"}:\n'
    '            raise ValueError(f"unknown layer_lr {self.layer_lr}")\n'
    '        if self.lr_scale_rule not in VALID_LR_SCALE_RULES:\n'
    '            raise ValueError(f"unknown lr_scale_rule {self.lr_scale_rule}")\n'
    '        if self.lr_reference_tokens_per_step < 1:\n'
    '            raise ValueError("lr_reference_tokens_per_step must be positive")\n'
    '        unknown_roles = sorted(set(self.matrix_lr_multipliers) - VALID_MATRIX_LR_ROLES)\n'
    '        if unknown_roles:\n'
    '            raise ValueError(f"unknown matrix_lr_multipliers roles: {unknown_roles}")\n'
    '        if any(float(value) <= 0 for value in self.matrix_lr_multipliers.values()):\n'
    '            raise ValueError("matrix_lr_multipliers values must be positive")\n',
)
replace_once(
    "src/wwgpt/config.py",
    '    use_detx: bool = True\n    warmup_events: int = 0\n',
    '    use_detx: bool = True\n'
    '    # auto uses CPU candidate generation on MPS/XLA and the live device on CPU/CUDA.\n'
    '    candidate_device: str = "auto"\n'
    '    warmup_events: int = 0\n',
)
replace_once(
    "src/wwgpt/config.py",
    '    if cfg.extension not in VALID_EXTENSIONS:\n        raise ValueError(f"unknown wwpgd.extension {cfg.extension}")\n',
    '    if cfg.extension not in VALID_EXTENSIONS:\n'
    '        raise ValueError(f"unknown wwpgd.extension {cfg.extension}")\n'
    '    if cfg.candidate_device not in VALID_WWPGD_CANDIDATE_DEVICES:\n'
    '        raise ValueError(\n'
    '            "wwpgd.candidate_device must be auto, live, or cpu"\n'
    '        )\n',
)
replace_once(
    "src/wwgpt/config.py",
    '    if cfg.layer_lr not in {"flat", "llrd", "manual"}:\n        raise ValueError(f"unknown layer_lr {cfg.layer_lr}")\n',
    '    if cfg.layer_lr not in {"flat", "llrd", "manual"}:\n'
    '        raise ValueError(f"unknown layer_lr {cfg.layer_lr}")\n'
    '    if cfg.lr_scale_rule not in VALID_LR_SCALE_RULES:\n'
    '        raise ValueError(f"unknown lr_scale_rule {cfg.lr_scale_rule}")\n'
    '    if cfg.lr_reference_tokens_per_step < 1:\n'
    '        raise ValueError("train.lr_reference_tokens_per_step must be positive")\n'
    '    unknown_roles = sorted(set(cfg.matrix_lr_multipliers) - VALID_MATRIX_LR_ROLES)\n'
    '    if unknown_roles:\n'
    '        raise ValueError(f"unknown train.matrix_lr_multipliers roles: {unknown_roles}")\n'
    '    if any(float(value) <= 0 for value in cfg.matrix_lr_multipliers.values()):\n'
    '        raise ValueError("train.matrix_lr_multipliers values must be positive")\n',
)

# ---------------------------------------------------------------------------
# Optimizer: fixed / linear-batch / sqrt-batch scaling with full provenance.
# ---------------------------------------------------------------------------
replace_once(
    "src/wwgpt/optim.py",
    "from dataclasses import dataclass\n",
    "from dataclasses import dataclass, field\n",
)
replace_once(
    "src/wwgpt/optim.py",
    'STABLEADAMW_IMPLEMENTATION = "optimi.StableAdamW"\n',
    'STABLEADAMW_IMPLEMENTATION = "optimi.StableAdamW"\n'
    'LR_SCALE_RULES = {"fixed", "linear_batch", "sqrt_batch"}\n'
    'VALID_PARAMETER_ROLES = frozenset(MANUAL_LAYER_LR_MULTIPLIERS)\n',
)
insert_marker = "\n\ndef arm_name(base_optimizer: str, extension: str) -> str:\n"
optim = read("src/wwgpt/optim.py")
if insert_marker not in optim:
    raise RuntimeError("optim.py arm_name marker not found")
lr_helpers = '''\n\ndef effective_batch_tokens(cfg: TrainConfig, block_size: int) -> int:\n    """Tokens contributing to one base-optimizer update."""\n    return int(cfg.batch_size * cfg.gradient_accumulation * block_size)\n\n\ndef learning_rate_scale_factor(cfg: TrainConfig, block_size: int) -> float:\n    """Resolve an explicit batch-size scaling heuristic.\n\n    ``fixed`` is the default and leaves the configured peak learning rates\n    unchanged. ``linear_batch`` and ``sqrt_batch`` are conventional batch-size\n    scaling ablations relative to ``lr_reference_tokens_per_step``. No implicit\n    model-width scaling is performed.\n    """\n    current = effective_batch_tokens(cfg, block_size)\n    reference = int(cfg.lr_reference_tokens_per_step)\n    if reference < 1:\n        raise ValueError("lr_reference_tokens_per_step must be positive")\n    ratio = current / reference\n    if cfg.lr_scale_rule == "fixed":\n        return 1.0\n    if cfg.lr_scale_rule == "linear_batch":\n        return float(ratio)\n    if cfg.lr_scale_rule == "sqrt_batch":\n        return float(math.sqrt(ratio))\n    raise ValueError(f"unknown lr_scale_rule {cfg.lr_scale_rule}")\n\n\ndef resolve_learning_rates(cfg: TrainConfig, block_size: int) -> dict[str, float | int | str]:\n    factor = learning_rate_scale_factor(cfg, block_size)\n    return {\n        "lr_scale_rule": cfg.lr_scale_rule,\n        "lr_scale_factor": factor,\n        "lr_reference_tokens_per_step": int(cfg.lr_reference_tokens_per_step),\n        "effective_batch_tokens": effective_batch_tokens(cfg, block_size),\n        "configured_adamw_learning_rate": float(cfg.learning_rate),\n        "effective_adamw_learning_rate": float(cfg.learning_rate * factor),\n        "configured_stableadamw_learning_rate": float(cfg.stable_learning_rate),\n        "effective_stableadamw_learning_rate": float(cfg.stable_learning_rate * factor),\n        "configured_muon_learning_rate": float(cfg.muon_learning_rate),\n        "effective_muon_learning_rate": float(cfg.muon_learning_rate * factor),\n    }\n'''
write("src/wwgpt/optim.py", optim.replace(insert_marker, lr_helpers + insert_marker, 1))
replace_once(
    "src/wwgpt/optim.py",
    '    implementation_versions: dict[str, str]\n\n    def zero_grad',
    '    implementation_versions: dict[str, str]\n'
    '    learning_rate_resolution: dict[str, Any] = field(default_factory=dict)\n\n'
    '    def zero_grad',
)
optim = read("src/wwgpt/optim.py")
start = optim.index("def build_param_groups(")
end = optim.index("\n\ndef muon_parameter_names", start)
new_build_groups = '''def build_param_groups(\n    model: nn.Module,\n    base_lr: float,\n    weight_decay: float,\n    cfg: TrainConfig,\n    *,\n    include_names: set[str] | None = None,\n    configured_base_lr: float | None = None,\n    lr_resolution: dict[str, Any] | None = None,\n) -> tuple[list[dict[str, Any]], float]:\n    named = [\n        (name, parameter)\n        for name, parameter in model.named_parameters()\n        if parameter.requires_grad and (include_names is None or name in include_names)\n    ]\n    depths = {name: parameter_role_depth(name, model)[1] for name, _ in named}\n    max_depth = max(depths.values(), default=1)\n    gamma = (\n        cfg.llrd_gamma\n        if cfg.llrd_gamma is not None\n        else cfg.llrd_min_multiplier ** (1.0 / max(max_depth, 1))\n    )\n    resolution = dict(lr_resolution or {})\n    groups: list[dict[str, Any]] = []\n    multipliers = cfg.matrix_lr_multipliers or {}\n    unknown_roles = sorted(set(multipliers) - VALID_PARAMETER_ROLES)\n    if unknown_roles:\n        raise ValueError(f"unknown matrix learning-rate roles: {unknown_roles}")\n    for name, parameter in named:\n        role, depth = parameter_role_depth(name, model)\n        if cfg.layer_lr == "flat":\n            layer_multiplier = 1.0\n        elif cfg.layer_lr == "llrd":\n            layer_multiplier = gamma ** (max_depth - depth)\n        elif cfg.layer_lr == "manual":\n            layer_multiplier = MANUAL_LAYER_LR_MULTIPLIERS.get(\n                role, MANUAL_LAYER_LR_MULTIPLIERS["other"]\n            )\n        else:\n            raise ValueError(f"unknown layer_lr {cfg.layer_lr}")\n        matrix_multiplier = float(multipliers.get(role, 1.0))\n        peak_lr = float(base_lr * layer_multiplier * matrix_multiplier)\n        groups.append(\n            {\n                "params": [parameter],\n                "lr": peak_lr,\n                "initial_lr": peak_lr,\n                "peak_lr": peak_lr,\n                "minimum_lr": peak_lr * cfg.min_lr_ratio,\n                "configured_base_lr": float(\n                    configured_base_lr if configured_base_lr is not None else base_lr\n                ),\n                "effective_base_lr": float(base_lr),\n                "lr_scale_rule": resolution.get("lr_scale_rule", cfg.lr_scale_rule),\n                "lr_scale_factor": float(resolution.get("lr_scale_factor", 1.0)),\n                "lr_reference_tokens_per_step": int(\n                    resolution.get(\n                        "lr_reference_tokens_per_step",\n                        cfg.lr_reference_tokens_per_step,\n                    )\n                ),\n                "effective_batch_tokens": int(\n                    resolution.get("effective_batch_tokens", 0)\n                ),\n                "weight_decay": weight_decay if _decay_for(name, parameter) else 0.0,\n                "group_name": name,\n                "parameter_name": name,\n                "role": role,\n                "depth": depth,\n                "layer_lr_multiplier": layer_multiplier,\n                "matrix_specific_multiplier": matrix_multiplier,\n                "parameter_count": parameter.numel(),\n            }\n        )\n    return groups, gamma\n'''
write("src/wwgpt/optim.py", optim[:start] + new_build_groups + optim[end:])
optim = read("src/wwgpt/optim.py")
start = optim.index("def build_optimizer_bundle(")
end = optim.index("\n\nSCHEDULER_IMPLEMENTATION", start)
new_bundle = '''def build_optimizer_bundle(\n    model: nn.Module, cfg: TrainConfig, base_optimizer: str\n) -> tuple[OptimizerBundle, float]:\n    base_optimizer = "stableadamw" if base_optimizer == "stable_adamw" else base_optimizer\n    block_size = int(getattr(getattr(model, "cfg", None), "block_size", 1))\n    resolution = resolve_learning_rates(cfg, block_size)\n    adamw_lr = float(resolution["effective_adamw_learning_rate"])\n    stable_lr = float(resolution["effective_stableadamw_learning_rate"])\n    muon_lr = float(resolution["effective_muon_learning_rate"])\n    if base_optimizer == "adamw":\n        groups, gamma = build_param_groups(\n            model, adamw_lr, cfg.weight_decay, cfg,\n            configured_base_lr=cfg.learning_rate, lr_resolution=resolution\n        )\n        optimizer = torch.optim.AdamW(groups, betas=cfg.betas, eps=cfg.epsilon)\n        return OptimizerBundle(\n            "adamw", [optimizer], [("adamw", optimizer)],\n            {"adamw": f"{ADAMW_IMPLEMENTATION}:{torch.__version__}"}, resolution\n        ), gamma\n    if base_optimizer == "stableadamw":\n        try:\n            from optimi import StableAdamW\n        except Exception as exc:\n            raise RuntimeError(\n                "cannot construct requested optimizer stableadamw: install the "\n                "'torch-optimi' package providing optimi.StableAdamW"\n            ) from exc\n        try:\n            optimi_version = metadata.version("torch-optimi")\n        except metadata.PackageNotFoundError:\n            try:\n                optimi_version = metadata.version("optimi")\n            except metadata.PackageNotFoundError:\n                optimi_version = "installed-version-unknown"\n        groups, gamma = build_param_groups(\n            model, stable_lr, cfg.weight_decay, cfg,\n            configured_base_lr=cfg.stable_learning_rate, lr_resolution=resolution\n        )\n        optimizer = StableAdamW(\n            groups, lr=stable_lr, betas=cfg.stable_betas, eps=cfg.stable_epsilon,\n            weight_decay=0.0, triton=cfg.stable_triton\n        )\n        return OptimizerBundle(\n            "stableadamw", [optimizer], [("stableadamw", optimizer)],\n            {"stableadamw": f"{STABLEADAMW_IMPLEMENTATION}:{optimi_version}"}, resolution\n        ), gamma\n    if base_optimizer == "muon":\n        muon_names = muon_parameter_names(model)\n        adamw_names = {\n            name for name, parameter in model.named_parameters() if parameter.requires_grad\n        } - muon_names\n        muon_groups, gamma = build_param_groups(\n            model, muon_lr, cfg.weight_decay, cfg, include_names=muon_names,\n            configured_base_lr=cfg.muon_learning_rate, lr_resolution=resolution\n        )\n        adamw_groups, gamma2 = build_param_groups(\n            model, adamw_lr, cfg.weight_decay, cfg, include_names=adamw_names,\n            configured_base_lr=cfg.learning_rate, lr_resolution=resolution\n        )\n        muon = Muon(\n            muon_groups, lr=muon_lr, momentum=cfg.muon_momentum,\n            newton_schulz_steps=cfg.newton_schulz_steps, weight_decay=0.0\n        )\n        auxiliary = torch.optim.AdamW(adamw_groups, betas=cfg.betas, eps=cfg.epsilon)\n        versions = {\n            "muon": MUON_IMPLEMENTATION_VERSION,\n            "muon_aux_adamw": f"{ADAMW_IMPLEMENTATION}:{torch.__version__}",\n        }\n        return OptimizerBundle(\n            "muon", [muon, auxiliary],\n            [("muon", muon), ("muon_aux_adamw", auxiliary)], versions, resolution\n        ), gamma or gamma2\n    raise ValueError(f"cannot construct requested optimizer {base_optimizer}: unknown optimizer")\n'''
write("src/wwgpt/optim.py", optim[:start] + new_bundle + optim[end:])
replace_once(
    "src/wwgpt/optim.py",
    '                "epsilon": None if epsilon is None else float(epsilon),\n',
    '                "epsilon": None if epsilon is None else float(epsilon),\n'
    '                "configured_base_lr": float(group.get("configured_base_lr", group.get("peak_lr", 0.0))),\n'
    '                "effective_base_lr": float(group.get("effective_base_lr", group.get("peak_lr", 0.0))),\n'
    '                "lr_scale_rule": group.get("lr_scale_rule", "fixed"),\n'
    '                "lr_scale_factor": float(group.get("lr_scale_factor", 1.0)),\n'
    '                "effective_batch_tokens": int(group.get("effective_batch_tokens", 0)),\n'
    '                "lr_reference_tokens_per_step": int(group.get("lr_reference_tokens_per_step", 0)),\n',
)
replace_once(
    "src/wwgpt/optim.py",
    '        "parameter_groups": tuple(groups),\n',
    '        "parameter_groups": tuple(groups),\n'
    '        "learning_rate_resolution": dict(bundle.learning_rate_resolution),\n',
)
replace_once(
    "src/wwgpt/optim.py",
    '"scheduler_implementation": SCHEDULER_IMPLEMENTATION})\n',
    '"scheduler_implementation": SCHEDULER_IMPLEMENTATION, '
    '"configured_base_lr": g.get("configured_base_lr"), '
    '"effective_base_lr": g.get("effective_base_lr"), '
    '"lr_scale_rule": g.get("lr_scale_rule", cfg.lr_scale_rule), '
    '"lr_scale_factor": g.get("lr_scale_factor", 1.0), '
    '"effective_batch_tokens": g.get("effective_batch_tokens", 0), '
    '"lr_reference_tokens_per_step": g.get("lr_reference_tokens_per_step", cfg.lr_reference_tokens_per_step)})\n',
)

# ---------------------------------------------------------------------------
# WWPGD candidate generation: CPU offload on MPS/XLA and explicit provenance.
# ---------------------------------------------------------------------------
replace_once("src/wwgpt/ww.py", "import json\n", "import copy\nimport json\n")
replace_once(
    "src/wwgpt/ww.py",
    '    max_relative_frobenius_change: float | None = None\n',
    '    max_relative_frobenius_change: float | None = None\n'
    '    candidate_device: str = "auto"\n',
)
replace_once(
    "src/wwgpt/ww.py",
    '        max_relative_frobenius_change=getattr(cfg, "max_relative_frobenius_change", None),\n',
    '        max_relative_frobenius_change=getattr(cfg, "max_relative_frobenius_change", None),\n'
    '        candidate_device=str(getattr(cfg, "candidate_device", "auto")),\n',
)
replace_once(
    "src/wwgpt/ww.py",
    '        max_relative_frobenius_change=cfg.max_relative_frobenius_change,\n',
    '        max_relative_frobenius_change=cfg.max_relative_frobenius_change,\n'
    '        candidate_device=cfg.candidate_device,\n',
)
replace_once(
    "src/wwgpt/ww.py",
    '        "use_detx": cfg.use_detx,\n',
    '        "use_detx": cfg.use_detx,\n'
    '        "candidate_device": cfg.candidate_device,\n',
)
replace_once(
    "src/wwgpt/ww.py",
    '    internal_diagnostics: list[dict[str, object]] = dataclasses.field(default_factory=list)\n    stock_commit: str = WWPGD_COMMIT\n',
    '    internal_diagnostics: list[dict[str, object]] = dataclasses.field(default_factory=list)\n'
    '    stock_commit: str = WWPGD_COMMIT\n'
    '    candidate_execution_device: str = "live"\n'
    '    live_model_device: str = "cpu"\n'
    '    candidate_offloaded: bool = False\n',
)
ww = read("src/wwgpt/ww.py")
start = ww.index("def build_stock_wwpgd_candidate(")
end = ww.index("\n\ndef apply_external_wwpgd", start)
new_candidate = '''def _model_device(model: nn.Module) -> torch.device:\n    try:\n        return next(model.parameters()).device\n    except StopIteration:\n        return torch.device("cpu")\n\n\ndef resolve_candidate_execution_device(model: nn.Module, requested: str) -> str:\n    requested = str(requested or "auto").lower()\n    if requested not in {"auto", "live", "cpu"}:\n        raise ValueError("candidate_device must be auto, live, or cpu")\n    if requested == "cpu":\n        return "cpu"\n    if requested == "live":\n        return "live"\n    return "cpu" if _model_device(model).type in {"mps", "xla"} else "live"\n\n\ndef _cpu_candidate_model(model: nn.Module) -> nn.Module:\n    clone = copy.deepcopy(model).to(torch.device("cpu"))\n    clone.load_state_dict(\n        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}\n    )\n    clone.train(model.training)\n    return clone\n\n\ndef build_stock_wwpgd_candidate(\n    model: nn.Module,\n    *,\n    event_index: int = 0,\n    actual_step: int = 0,\n    cfg: ExternalWWTailConfigSpec | None = None,\n    selected_names: set[str] | None = None,\n    layer_selector: object | None = None,\n) -> StockWWPGDCandidate:\n    cfg = cfg or resolved_external_wwpgd_config()\n    full_cfg = ExternalWWTailConfigSpec(\n        enable_tail_pgd=cfg.enable_tail_pgd,\n        target_alpha=cfg.target_alpha,\n        blend_eta=cfg.blend_eta,\n        cayley_eta=cfg.cayley_eta,\n        min_tail=cfg.min_tail,\n        use_detx=cfg.use_detx,\n        warmup_epochs=cfg.warmup_epochs,\n        ramp_epochs=cfg.ramp_epochs,\n        verbose=cfg.verbose,\n        max_relative_frobenius_change=cfg.max_relative_frobenius_change,\n        candidate_device=cfg.candidate_device,\n    )\n    ww_pgd_module = _external_wwpgd_module()\n    projector = getattr(ww_pgd_module, "ww_pgd_project")\n    _assert_stock_wwpgd_api(projector)\n    selected_names = selected_names or set(external_projected_layer_names(model))\n    originals = {name: weight.detach().clone() for name, weight in projected_matrix_modules(model)}\n    live_device = _model_device(model)\n    execution_mode = resolve_candidate_execution_device(model, full_cfg.candidate_device)\n    execution_model = model if execution_mode == "live" else _cpu_candidate_model(model)\n    selector = layer_selector or _selected_layer_selector(set(selected_names))\n    start = time.perf_counter()\n    candidates: dict[str, torch.Tensor] = {}\n    result: dict[str, object] = {}\n    try:\n        with torch.no_grad():\n            result = run_pip_wwpgd_candidate(\n                execution_model,\n                _external_config_object(ww_pgd_module, full_cfg),\n                epoch=event_index,\n                num_epochs=max(event_index + 1, 1),\n                global_step=actual_step,\n                layer_selector=selector,\n            )\n            candidates = {\n                name: weight.detach().clone().cpu()\n                for name, weight in projected_matrix_modules(execution_model)\n            }\n    finally:\n        if execution_mode == "live":\n            with torch.no_grad():\n                for name, weight in projected_matrix_modules(model):\n                    weight.copy_(originals[name].to(weight.device, dtype=weight.dtype))\n                    if not torch.equal(weight.detach().cpu(), originals[name].cpu()):\n                        raise RuntimeError(\n                            f"failed to restore original WW_PGD weight bitwise for {name}"\n                        )\n    runtime = time.perf_counter() - start\n    missing_candidates = sorted(set(originals) - set(candidates))\n    if missing_candidates:\n        raise RuntimeError(f"WWPGD candidate is missing projected matrices: {missing_candidates}")\n    ww_logs = result.get("ww_logs", [])\n    diagnostic_logs = list(result.get("diagnostic_logs", []))\n    usable = [item for item in ww_logs if isinstance(item, pd.DataFrame) and not item.empty]\n    if len(usable) != 1:\n        raise RuntimeError(\n            "stock WW_PGD candidate generation expected exactly one usable "\n            f"ww_logs DataFrame, got {len(usable)}"\n        )\n    relative_change: dict[str, float] = {}\n    changed: dict[str, bool] = {}\n    with torch.no_grad():\n        for name, original in originals.items():\n            candidate = candidates[name].to(original.device, dtype=original.dtype)\n            displacement = (candidate - original).float()\n            relative_change[name] = float(\n                torch.linalg.norm(displacement)\n                / max(float(torch.linalg.norm(original.float())), 1e-12)\n            )\n            changed[name] = not torch.equal(candidates[name].cpu(), original.cpu())\n    common = {\n        "candidate_execution_device": execution_mode,\n        "live_model_device": str(live_device),\n        "candidate_offloaded": execution_mode != "live",\n    }\n    if result.get("native_internal_diagnostics"):\n        for row in diagnostic_logs:\n            row.setdefault("diagnostics_schema_version", 1)\n            row.setdefault("diagnostics_mode", "native")\n            row.setdefault("native_internal_diagnostics", True)\n            row.setdefault("valid_observable_diagnostic", True)\n            row.setdefault("unsupported_internal_fields", json.dumps([]))\n            row.update(common)\n    else:\n        unsupported = [\n            "k_pl", "k_detx", "k_star", "selected_lambda_threshold",\n            "selected_tail_size", "TraceLog", "cayley_ratios", "clipping_counts",\n            "shaped_movement",\n        ]\n        frame = usable[0]\n        name_column = "longname" if "longname" in frame.columns else "name"\n        by_name = {str(row.get(name_column, "")): row for _, row in frame.iterrows()}\n        for name, original in originals.items():\n            observed = by_name.get(name, {})\n            diagnostic_logs.append(\n                {\n                    "diagnostics_schema_version": 1,\n                    "diagnostics_mode": "compatibility",\n                    "native_internal_diagnostics": False,\n                    "valid_observable_diagnostic": bool(\n                        math.isfinite(relative_change[name])\n                    ),\n                    "status": "unsupported_internal_fields",\n                    "unsupported_internal_fields": json.dumps(unsupported),\n                    "layer_name": name,\n                    "layer_shape": list(original.shape),\n                    "alpha": observed.get("alpha"),\n                    "D": observed.get("D"),\n                    "xmin": observed.get("xmin"),\n                    "detX_num": observed.get("detX_num"),\n                    "num_evals": observed.get("num_evals"),\n                    "candidate_changed": changed[name],\n                    "original_to_candidate_relative_frobenius_change": relative_change[name],\n                    "original_frobenius_norm": float(original.float().norm()),\n                    "candidate_frobenius_norm": float(candidates[name].float().norm()),\n                    "target_alpha": full_cfg.target_alpha,\n                    "candidate_relative_frobenius_change": relative_change[name],\n                    "configured_blend_eta": full_cfg.blend_eta,\n                    "configured_cayley_eta": full_cfg.cayley_eta,\n                    "configured_min_tail": full_cfg.min_tail,\n                    "configured_use_detx": full_cfg.use_detx,\n                    "projection_runtime": runtime,\n                    "warning_message": (\n                        "private WWPGD internals are unsupported by the installed package"\n                    ),\n                    **common,\n                    **_WWPGD_PROVENANCE,\n                }\n            )\n    return StockWWPGDCandidate(\n        usable[0].copy(),\n        originals,\n        candidates,\n        relative_change,\n        changed,\n        runtime,\n        full_cfg,\n        diagnostic_logs,\n        WWPGD_COMMIT,\n        execution_mode,\n        str(live_device),\n        execution_mode != "live",\n    )\n'''
write("src/wwgpt/ww.py", ww[:start] + new_candidate + ww[end:])

# ---------------------------------------------------------------------------
# CLI: expose all scientifically relevant LR, dose, and candidate controls.
# ---------------------------------------------------------------------------
replace_once(
    "src/wwgpt/cli.py",
    '        ("llrd_min_multiplier", "llrd_min_multiplier", train_updates),\n',
    '        ("llrd_min_multiplier", "llrd_min_multiplier", train_updates),\n'
    '        ("lr_scale_rule", "lr_scale_rule", train_updates),\n'
    '        ("lr_reference_tokens_per_step", "lr_reference_tokens_per_step", train_updates),\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '                     ("wwpgd_use_detx", "use_detx")]:\n',
    '                     ("wwpgd_use_detx", "use_detx"),\n'
    '                     ("wwpgd_candidate_device", "candidate_device")]:\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '("wwpgd_log_every_fast_step","log_every_fast_step")]:\n',
    '("wwpgd_log_every_fast_step","log_every_fast_step"),'
    '("wwpgd_dose_schedule","dose_schedule"),'
    '("wwpgd_max_endpoint_fraction_per_refresh","max_endpoint_fraction_per_refresh"),'
    '("wwpgd_max_cumulative_relative_change_per_refresh","max_cumulative_relative_frobenius_change_per_refresh")]:\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '        role, value = item.split("=", 1)\n        if not role or role in cli_matrix_roles:\n',
    '        role, value = item.split("=", 1)\n'
    '        from wwgpt.optim import VALID_PARAMETER_ROLES\n'
    '        if role not in VALID_PARAMETER_ROLES:\n'
    '            raise SystemExit(\n'
    '                f"unknown matrix learning-rate role {role!r}; "\n'
    '                f"expected one of {sorted(VALID_PARAMETER_ROLES)}"\n'
    '            )\n'
    '        if not role or role in cli_matrix_roles:\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '        parser.add_argument("--matrix-lr-multiplier", action="append", default=[])\n',
    '        parser.add_argument("--matrix-lr-multiplier", action="append", default=[])\n'
    '        parser.add_argument("--lr-scale-rule", choices=["fixed","linear_batch","sqrt_batch"])\n'
    '        parser.add_argument("--lr-reference-tokens-per-step", type=int)\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '        parser.add_argument("--wwpgd-use-detx", action=argparse.BooleanOptionalAction, default=None)\n',
    '        parser.add_argument("--wwpgd-use-detx", action=argparse.BooleanOptionalAction, default=None)\n'
    '        parser.add_argument("--wwpgd-candidate-device", choices=["auto","live","cpu"])\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '        parser.add_argument("--wwpgd-log-every-fast-step", action=argparse.BooleanOptionalAction, default=None)\n',
    '        parser.add_argument("--wwpgd-log-every-fast-step", action=argparse.BooleanOptionalAction, default=None)\n'
    '        parser.add_argument("--wwpgd-dose-schedule", choices=["bounded_refresh_fraction","fixed_per_step_gain"])\n'
    '        parser.add_argument("--wwpgd-max-endpoint-fraction-per-refresh", type=float)\n'
    '        parser.add_argument("--wwpgd-max-cumulative-relative-change-per-refresh", type=float)\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '    dp=sub.add_parser("device-preflight"); dp.add_argument("--device", default="auto"); dp.add_argument("--output", type=Path, default=Path("."))\n',
    '    dp=sub.add_parser("device-preflight"); dp.add_argument("--device", default="auto"); dp.add_argument("--output", type=Path, default=Path("."))\n'
    '    lr=sub.add_parser("local-readiness", help="validate a local Mac/CPU/CUDA environment with real pip-installed WeightWatcher and WWPGD")\n'
    '    lr.add_argument("--device", default="auto")\n'
    '    lr.add_argument("--levels", default="0,1,2")\n'
    '    lr.add_argument("--optimizers", default="adamw,stableadamw,muon")\n'
    '    lr.add_argument("--output", type=Path, default=Path("local-readiness"))\n'
    '    lr.add_argument("--skip-package-smoke", action="store_true")\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '    elif args.cmd=="device-preflight":\n        import json; print(json.dumps(run_device_preflight(args.output, args.device), indent=2, sort_keys=True, default=str))\n',
    '    elif args.cmd=="device-preflight":\n'
    '        import json; print(json.dumps(run_device_preflight(args.output, args.device), indent=2, sort_keys=True, default=str))\n'
    '    elif args.cmd=="local-readiness":\n'
    '        import json\n'
    '        from wwgpt.local_readiness import run_local_readiness\n'
    '        levels=[int(value) for value in args.levels.split(",") if value]\n'
    '        optimizers=[value for value in args.optimizers.split(",") if value]\n'
    '        report=run_local_readiness(args.output,args.device,levels,optimizers,not args.skip_package_smoke)\n'
    '        print(json.dumps(report,indent=2,sort_keys=True,default=str))\n'
    '        raise SystemExit(0 if report.get("ready") else 1)\n',
)

# ---------------------------------------------------------------------------
# Explicit defaults in all three scientific configurations.
# ---------------------------------------------------------------------------
for level in (0, 1, 2):
    path = f"configs/level{level}_adaptive_alpha.yaml"
    text = read(path)
    text = text.replace(
        "  matrix_lr_multipliers: {}\n" if "  matrix_lr_multipliers: {}\n" in text else "  llrd_min_multiplier: 0.5\n",
        ("  matrix_lr_multipliers: {}\n  lr_scale_rule: fixed\n  lr_reference_tokens_per_step: 4096\n"
         if "  matrix_lr_multipliers: {}\n" in text
         else "  llrd_min_multiplier: 0.5\n  lr_scale_rule: fixed\n  lr_reference_tokens_per_step: 4096\n"),
        1,
    )
    if "  candidate_device:" not in text:
        text = text.replace("  use_detx: true\n", "  use_detx: true\n  candidate_device: auto\n", 1)
    write(path, text)

# ---------------------------------------------------------------------------
# Local readiness module: real package smoke at Level 0/1/2 and all optimizers.
# ---------------------------------------------------------------------------
write(
    "src/wwgpt/local_readiness.py",
    '''from __future__ import annotations\n\nimport json\nimport math\nimport os\nimport platform\nimport shutil\nimport sys\nimport time\nfrom dataclasses import asdict, replace\nfrom pathlib import Path\nfrom typing import Any\n\nimport torch\n\nfrom wwgpt.adaptive_wwpgd import validate_adaptive_level_schedule\nfrom wwgpt.config import load_config\nfrom wwgpt.device import device_summary, resolve_device, synchronize_device\nfrom wwgpt.model import GPT\nfrom wwgpt.optim import (\n    apply_lr_schedule,\n    build_optimizer_bundle,\n    optimizer_fingerprint,\n    resolve_learning_rates,\n    resolve_warmup_steps,\n)\nfrom wwgpt.pip_wwpgd_adapter import resolve_pip_wwpgd_provenance\nfrom wwgpt.scaling import plan_budget, selected_parameter_count\nfrom wwgpt.ww import (\n    apply_external_wwpgd,\n    build_stock_wwpgd_candidate,\n    external_wwpgd_config_from_experiment,\n    is_projected_layer,\n)\n\n\ndef _memory_bytes() -> int | None:\n    try:\n        pages = int(os.sysconf("SC_PHYS_PAGES"))\n        page_size = int(os.sysconf("SC_PAGE_SIZE"))\n        return pages * page_size\n    except (AttributeError, OSError, ValueError):\n        return None\n\n\ndef _level_config(level: int) -> Path:\n    return Path(f"configs/level{level}_adaptive_alpha.yaml")\n\n\ndef estimate_level(level: int, token_multiplier: int = 20) -> dict[str, Any]:\n    cfg = load_config(_level_config(level), level)\n    model = GPT(cfg.model)\n    report = model.parameter_report()\n    count = selected_parameter_count(report, cfg.parameter_count_convention)\n    budget = plan_budget(\n        count, token_multiplier, cfg.train.batch_size, cfg.model.block_size,\n        cfg.train.gradient_accumulation, 10**18\n    )\n    schedule = validate_adaptive_level_schedule(\n        cfg.wwpgd.adaptive, budget.steps, cfg.measurement.alpha_interval\n    )\n    evaluation_events = math.ceil(budget.steps / cfg.train.eval_interval)\n    return {\n        "level": level,\n        "config": str(_level_config(level)),\n        "model": asdict(cfg.model),\n        "parameter_report": asdict(report),\n        "selected_parameter_count": count,\n        "optimizer_steps_per_arm": budget.steps,\n        "tokens_per_arm": budget.realized_tokens,\n        "tokens_per_optimizer_step": (\n            cfg.train.batch_size * cfg.train.gradient_accumulation * cfg.model.block_size\n        ),\n        "evaluation_events_per_arm": evaluation_events,\n        "evaluation_tokens_per_arm": (\n            evaluation_events * cfg.train.eval_batches * cfg.train.batch_size\n            * cfg.model.block_size\n        ),\n        "alpha_measurement_interval": cfg.measurement.alpha_interval,\n        "trap_measurement_interval": cfg.measurement.trap_diagnostic_interval,\n        "adaptive_schedule": schedule,\n        "learning_rate_resolution": resolve_learning_rates(cfg.train, cfg.model.block_size),\n        "candidate_device": cfg.wwpgd.candidate_device,\n    }\n\n\ndef _protected_state(model: GPT) -> dict[str, torch.Tensor]:\n    eligible = {\n        f"{name}.weight"\n        for name, module in model.named_modules()\n        if is_projected_layer(name) and getattr(module, "weight", None) is not None\n    }\n    return {\n        name: value.detach().cpu().clone()\n        for name, value in model.state_dict().items()\n        if name not in eligible\n    }\n\n\ndef _assert_protected(model: GPT, before: dict[str, torch.Tensor]) -> None:\n    state = model.state_dict()\n    changed = [\n        name for name, value in before.items()\n        if name not in state or not torch.equal(value, state[name].detach().cpu())\n    ]\n    if changed:\n        raise RuntimeError(f"WWPGD changed protected tensors: {changed}")\n\n\ndef run_real_package_smoke(level: int, optimizer_name: str, device: str) -> dict[str, Any]:\n    cfg = load_config(_level_config(level), level)\n    model_cfg = replace(cfg.model, block_size=16, vocab_size=256)\n    train_cfg = replace(\n        cfg.train, batch_size=1, gradient_accumulation=1, max_steps=2,\n        eval_batches=1, lr_scale_rule="fixed"\n    )\n    resolved_device = resolve_device(device)\n    torch.manual_seed(1000 + level)\n    model = GPT(model_cfg).to(resolved_device)\n    bundle, _ = build_optimizer_bundle(model, train_cfg, optimizer_name)\n    warmup = resolve_warmup_steps(2, train_cfg.warmup_ratio, train_cfg.warmup_steps)\n    x = torch.randint(0, model_cfg.vocab_size, (1, model_cfg.block_size), device=resolved_device)\n    y = torch.randint(0, model_cfg.vocab_size, (1, model_cfg.block_size), device=resolved_device)\n    _, loss = model(x, y)\n    assert loss is not None\n    loss.backward()\n    apply_lr_schedule(bundle, 0, 2, warmup, train_cfg)\n    for optimizer in bundle.optimizers:\n        optimizer.step()\n    bundle.zero_grad()\n    synchronize_device(resolved_device)\n    protected = _protected_state(model)\n    ww_cfg = replace(cfg.wwpgd, candidate_device="auto")\n    candidate = build_stock_wwpgd_candidate(\n        model, event_index=0, actual_step=1,\n        cfg=external_wwpgd_config_from_experiment(ww_cfg),\n    )\n    hardness = {name: 0.02 for name in candidate.original_weights}\n    rows = apply_external_wwpgd(\n        model, event_index=0, actual_step=1, actual_tokens_seen=model_cfg.block_size,\n        cfg=external_wwpgd_config_from_experiment(ww_cfg),\n        layer_hardness=hardness, global_event_hardness=1.0,\n        stock_candidate=candidate,\n    )\n    synchronize_device(resolved_device)\n    _assert_protected(model, protected)\n    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):\n        raise RuntimeError("nonfinite model parameter after WWPGD smoke")\n    return {\n        "level": level,\n        "optimizer": optimizer_name,\n        "device": str(resolved_device),\n        "loss": float(loss.detach().cpu()),\n        "wwpgd_rows": len(rows),\n        "stock_candidate_changed_layers": int(sum(candidate.stock_candidate_changed.values())),\n        "applied_changed_layers": int(sum(bool(row.get("changed")) for row in rows)),\n        "candidate_execution_device": candidate.candidate_execution_device,\n        "candidate_offloaded": candidate.candidate_offloaded,\n        "diagnostics_mode": sorted({\n            str(row.get("diagnostics_mode", "unknown"))\n            for row in candidate.internal_diagnostics\n        }),\n        "optimizer_fingerprint": optimizer_fingerprint(bundle),\n        "finite": True,\n    }\n\n\ndef run_local_readiness(\n    output: Path,\n    device: str = "auto",\n    levels: list[int] | None = None,\n    optimizers: list[str] | None = None,\n    run_package_smoke: bool = True,\n) -> dict[str, Any]:\n    output = Path(output)\n    output.mkdir(parents=True, exist_ok=True)\n    levels = levels or [0, 1, 2]\n    optimizers = optimizers or ["adamw", "stableadamw", "muon"]\n    findings: list[dict[str, str]] = []\n    started = time.perf_counter()\n    try:\n        resolved = device_summary(device)\n    except Exception as exc:\n        resolved = {"requested_device": device, "error": str(exc)}\n        findings.append({"severity": "ERROR", "check": "device", "message": str(exc)})\n    estimates = [estimate_level(level) for level in levels]\n    disk = shutil.disk_usage(output)\n    memory = _memory_bytes()\n    if disk.free < 10 * 1024**3:\n        findings.append({\n            "severity": "WARNING", "check": "disk",\n            "message": f"less than 10 GiB free at {output}",\n        })\n    smokes: list[dict[str, Any]] = []\n    if run_package_smoke and not any(row["severity"] == "ERROR" for row in findings):\n        for level in levels:\n            for optimizer in optimizers:\n                try:\n                    smokes.append(run_real_package_smoke(level, optimizer, device))\n                except Exception as exc:\n                    findings.append({\n                        "severity": "ERROR",\n                        "check": f"level_{level}_{optimizer}_real_package_smoke",\n                        "message": f"{type(exc).__name__}: {exc}",\n                    })\n    report = {\n        "ready": not any(row["severity"] == "ERROR" for row in findings),\n        "platform": platform.platform(),\n        "machine": platform.machine(),\n        "python": sys.version,\n        "torch": torch.__version__,\n        "device": resolved,\n        "package_provenance": resolve_pip_wwpgd_provenance(),\n        "physical_memory_bytes": memory,\n        "disk_total_bytes": disk.total,\n        "disk_free_bytes": disk.free,\n        "levels": estimates,\n        "real_package_smokes": smokes,\n        "findings": findings,\n        "elapsed_seconds": time.perf_counter() - started,\n    }\n    (output / "local_readiness.json").write_text(\n        json.dumps(report, indent=2, sort_keys=True, default=str) + "\\n"\n    )\n    return report\n''',
)

# ---------------------------------------------------------------------------
# Mac/local experiment runner with optimizer, LR, and WWPGD-mode matrix.
# ---------------------------------------------------------------------------
write(
    "scripts/run_local_mac_experiments.sh",
    '''#!/usr/bin/env bash\nset -euo pipefail\n\nif [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then\n  echo "usage: $0 DATA_ROOT RESULTS_ROOT [TOKEN_MULTIPLIER=20] [DEVICE=auto]" >&2\n  exit 2\nfi\n\nDATA_ROOT="$1"\nRESULTS_ROOT="$2"\nTOKEN_MULTIPLIER="${3:-20}"\nDEVICE="${4:-auto}"\nLEVELS="${WWGPT_LEVELS:-0,1,2}"\nSEEDS="${WWGPT_SEEDS:-1337,2027,4099}"\nOPTIMIZERS="${WWGPT_OPTIMIZERS:-adamw}"\nWWPGD_MODES="${WWGPT_WWPGD_MODES:-adaptive}"\nUNIFORM_HARDNESS="${WWGPT_UNIFORM_HARDNESS:-0.25}"\nLAYER_LR="${WWGPT_LAYER_LR:-flat}"\nLR_SCHEDULE="${WWGPT_LR_SCHEDULE:-warmup_cosine}"\nLR_SCALE_RULE="${WWGPT_LR_SCALE_RULE:-fixed}"\nLR_REFERENCE_TOKENS="${WWGPT_LR_REFERENCE_TOKENS:-4096}"\nCANDIDATE_DEVICE="${WWGPT_CANDIDATE_DEVICE:-auto}"\nRESUME="${WWGPT_RESUME:-1}"\nRUN_NOTEBOOKS="${WWGPT_RUN_NOTEBOOKS:-1}"\nANALYSIS_PLAN="${WWGPT_ANALYSIS_PLAN:-configs/analysis_plan_exploratory.yaml}"\n\nmkdir -p "$RESULTS_ROOT"\nwwgpt local-readiness \\\n  --device "$DEVICE" \\\n  --levels "$LEVELS" \\\n  --optimizers "$OPTIMIZERS" \\\n  --output "$RESULTS_ROOT/local-readiness"\n\nIFS=',' read -r -a LEVEL_ARRAY <<< "$LEVELS"\nIFS=',' read -r -a OPTIMIZER_ARRAY <<< "$OPTIMIZERS"\nIFS=',' read -r -a MODE_ARRAY <<< "$WWPGD_MODES"\n\nfor MODE in "${MODE_ARRAY[@]}"; do\n  if [ "$MODE" = "adaptive" ]; then\n    MODE_ARGS=(--wwpgd-adaptive-mode alpha_distance)\n  elif [ "$MODE" = "uniform" ]; then\n    MODE_ARGS=(--wwpgd-adaptive-mode uniform --wwpgd-alpha-max-hardness "$UNIFORM_HARDNESS")\n  else\n    echo "unknown WWPGD mode: $MODE (expected adaptive or uniform)" >&2\n    exit 2\n  fi\n  for OPTIMIZER in "${OPTIMIZER_ARRAY[@]}"; do\n    VARIANT_ROOT="$RESULTS_ROOT/${MODE}/${OPTIMIZER}/${LAYER_LR}/${LR_SCALE_RULE}"\n    mkdir -p "$VARIANT_ROOT"\n    for LEVEL in "${LEVEL_ARRAY[@]}"; do\n      CONFIG="configs/level${LEVEL}_adaptive_alpha.yaml"\n      echo "[local-mac] prepare level=$LEVEL mode=$MODE optimizer=$OPTIMIZER" >&2\n      wwgpt prepare-data \\\n        --level "$LEVEL" --config "$CONFIG" --data-root "$DATA_ROOT" \\\n        --token-multiplier "$TOKEN_MULTIPLIER"\n      COMMON=(\n        --level "$LEVEL" --config "$CONFIG" --analysis-plan "$ANALYSIS_PLAN"\n        --data-root "$DATA_ROOT" --results-root "$VARIANT_ROOT"\n        --token-multiplier "$TOKEN_MULTIPLIER" --seeds "$SEEDS"\n        --optimizer "$OPTIMIZER" --extensions none,wwpgd --device "$DEVICE"\n        --layer-lr "$LAYER_LR" --lr-schedule "$LR_SCHEDULE"\n        --lr-scale-rule "$LR_SCALE_RULE"\n        --lr-reference-tokens-per-step "$LR_REFERENCE_TOKENS"\n        --wwpgd-candidate-device "$CANDIDATE_DEVICE"\n        "${MODE_ARGS[@]}"\n      )\n      wwgpt run-multiseed "${COMMON[@]}" --dry-run \\\n        > "$VARIANT_ROOT/level${LEVEL}_resolved_execution.txt"\n      if [ "$RESUME" = "1" ]; then\n        COMMON+=(--resume)\n      fi\n      wwgpt run-multiseed "${COMMON[@]}"\n    done\n    wwgpt analyze-results "$VARIANT_ROOT" --analysis-plan "$ANALYSIS_PLAN"\n    python -m wwgpt.cross_level_analysis \\\n      --results-root "$VARIANT_ROOT" \\\n      --output-dir "$VARIANT_ROOT/cross_level_analysis" \\\n      --figures-dir "$VARIANT_ROOT/cross_level_analysis/figures"\n    if [ "$RUN_NOTEBOOKS" = "1" ]; then\n      export WWGPT_RESULTS_ROOT="$VARIANT_ROOT"\n      export WWGPT_NOTEBOOK_OUTPUT_DIR="$VARIANT_ROOT/notebook-analysis"\n      export WWGPT_ANALYSIS_PLAN="$ANALYSIS_PLAN"\n      export WWGPT_BASE_OPTIMIZER="$OPTIMIZER"\n      export WWGPT_LEVEL=""\n      export WWGPT_TOKEN_MULTIPLIER="$TOKEN_MULTIPLIER"\n      ./scripts/run_analysis_notebooks.sh\n    fi\n  done\ndone\n\necho "[local-mac] complete: $RESULTS_ROOT" >&2\n''',
    executable=True,
)

# The older helper is now a thin compatibility wrapper.
write(
    "scripts/run_level0_2_experiment.sh",
    '''#!/usr/bin/env bash\nset -euo pipefail\nSCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\nexec "$SCRIPT_DIR/run_local_mac_experiments.sh" "$@"\n''',
    executable=True,
)

# ---------------------------------------------------------------------------
# Focused regression tests.
# ---------------------------------------------------------------------------
write(
    "tests/test_local_mac_readiness.py",
    '''from __future__ import annotations\n\nfrom dataclasses import replace\nfrom pathlib import Path\n\nimport pandas as pd\nimport pytest\nimport torch\n\nfrom wwgpt.config import ModelConfig, TrainConfig, load_config\nfrom wwgpt.model import GPT\nfrom wwgpt.optim import (\n    build_optimizer_bundle,\n    learning_rate_scale_factor,\n    resolve_learning_rates,\n)\nfrom wwgpt.ww import (\n    ExternalWWTailConfigSpec,\n    build_stock_wwpgd_candidate,\n    resolve_candidate_execution_device,\n)\n\n\ndef test_standard_lr_scale_rules() -> None:\n    cfg = TrainConfig(batch_size=16, gradient_accumulation=1, lr_reference_tokens_per_step=4096)\n    assert learning_rate_scale_factor(cfg, 256) == pytest.approx(1.0)\n    linear = replace(cfg, batch_size=32, lr_scale_rule="linear_batch")\n    sqrt = replace(cfg, batch_size=32, lr_scale_rule="sqrt_batch")\n    assert learning_rate_scale_factor(linear, 256) == pytest.approx(2.0)\n    assert learning_rate_scale_factor(sqrt, 256) == pytest.approx(2.0**0.5)\n\n\ndef test_optimizer_groups_record_resolved_lr_scaling() -> None:\n    model = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=32, vocab_size=128))\n    cfg = TrainConfig(\n        batch_size=8, gradient_accumulation=2, learning_rate=1e-3,\n        lr_scale_rule="linear_batch", lr_reference_tokens_per_step=256,\n    )\n    bundle, _ = build_optimizer_bundle(model, cfg, "adamw")\n    resolution = resolve_learning_rates(cfg, 32)\n    assert resolution["lr_scale_factor"] == pytest.approx(2.0)\n    assert bundle.learning_rate_resolution == resolution\n    assert all(group["effective_base_lr"] == pytest.approx(2e-3) for group in bundle.optimizers[0].param_groups)\n\n\ndef test_unknown_matrix_lr_role_is_rejected() -> None:\n    with pytest.raises(ValueError, match="unknown matrix_lr_multipliers"):\n        TrainConfig(matrix_lr_multipliers={"typo_role": 1.0})\n\n\ndef test_auto_candidate_device_offloads_mps_and_xla(monkeypatch) -> None:\n    model = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=16, vocab_size=64))\n    monkeypatch.setattr("wwgpt.ww._model_device", lambda _model: torch.device("mps"))\n    assert resolve_candidate_execution_device(model, "auto") == "cpu"\n    monkeypatch.setattr("wwgpt.ww._model_device", lambda _model: torch.device("cuda"))\n    assert resolve_candidate_execution_device(model, "auto") == "live"\n\n\ndef test_cpu_candidate_offload_does_not_mutate_live_model(monkeypatch) -> None:\n    import wwgpt.ww as ww\n\n    model = GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=16, vocab_size=64))\n    before = {name: value.detach().clone() for name, value in model.state_dict().items()}\n\n    class DummyConfig:\n        pass\n\n    monkeypatch.setattr(ww, "_external_config_object", lambda *_args, **_kwargs: DummyConfig())\n    monkeypatch.setattr(ww, "_assert_stock_wwpgd_api", lambda *_args, **_kwargs: None)\n    monkeypatch.setattr(\n        ww, "_external_wwpgd_module",\n        lambda: type("Module", (), {"ww_pgd_project": object()})(),\n    )\n\n    def fake_run(execution_model, _config, **_kwargs):\n        rows = []\n        with torch.no_grad():\n            for name, weight in ww.projected_matrix_modules(execution_model):\n                weight.add_(0.001)\n                rows.append({"longname": name, "alpha": 2.5, "D": 0.05, "xmin": 0.1, "detX_num": 8, "num_evals": weight.shape[0]})\n        return {"ww_logs": [pd.DataFrame(rows)], "diagnostic_logs": [], "native_internal_diagnostics": False}\n\n    monkeypatch.setattr(ww, "run_pip_wwpgd_candidate", fake_run)\n    candidate = build_stock_wwpgd_candidate(\n        model, cfg=ExternalWWTailConfigSpec(candidate_device="cpu")\n    )\n    assert candidate.candidate_offloaded is True\n    assert candidate.candidate_execution_device == "cpu"\n    assert any(candidate.stock_candidate_changed.values())\n    for name, value in model.state_dict().items():\n        assert torch.equal(value, before[name])\n\n\ndef test_level_configs_declare_safe_local_defaults() -> None:\n    for level in (0, 1, 2):\n        cfg = load_config(Path(f"configs/level{level}_adaptive_alpha.yaml"), level)\n        assert cfg.train.lr_scale_rule == "fixed"\n        assert cfg.train.lr_reference_tokens_per_step == 4096\n        assert cfg.wwpgd.candidate_device == "auto"\n''',
)

# ---------------------------------------------------------------------------
# CI additions: real pip-package smoke and macOS installation/readiness.
# ---------------------------------------------------------------------------
ci = read(".github/workflows/ci.yml")
if "real-package-level-smoke:" not in ci:
    ci += '''\n\n  real-package-level-smoke:\n    name: real pip WWPGD Level 0-2 smoke\n    runs-on: ubuntu-latest\n    timeout-minutes: 30\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: '3.11'\n      - name: Install pip dependencies\n        run: |\n          python -m pip install --upgrade pip\n          python -m pip install --no-cache-dir -e .\n      - name: Run real WeightWatcher and WWPGD smoke for all levels and optimizers\n        run: |\n          wwgpt local-readiness \\\n            --device cpu \\\n            --levels 0,1,2 \\\n            --optimizers adamw,stableadamw,muon \\\n            --output artifacts/real-package-readiness\n      - uses: actions/upload-artifact@v4\n        if: always()\n        with:\n          name: real-package-readiness\n          path: artifacts/real-package-readiness/**\n          if-no-files-found: error\n\n  macos-local-readiness:\n    name: macOS local workflow readiness\n    runs-on: macos-14\n    timeout-minutes: 30\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: '3.11'\n      - name: Install pip dependencies\n        run: |\n          python -m pip install --upgrade pip\n          python -m pip install --no-cache-dir -e .\n      - name: Validate macOS CPU workflow with real packages\n        run: |\n          wwgpt local-readiness \\\n            --device cpu \\\n            --levels 0,1,2 \\\n            --optimizers adamw \\\n            --output artifacts/macos-readiness\n      - name: Report MPS availability\n        run: |\n          python - <<'PY'\n          import json\n          from pathlib import Path\n          import torch\n          report = {\n              "mps_built": bool(torch.backends.mps.is_built()),\n              "mps_available": bool(torch.backends.mps.is_available()),\n              "note": "GitHub-hosted macOS may not expose an Apple GPU; local-readiness uses CPU-offloaded WWPGD candidates on MPS."\n          }\n          path = Path("artifacts/macos-readiness/mps-availability.json")\n          path.parent.mkdir(parents=True, exist_ok=True)\n          path.write_text(json.dumps(report, indent=2) + "\\n")\n          PY\n      - uses: actions/upload-artifact@v4\n        if: always()\n        with:\n          name: macos-local-readiness\n          path: artifacts/macos-readiness/**\n          if-no-files-found: error\n'''
write(".github/workflows/ci.yml", ci)

print("Applied phase-one local Mac readiness changes")
