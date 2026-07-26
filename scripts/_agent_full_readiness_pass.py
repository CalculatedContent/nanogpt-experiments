#!/usr/bin/env python3
"""Apply the final local-readiness integration patch.

This one-time helper modifies only CalculatedContent/nanogpt-experiments.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def replace_once(path: str, old: str, new: str) -> None:
    content = read(path)
    if old not in content:
        raise RuntimeError(f"expected text not found in {path}: {old[:100]!r}")
    write(path, content.replace(old, new, 1))


def regex_once(path: str, pattern: str, replacement: str) -> None:
    content = read(path)
    updated, count = re.subn(pattern, replacement, content, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"expected one regex replacement in {path}, got {count}: {pattern}")
    write(path, updated)


# ---------------------------------------------------------------------------
# Fix and complete the readiness module.
# ---------------------------------------------------------------------------
replace_once(
    "src/wwgpt/readiness.py",
    "from dataclasses import asdict, replace",
    "from dataclasses import replace",
)
replace_once(
    "src/wwgpt/readiness.py",
    "from wwgpt.device import device_summary, precision_policy, resolve_device, synchronize_device\nfrom wwgpt.model import GPT\nfrom wwgpt.optimizers import (\n    apply_lr_schedule,\n    build_optimizer_bundle,\n    optimizer_fingerprint,\n    optimizer_step,\n    resolve_warmup_steps,\n)",
    "from wwgpt.device import (\n    device_summary,\n    optimizer_step,\n    precision_policy,\n    resolve_device,\n    synchronize_device,\n)\nfrom wwgpt.model import GPT\nfrom wwgpt.optim import (\n    apply_lr_schedule,\n    build_optimizer_bundle,\n    optimizer_fingerprint,\n    resolve_warmup_steps,\n)",
)

# ---------------------------------------------------------------------------
# Config: explicit, opt-in token-batch LR scaling and checkpoint retention.
# ---------------------------------------------------------------------------
replace_once("src/wwgpt/config.py", "from __future__ import annotations\n", "from __future__ import annotations\n\nimport math\n")
replace_once(
    "src/wwgpt/config.py",
    'VALID_EXTENSIONS = {"none", "wwpgd", "measurement_only", "norm_matched_sham", "delayed_onset"}\n',
    'VALID_EXTENSIONS = {"none", "wwpgd", "measurement_only", "norm_matched_sham", "delayed_onset"}\n'
    'VALID_PARAMETER_ROLES = {\n'
    '    "token_embedding", "position_embedding", "attention_key", "attention_query",\n'
    '    "attention_value", "attention_projection", "mlp_input", "mlp_output",\n'
    '    "block_layernorm", "final_layernorm", "lm_head", "other", "block_other",\n'
    '}\n'
    'LR_SCALE_RULES = {"none", "linear", "sqrt"}\n',
)
replace_once(
    "src/wwgpt/config.py",
    "    matrix_lr_multipliers: dict[str, float] = field(default_factory=dict)\n",
    "    matrix_lr_multipliers: dict[str, float] = field(default_factory=dict)\n"
    "    # Opt-in global-batch/token scaling. The default preserves the standard\n"
    "    # fixed nanoGPT learning rate; linear and sqrt are explicit ablations.\n"
    "    lr_scale_rule: str = \"none\"\n"
    "    lr_reference_tokens_per_step: int | None = None\n"
    "    lr_scale_max_factor: float | None = None\n"
    "    # Number of resumable checkpoint_step_*.pt files retained. Zero keeps all.\n"
    "    checkpoint_keep_last: int = 2\n",
)
replace_once(
    "src/wwgpt/config.py",
    '        if self.layer_lr not in {"flat", "llrd", "manual"}:\n            raise ValueError(f"unknown layer_lr {self.layer_lr}")\n',
    '        if self.layer_lr not in {"flat", "llrd", "manual"}:\n'
    '            raise ValueError(f"unknown layer_lr {self.layer_lr}")\n'
    '        if self.lr_scale_rule not in LR_SCALE_RULES:\n'
    '            raise ValueError(f"unknown lr_scale_rule {self.lr_scale_rule}")\n'
    '        if self.lr_scale_rule != "none" and (\n'
    '            self.lr_reference_tokens_per_step is None\n'
    '            or self.lr_reference_tokens_per_step < 1\n'
    '        ):\n'
    '            raise ValueError(\n'
    '                "lr_reference_tokens_per_step must be positive when LR scaling is enabled"\n'
    '            )\n'
    '        if self.lr_scale_max_factor is not None and self.lr_scale_max_factor <= 0:\n'
    '            raise ValueError("lr_scale_max_factor must be positive when supplied")\n'
    '        if self.checkpoint_keep_last < 0:\n'
    '            raise ValueError("checkpoint_keep_last must be nonnegative")\n'
    '        unknown_roles = sorted(set(self.matrix_lr_multipliers) - VALID_PARAMETER_ROLES)\n'
    '        if unknown_roles:\n'
    '            raise ValueError(f"unknown matrix LR role(s): {unknown_roles}")\n'
    '        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in self.matrix_lr_multipliers.values()):\n'
    '            raise ValueError("matrix LR multipliers must be finite and positive")\n',
)

# ---------------------------------------------------------------------------
# Optimizers: apply explicit linear/sqrt token-batch scaling to every base
# optimizer while preserving flat/LLRD/manual parameter-group multipliers.
# ---------------------------------------------------------------------------
replace_once(
    "src/wwgpt/optim.py",
    "    implementation_versions: dict[str, str]\n",
    "    implementation_versions: dict[str, str]\n    learning_rate_resolution: dict[str, Any] | None = None\n",
)
optimizer_helper = '''\n\ndef resolve_learning_rate_scaling(model: nn.Module, cfg: TrainConfig) -> dict[str, Any]:\n    \"\"\"Resolve an explicit token-per-step LR rule.\n\n    ``none`` is the standard fixed-LR baseline. ``linear`` and ``sqrt`` are\n    opt-in global-batch scaling ablations; neither is silently enabled.\n    \"\"\"\n    block_size = int(getattr(getattr(model, "cfg", None), "block_size", 1))\n    tokens_per_step = int(cfg.batch_size * cfg.gradient_accumulation * block_size)\n    reference = cfg.lr_reference_tokens_per_step\n    if cfg.lr_scale_rule == "none":\n        factor = 1.0\n    else:\n        if reference is None or reference < 1:\n            raise ValueError("LR scaling requires a positive reference token batch")\n        ratio = tokens_per_step / float(reference)\n        factor = ratio if cfg.lr_scale_rule == "linear" else math.sqrt(ratio)\n        if cfg.lr_scale_max_factor is not None:\n            factor = min(factor, float(cfg.lr_scale_max_factor))\n    if not math.isfinite(factor) or factor <= 0:\n        raise ValueError("resolved LR scale factor must be finite and positive")\n    return {\n        "rule": cfg.lr_scale_rule,\n        "tokens_per_optimizer_step": tokens_per_step,\n        "reference_tokens_per_optimizer_step": reference,\n        "scale_factor": factor,\n        "maximum_scale_factor": cfg.lr_scale_max_factor,\n        "default_rule_enabled": cfg.lr_scale_rule == "none",\n    }\n'''
replace_once(
    "src/wwgpt/optim.py",
    "\ndef build_optimizer_bundle(model: nn.Module, cfg: TrainConfig, base_optimizer: str) -> tuple[OptimizerBundle, float]:\n",
    optimizer_helper + "\ndef build_optimizer_bundle(model: nn.Module, cfg: TrainConfig, base_optimizer: str) -> tuple[OptimizerBundle, float]:\n",
)
regex_once(
    "src/wwgpt/optim.py",
    r"def build_optimizer_bundle\(model: nn\.Module, cfg: TrainConfig, base_optimizer: str\) -> tuple\[OptimizerBundle, float\]:.*?    raise ValueError\(f\"cannot construct requested optimizer \{base_optimizer\}: unknown optimizer\"\)",
    '''def build_optimizer_bundle(model: nn.Module, cfg: TrainConfig, base_optimizer: str) -> tuple[OptimizerBundle, float]:\n    base_optimizer = "stableadamw" if base_optimizer == "stable_adamw" else base_optimizer\n    lr_resolution = resolve_learning_rate_scaling(model, cfg)\n    factor = float(lr_resolution["scale_factor"])\n    resolved_adamw_lr = cfg.learning_rate * factor\n    resolved_muon_lr = cfg.muon_learning_rate * factor\n    resolved_stable_lr = cfg.stable_learning_rate * factor\n    lr_resolution.update({\n        "configured_adamw_learning_rate": cfg.learning_rate,\n        "resolved_adamw_learning_rate": resolved_adamw_lr,\n        "configured_muon_learning_rate": cfg.muon_learning_rate,\n        "resolved_muon_learning_rate": resolved_muon_lr,\n        "configured_stable_learning_rate": cfg.stable_learning_rate,\n        "resolved_stable_learning_rate": resolved_stable_lr,\n    })\n    if base_optimizer == "adamw":\n        groups, gamma = build_param_groups(model, resolved_adamw_lr, cfg.weight_decay, cfg)\n        opt = torch.optim.AdamW(groups, betas=cfg.betas, eps=cfg.epsilon)\n        return OptimizerBundle(\n            "adamw", [opt], [("adamw", opt)],\n            {"adamw": f"{ADAMW_IMPLEMENTATION}:{torch.__version__}"}, lr_resolution\n        ), gamma\n    if base_optimizer == "stableadamw":\n        try:\n            from optimi import StableAdamW\n        except Exception as exc:\n            raise RuntimeError(\n                "cannot construct requested optimizer stableadamw: install the "\n                "'torch-optimi' package providing optimi.StableAdamW"\n            ) from exc\n        try:\n            optimi_version = metadata.version("torch-optimi")\n        except metadata.PackageNotFoundError:\n            try:\n                optimi_version = metadata.version("optimi")\n            except metadata.PackageNotFoundError:\n                optimi_version = "installed-version-unknown"\n        groups, gamma = build_param_groups(model, resolved_stable_lr, cfg.weight_decay, cfg)\n        opt = StableAdamW(\n            groups, lr=resolved_stable_lr, betas=cfg.stable_betas,\n            eps=cfg.stable_epsilon, weight_decay=0.0, triton=cfg.stable_triton\n        )\n        return OptimizerBundle(\n            "stableadamw", [opt], [("stableadamw", opt)],\n            {"stableadamw": f"{STABLEADAMW_IMPLEMENTATION}:{optimi_version}"},\n            lr_resolution,\n        ), gamma\n    if base_optimizer == "muon":\n        matrix_names = muon_parameter_names(model)\n        auxiliary_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad} - matrix_names\n        muon_groups, gamma = build_param_groups(\n            model, resolved_muon_lr, cfg.weight_decay, cfg, include_names=matrix_names\n        )\n        auxiliary_groups, gamma_aux = build_param_groups(\n            model, resolved_adamw_lr, cfg.weight_decay, cfg, include_names=auxiliary_names\n        )\n        muon = Muon(\n            muon_groups, lr=resolved_muon_lr, momentum=cfg.muon_momentum,\n            newton_schulz_steps=cfg.newton_schulz_steps, weight_decay=0.0\n        )\n        auxiliary = torch.optim.AdamW(auxiliary_groups, betas=cfg.betas, eps=cfg.epsilon)\n        versions = {\n            "muon": MUON_IMPLEMENTATION_VERSION,\n            "muon_aux_adamw": f"{ADAMW_IMPLEMENTATION}:{torch.__version__}",\n        }\n        return OptimizerBundle(\n            "muon", [muon, auxiliary],\n            [("muon", muon), ("muon_aux_adamw", auxiliary)],\n            versions, lr_resolution,\n        ), gamma or gamma_aux\n    raise ValueError(f"cannot construct requested optimizer {base_optimizer}: unknown optimizer")''',
)
replace_once(
    "src/wwgpt/optim.py",
    '        "parameter_groups": tuple(groups),\n    }\n',
    '        "parameter_groups": tuple(groups),\n        "learning_rate_resolution": dict(bundle.learning_rate_resolution or {}),\n    }\n',
)

# ---------------------------------------------------------------------------
# WWPGD adapter: MPS/XLA safely use a detached CPU model copy for the external
# WeightWatcher/SVD candidate computation. Live model and optimizer stay put.
# ---------------------------------------------------------------------------
replace_once("src/wwgpt/ww.py", "import json\nimport hashlib", "import json\nimport hashlib\nimport copy")
replace_once(
    "src/wwgpt/ww.py",
    "    internal_diagnostics: list[dict[str, object]] = dataclasses.field(default_factory=list)\n    stock_commit: str = WWPGD_COMMIT\n",
    "    internal_diagnostics: list[dict[str, object]] = dataclasses.field(default_factory=list)\n"
    "    candidate_compute_device: str = \"live\"\n"
    "    stock_commit: str = WWPGD_COMMIT\n",
)
replace_once(
    "src/wwgpt/ww.py",
    "    originals = {name: w.detach().clone() for name, w in projected_matrix_modules(model)}\n    start = time.perf_counter()\n    candidates: dict[str, torch.Tensor] = {}\n    result: dict[str, object] = {}\n    try:\n        with torch.no_grad():\n            result = run_pip_wwpgd_candidate(\n                model, _external_config_object(ww_pgd_module, full_cfg),\n                epoch=event_index, num_epochs=max(event_index + 1, 1),\n                global_step=actual_step, layer_selector=selector,\n            )\n            candidates = {name: w.detach().clone() for name, w in projected_matrix_modules(model)}\n    finally:\n        with torch.no_grad():\n            for name, weight in projected_matrix_modules(model):\n                weight.copy_(originals[name].to(weight.device, dtype=weight.dtype))\n                if not torch.equal(weight.detach().cpu(), originals[name].cpu()):\n                    raise RuntimeError(f\"failed to restore original WW_PGD weight bitwise for {name}\")\n",
    "    originals = {name: w.detach().clone() for name, w in projected_matrix_modules(model)}\n"
    "    device_types = {weight.device.type for _name, weight in projected_matrix_modules(model)}\n"
    "    use_cpu_copy = bool(device_types & {\"mps\", \"xla\"})\n"
    "    working_model = copy.deepcopy(model).cpu() if use_cpu_copy else model\n"
    "    candidate_compute_device = \"cpu_copy\" if use_cpu_copy else next(iter(device_types), \"cpu\")\n"
    "    start = time.perf_counter()\n"
    "    candidates: dict[str, torch.Tensor] = {}\n"
    "    result: dict[str, object] = {}\n"
    "    try:\n"
    "        with torch.no_grad():\n"
    "            result = run_pip_wwpgd_candidate(\n"
    "                working_model, _external_config_object(ww_pgd_module, full_cfg),\n"
    "                epoch=event_index, num_epochs=max(event_index + 1, 1),\n"
    "                global_step=actual_step, layer_selector=selector,\n"
    "            )\n"
    "            candidates = {\n"
    "                name: weight.detach().cpu().clone()\n"
    "                for name, weight in projected_matrix_modules(working_model)\n"
    "            }\n"
    "    finally:\n"
    "        if working_model is model:\n"
    "            with torch.no_grad():\n"
    "                for name, weight in projected_matrix_modules(model):\n"
    "                    weight.copy_(originals[name].to(weight.device, dtype=weight.dtype))\n"
    "                    if not torch.equal(weight.detach().cpu(), originals[name].cpu()):\n"
    "                        raise RuntimeError(\n"
    "                            f\"failed to restore original WW_PGD weight bitwise for {name}\"\n"
    "                        )\n",
)
replace_once(
    "src/wwgpt/ww.py",
    "            orig = originals[name].to(weight.device, dtype=weight.dtype)\n            cand = candidates[name].to(weight.device, dtype=weight.dtype)\n            disp = (cand - orig).float()\n            rel[name] = float(torch.linalg.norm(disp) / max(float(torch.linalg.norm(orig.float())), 1e-12))\n",
    "            orig = originals[name].detach().float().cpu()\n"
    "            cand = candidates[name].detach().float().cpu()\n"
    "            disp = cand - orig\n"
    "            rel[name] = float(\n"
    "                torch.linalg.norm(disp) / max(float(torch.linalg.norm(orig)), 1e-12)\n"
    "            )\n",
)
replace_once(
    "src/wwgpt/ww.py",
    '                "projection_runtime": runtime,\n                "warning_message": "private WWPGD internals are unsupported by the installed package",\n',
    '                "projection_runtime": runtime,\n                "candidate_compute_device": candidate_compute_device,\n                "warning_message": "private WWPGD internals are unsupported by the installed package",\n',
)
replace_once(
    "src/wwgpt/ww.py",
    "    return StockWWPGDCandidate(usable[0].copy(), originals, candidates, rel, changed, runtime, full_cfg, diagnostic_logs)\n",
    "    return StockWWPGDCandidate(\n"
    "        usable[0].copy(), originals, candidates, rel, changed, runtime, full_cfg,\n"
    "        diagnostic_logs, candidate_compute_device\n"
    "    )\n",
)
replace_once(
    "src/wwgpt/ww.py",
    '                "wwpgd_package": "ww_pgd", "wwpgd_commit": WWPGD_COMMIT, "target_alpha": cfg.target_alpha, "blend_eta": cfg.blend_eta, "cayley_eta": cfg.cayley_eta, "min_tail": cfg.min_tail,\n',
    '                "wwpgd_package": "ww_pgd", "wwpgd_commit": WWPGD_COMMIT, "candidate_compute_device": candidate.candidate_compute_device, "target_alpha": cfg.target_alpha, "blend_eta": cfg.blend_eta, "cayley_eta": cfg.cayley_eta, "min_tail": cfg.min_tail,\n',
)

# ---------------------------------------------------------------------------
# CLI: immutable resolved configs, role validation, complete optimizer/LR/WWPGD
# controls, readiness and health commands, and reusable data preparation.
# ---------------------------------------------------------------------------
replace_once(
    "src/wwgpt/cli.py",
    "    from wwgpt.config import load_config\n",
    "    from wwgpt.config import VALID_PARAMETER_ROLES, load_config\n",
)
replace_once(
    "src/wwgpt/cli.py",
    '        role, value = item.split("=", 1)\n        if not role or role in cli_matrix_roles:\n',
    '        role, value = item.split("=", 1)\n'
    '        if role not in VALID_PARAMETER_ROLES:\n'
    '            raise SystemExit(\n'
    '                f"unknown matrix LR role {role!r}; expected one of {sorted(VALID_PARAMETER_ROLES)}"\n'
    '            )\n'
    '        if not role or role in cli_matrix_roles:\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '        ("stable_triton", "stable_triton", train_updates),\n        ("dropout", "dropout", model_updates),\n',
    '        ("stable_triton", "stable_triton", train_updates),\n'
    '        ("lr_scale_rule", "lr_scale_rule", train_updates),\n'
    '        ("lr_reference_tokens_per_step", "lr_reference_tokens_per_step", train_updates),\n'
    '        ("lr_scale_max_factor", "lr_scale_max_factor", train_updates),\n'
    '        ("checkpoint_keep_last", "checkpoint_keep_last", train_updates),\n'
    '        ("dropout", "dropout", model_updates),\n',
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
    '    out = args.results_root / "cli_overrides_config.yaml"\n    out.parent.mkdir(parents=True, exist_ok=True)\n    out.write_text(yaml.safe_dump(asdict(cfg)))\n    return out\n',
    '    import hashlib\n'
    '    payload = yaml.safe_dump(asdict(cfg), sort_keys=True)\n'
    '    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]\n'
    '    out = args.results_root / "_resolved_configs" / f"level{args.level}_{digest}.yaml"\n'
    '    out.parent.mkdir(parents=True, exist_ok=True)\n'
    '    if out.exists() and out.read_text() != payload:\n'
    '        raise RuntimeError(f"resolved config hash collision at {out}")\n'
    '    if not out.exists():\n'
    '        out.write_text(payload)\n'
    '    return out\n',
)
replace_once(
    "src/wwgpt/cli.py",
    'pd.add_argument("--docs-file", type=Path, help="newline-delimited local documents for offline data-preparation tests"); pd.add_argument("--dry-run", action="store_true")',
    'pd.add_argument("--docs-file", type=Path, help="newline-delimited local documents for offline data-preparation tests"); pd.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True); pd.add_argument("--dry-run", action="store_true")',
)
replace_once(
    "src/wwgpt/cli.py",
    '    for parser in (rm, rt):\n        parser.add_argument("--learning-rate", type=float)\n',
    '    # Canonical trials expose the same baseline scheduling controls as run-multiseed.\n'
    '    rt.add_argument("--batch-size", type=int); rt.add_argument("--gradient-accumulation", type=int)\n'
    '    rt.add_argument("--weight-decay", type=float); rt.add_argument("--grad-clip", type=float)\n'
    '    rt.add_argument("--eval-batches", type=int); rt.add_argument("--dropout", type=float)\n'
    '    rt.add_argument("--lr-schedule", choices=["constant","warmup_cosine","warmup_linear"])\n'
    '    rt.add_argument("--warmup-ratio", type=float); rt.add_argument("--warmup-steps", type=int)\n'
    '    rt.add_argument("--lr-decay-steps", type=int); rt.add_argument("--min-lr-ratio", type=float)\n'
    '    rt.add_argument("--layer-lr", choices=["flat","llrd","manual"])\n'
    '    rt.add_argument("--llrd-gamma", type=float); rt.add_argument("--llrd-min-multiplier", type=float)\n'
    '    rt.add_argument("--max-train-tokens", type=int)\n'
    '    for parser in (rm, rt):\n'
    '        parser.add_argument("--learning-rate", type=float)\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '        parser.add_argument("--stable-triton", action=argparse.BooleanOptionalAction, default=None)\n',
    '        parser.add_argument("--stable-triton", action=argparse.BooleanOptionalAction, default=None)\n'
    '        parser.add_argument("--lr-scale-rule", choices=["none", "linear", "sqrt"])\n'
    '        parser.add_argument("--lr-reference-tokens-per-step", type=int)\n'
    '        parser.add_argument("--lr-scale-max-factor", type=float)\n'
    '        parser.add_argument("--checkpoint-keep-last", type=int)\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '        parser.add_argument("--wwpgd-log-every-fast-step", action=argparse.BooleanOptionalAction, default=None)\n',
    '        parser.add_argument("--wwpgd-log-every-fast-step", action=argparse.BooleanOptionalAction, default=None)\n'
    '        parser.add_argument("--wwpgd-dose-schedule", choices=["bounded_refresh_fraction", "fixed_per_step_gain"])\n'
    '        parser.add_argument("--wwpgd-max-endpoint-fraction-per-refresh", type=float)\n'
    '        parser.add_argument("--wwpgd-max-cumulative-relative-change-per-refresh", type=float)\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '    pl=sub.add_parser("plan-scaling"); pl.add_argument("--params", type=int); pl.add_argument("--level", type=int); pl.add_argument("--token-multiplier", type=int, required=True); pl.add_argument("--available-tokens", type=int, required=True); pl.add_argument("--batch-size", type=int, default=8); pl.add_argument("--block-size", type=int, default=256); pl.add_argument("--grad-accum", type=int, default=1)\n',
    '    ready=sub.add_parser("readiness-check", help="run real pip-installed WeightWatcher/WWPGD and optimizer checks on synthetic tokens"); ready.add_argument("--output", type=Path, required=True); ready.add_argument("--device", default="auto"); ready.add_argument("--levels", default="0,1,2"); ready.add_argument("--optimizers", default="adamw,muon,stableadamw"); ready.add_argument("--precision")\n'
    '    health=sub.add_parser("health-report", help="write run_health.json/csv for all completed runs"); health.add_argument("--results-root", type=Path, required=True); health.add_argument("--output-dir", type=Path)\n'
    '    pl=sub.add_parser("plan-scaling"); pl.add_argument("--params", type=int); pl.add_argument("--level", type=int); pl.add_argument("--token-multiplier", type=int, required=True); pl.add_argument("--available-tokens", type=int, required=True); pl.add_argument("--batch-size", type=int, default=8); pl.add_argument("--block-size", type=int, default=256); pl.add_argument("--grad-accum", type=int, default=1)\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '        print(prepare_data_for_mode(args.data_root, args.level, args.token_multiplier, _resolve_config_path(args), docs=docs, min_validation_tokens=1 if docs is not None else 100_000).root, flush=True)\n',
    '        print(prepare_data_for_mode(args.data_root, args.level, args.token_multiplier, _resolve_config_path(args), docs=docs, min_validation_tokens=1 if docs is not None else 100_000, reuse_existing=args.reuse_existing).root, flush=True)\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '        if args.config is not None and Path(args.config).name == "cli_overrides_config.yaml":\n            args.profile = None\n',
    '        if args.config is not None and Path(args.config).parent.name == "_resolved_configs":\n            args.profile = None\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '        if args.config is not None and Path(args.config).name == "cli_overrides_config.yaml":\n            args.profile = None\n',
    '        if args.config is not None and Path(args.config).parent.name == "_resolved_configs":\n            args.profile = None\n',
)
replace_once(
    "src/wwgpt/cli.py",
    '    elif args.cmd=="run-notebooks":\n',
    '    elif args.cmd=="readiness-check":\n'
    '        from wwgpt.readiness import run_readiness_check\n'
    '        report = run_readiness_check(args.output, device=args.device, levels=args.levels, optimizers=args.optimizers, precision=args.precision)\n'
    '        print(json.dumps(report, indent=2, sort_keys=True, default=str))\n'
    '        if not report.get("ready"):\n'
    '            raise SystemExit(1)\n'
    '    elif args.cmd=="health-report":\n'
    '        from wwgpt.run_health import write_experiment_health\n'
    '        frame = write_experiment_health(args.results_root, args.output_dir)\n'
    '        print(frame.to_string(index=False))\n'
    '        if not frame.empty and frame.status.eq("ERROR").any():\n'
    '            raise SystemExit(1)\n'
    '    elif args.cmd=="run-notebooks":\n',
)

# ---------------------------------------------------------------------------
# Data reuse: exact immutable prepared identities are reused by default.
# ---------------------------------------------------------------------------
replace_once(
    "src/wwgpt/data.py",
    'def prepare_data_for_mode(data_root: Path, level: int, token_multiplier: int,\n                          config_path: Path | None = None, docs: Iterable[str] | None = None,\n                          min_validation_tokens: int = 100_000) -> TokenData:\n',
    'def prepare_data_for_mode(data_root: Path, level: int, token_multiplier: int,\n'
    '                          config_path: Path | None = None, docs: Iterable[str] | None = None,\n'
    '                          min_validation_tokens: int = 100_000,\n'
    '                          reuse_existing: bool = True) -> TokenData:\n',
)
replace_once(
    "src/wwgpt/data.py",
    '    cfg = load_config(config_path, level)\n    if cfg.data_mode not in DATA_MODES:\n',
    '    cfg = load_config(config_path, level)\n'
    '    if reuse_existing and docs is None:\n'
    '        try:\n'
    '            existing = load_prepared_scientific_data(\n'
    '                data_root, level, token_multiplier, config_path\n'
    '            )\n'
    '            _log_prepare_progress(f"reusing prepared data identity at {existing.root}")\n'
    '            return existing\n'
    '        except FileNotFoundError:\n'
    '            pass\n'
    '    if cfg.data_mode not in DATA_MODES:\n',
)

# ---------------------------------------------------------------------------
# Checkpoint retention to avoid multi-gigabyte local Level 2 directories.
# ---------------------------------------------------------------------------
checkpoint_helper = '''\n\ndef _prune_checkpoint_history(run_dir: Path, keep_last: int) -> None:\n    if keep_last <= 0:\n        return\n    checkpoint_dir = Path(run_dir) / "checkpoints"\n    paths = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))\n    keep = {path.name for path in paths[-keep_last:]}\n    latest = checkpoint_dir / "latest.json"\n    if latest.exists():\n        try:\n            keep.add(json.loads(latest.read_text())["checkpoint"])\n        except Exception:\n            pass\n    for path in paths:\n        if path.name not in keep:\n            path.unlink()\n    inventory = checkpoint_dir / "checkpoint_inventory.csv"\n    if inventory.exists():\n        import csv\n        with inventory.open(newline="") as handle:\n            reader = csv.DictReader(handle)\n            fieldnames = list(reader.fieldnames or INVENTORY_FIELDS)\n            rows = [row for row in reader if row.get("checkpoint") in keep]\n        tmp = inventory.with_suffix(".csv.tmp")\n        with tmp.open("w", newline="") as handle:\n            writer = csv.DictWriter(handle, fieldnames=fieldnames)\n            writer.writeheader()\n            writer.writerows(rows)\n            handle.flush()\n            os.fsync(handle.fileno())\n        os.replace(tmp, inventory)\n'''
replace_once(
    "src/wwgpt/checkpointing.py",
    "\ndef save_checkpoint(run_dir: Path, state: dict):\n",
    checkpoint_helper + "\ndef save_checkpoint(run_dir: Path, state: dict):\n",
)
replace_once(
    "src/wwgpt/checkpointing.py",
    '    _append_inventory_atomic(ck/"checkpoint_inventory.csv", meta)\n    return path\n',
    '    _append_inventory_atomic(ck/"checkpoint_inventory.csv", meta)\n'
    '    keep_last = int(\n'
    '        ((state.get("resolved_config") or {}).get("train") or {}).get(\n'
    '            "checkpoint_keep_last", 0\n'
    '        )\n'
    '        or 0\n'
    '    )\n'
    '    _prune_checkpoint_history(run_dir, keep_last)\n'
    '    return path\n',
)

# ---------------------------------------------------------------------------
# Training: restore complete endpoint histories, retain only the current best
# validation model, add model norms, and write run health at completion.
# ---------------------------------------------------------------------------
replace_once("src/wwgpt/train.py", "import numpy as np\n", "import numpy as np\nimport pandas as pd\n")
train_helpers = '''\n\ndef _load_csv_records(path: Path) -> list[dict[str, object]]:\n    if not path.exists():\n        return []\n    try:\n        return pd.read_csv(path).to_dict("records")\n    except pd.errors.EmptyDataError:\n        return []\n\n\ndef _model_parameter_norms(model: GPT) -> dict[str, float]:\n    from wwgpt.ww import is_projected_layer\n    total_sq = 0.0\n    eligible_sq = 0.0\n    embedding_sq = 0.0\n    for name, parameter in model.named_parameters():\n        value = float(torch.sum(parameter.detach().float() ** 2).cpu())\n        total_sq += value\n        module_name = name.removesuffix(".weight").removesuffix(".bias")\n        if is_projected_layer(module_name):\n            eligible_sq += value\n        if name.startswith(("wte.", "wpe.")):\n            embedding_sq += value\n    return {\n        "model_parameter_norm": math.sqrt(total_sq),\n        "eligible_matrix_parameter_norm": math.sqrt(eligible_sq),\n        "embedding_parameter_norm": math.sqrt(embedding_sq),\n    }\n'''
replace_once(
    "src/wwgpt/train.py",
    "\ndef _gradient_norm(parameters) -> torch.Tensor:\n",
    train_helpers + "\ndef _gradient_norm(parameters) -> torch.Tensor:\n",
)
replace_once(
    "src/wwgpt/train.py",
    '        if hasattr(extension, "load_state_dict"):\n            extension.load_state_dict(loaded.get("wwpgd_adaptive_controller_state", loaded.get("wwpgd_state", {})))\n        if hasattr(extension, "internal_diagnostic_rows"):\n            extension.internal_diagnostic_rows = list(loaded.get("wwpgd_internal_diagnostic_rows", []))\n',
    '        if hasattr(extension, "load_state_dict"):\n'
    '            extension.load_state_dict(\n'
    '                loaded.get("wwpgd_adaptive_controller_state", loaded.get("wwpgd_state", {}))\n'
    '            )\n'
    '        for attribute, filename in (\n'
    '            ("measurement_rows", "wwpgd_endpoint_measurements.csv"),\n'
    '            ("relaxation_rows", "wwpgd_endpoint_relaxation.csv"),\n'
    '            ("fast_step_rows", "wwpgd_fast_control_steps.csv"),\n'
    '            ("internal_diagnostic_rows", "wwpgd_internal_diagnostics.csv"),\n'
    '        ):\n'
    '            if hasattr(extension, attribute):\n'
    '                setattr(extension, attribute, _load_csv_records(run_dir / filename))\n',
)
replace_once(
    "src/wwgpt/train.py",
    '                    "peak_memory": float(\n                        memory_stats(selected_device).get("max_allocated", 0.0)\n                    ),\n                    **timings,\n',
    '                    "peak_memory": float(\n'
    '                        memory_stats(selected_device).get("max_allocated", 0.0)\n'
    '                    ),\n'
    '                    **_model_parameter_norms(model),\n'
    '                    **timings,\n',
)
replace_once(
    "src/wwgpt/train.py",
    '            if validation_probe_loss < best_validation_loss:\n                best_validation_loss = validation_probe_loss\n                best_validation_step = step\n                torch.save(model.state_dict(), ckpt / f"best_val_step_{step:06d}_{seed}.pt")\n',
    '            if validation_probe_loss < best_validation_loss:\n'
    '                best_validation_loss = validation_probe_loss\n'
    '                best_validation_step = step\n'
    '                best_path = ckpt / f"best_val_step_{step:06d}_{seed}.pt"\n'
    '                torch.save(model.state_dict(), best_path)\n'
    '                for obsolete_best in ckpt.glob(f"best_val_step_*_{seed}.pt"):\n'
    '                    if obsolete_best != best_path:\n'
    '                        obsolete_best.unlink()\n',
)
replace_once(
    "src/wwgpt/train.py",
    '        "optimizer_fingerprint": json.loads(json.dumps(optimizer_fingerprint(bundle), default=str)),\n',
    '        "optimizer_fingerprint": json.loads(json.dumps(optimizer_fingerprint(bundle), default=str)),\n'
    '        "learning_rate_resolution": dict(bundle.learning_rate_resolution or {}),\n',
)
replace_once(
    "src/wwgpt/train.py",
    '    write_json(run_dir / "run_complete.json", common_complete)\n    _log_train_progress(\n',
    '    write_json(run_dir / "run_complete.json", common_complete)\n'
    '    from wwgpt.run_health import write_run_health\n'
    '    health = write_run_health(run_dir)\n'
    '    _log_train_progress(\n'
    '        f"run health status={health[\'status\']} errors={health[\'error_count\']} "\n'
    '        f"warnings={health[\'warning_count\']} output={run_dir}"\n'
    '    )\n'
    '    _log_train_progress(\n',
)

# ---------------------------------------------------------------------------
# Analysis command now emits all seed, WeightWatcher, WWPGD, generalization,
# cross-level, and run-health outputs in one invocation.
# ---------------------------------------------------------------------------
replace_once(
    "src/wwgpt/analysis.py",
    '    from wwgpt.cross_level_analysis import analyze_cross_level_effects\n    analyze_cross_level_effects(results_root, out, figures_dir=out / "figures")\n',
    '    from wwgpt.cross_level_analysis import analyze_cross_level_effects\n'
    '    from wwgpt.generalization_analysis import analyze_generalization_results\n'
    '    from wwgpt.run_health import write_experiment_health\n'
    '    from wwgpt.seed_analysis import analyze_seed_results\n'
    '    from wwgpt.weightwatcher_analysis import analyze_weightwatcher_results\n'
    '    from wwgpt.wwpgd_diagnostics_analysis import analyze_wwpgd_diagnostics\n'
    '    figure_dir = out / "figures"\n'
    '    analyze_seed_results(results_root, out, figures_dir=figure_dir)\n'
    '    analyze_weightwatcher_results(results_root, out, figures_dir=figure_dir)\n'
    '    analyze_generalization_results(results_root, out, figures_dir=figure_dir)\n'
    '    analyze_wwpgd_diagnostics(results_root, out, figures_dir=figure_dir)\n'
    '    analyze_cross_level_effects(results_root, out, figures_dir=figure_dir)\n'
    '    write_experiment_health(results_root, out)\n',
)

# ---------------------------------------------------------------------------
# End-to-end workflow and Mac setup.
# ---------------------------------------------------------------------------
write(
    "scripts/run_level0_2_experiment.sh",
    '''#!/usr/bin/env bash\nset -euo pipefail\n\nif [ "$#" -lt 2 ] || [ "$#" -gt 9 ]; then\n  echo "usage: $0 DATA_ROOT RESULTS_ROOT [TOKEN_MULTIPLIER=20] [DEVICE=auto] [SEEDS=1337,2027,4099] [OPTIMIZERS=adamw,muon,stableadamw] [LAYER_LR=flat] [WWPGD_MODE=adaptive] [FIXED_HARDNESS=0.25]" >&2\n  exit 2\nfi\n\nDATA_ROOT="$1"\nRESULTS_ROOT="$2"\nTOKEN_MULTIPLIER="${3:-20}"\nDEVICE="${4:-auto}"\nSEEDS="${5:-1337,2027,4099}"\nOPTIMIZERS="${6:-adamw,muon,stableadamw}"\nLAYER_LR="${7:-flat}"\nWWPGD_MODE="${8:-adaptive}"\nFIXED_HARDNESS="${9:-0.25}"\nANALYSIS_PLAN="configs/analysis_plan_exploratory.yaml"\nRESUME_FLAG=()\nif [ "${WWGPT_RESUME:-0}" = "1" ]; then RESUME_FLAG+=(--resume); fi\n\ncase "$LAYER_LR" in flat|llrd|manual) ;; *) echo "invalid LAYER_LR: $LAYER_LR" >&2; exit 2;; esac\ncase "$WWPGD_MODE" in adaptive|fixed) ;; *) echo "invalid WWPGD_MODE: $WWPGD_MODE" >&2; exit 2;; esac\n\nmkdir -p "$RESULTS_ROOT/readiness"\nwwgpt readiness-check \\\n  --output "$RESULTS_ROOT/readiness" \\\n  --device "$DEVICE" \\\n  --levels 0,1,2 \\\n  --optimizers "$OPTIMIZERS"\n\nIFS=',' read -r -a OPT_ARRAY <<< "$OPTIMIZERS"\nfor LEVEL in 0 1 2; do\n  CONFIG="configs/level${LEVEL}_adaptive_alpha.yaml"\n  echo "[level0-2] prepare/reuse Level ${LEVEL}" >&2\n  wwgpt prepare-data \\\n    --level "$LEVEL" --config "$CONFIG" --data-root "$DATA_ROOT" \\\n    --token-multiplier "$TOKEN_MULTIPLIER" --reuse-existing\n\n  for OPTIMIZER in "${OPT_ARRAY[@]}"; do\n    EXTRA=(--layer-lr "$LAYER_LR")\n    if [ "$WWPGD_MODE" = "fixed" ]; then\n      EXTRA+=(--wwpgd-adaptive-mode uniform --wwpgd-alpha-max-hardness "$FIXED_HARDNESS")\n    else\n      EXTRA+=(--wwpgd-adaptive-mode alpha_distance)\n    fi\n    DRY="$RESULTS_ROOT/level${LEVEL}_${OPTIMIZER}_${LAYER_LR}_${WWPGD_MODE}_resolved.json"\n    wwgpt run-multiseed \\\n      --level "$LEVEL" --config "$CONFIG" --analysis-plan "$ANALYSIS_PLAN" \\\n      --data-root "$DATA_ROOT" --results-root "$RESULTS_ROOT" \\\n      --token-multiplier "$TOKEN_MULTIPLIER" --seeds "$SEEDS" \\\n      --optimizer "$OPTIMIZER" --extensions none,wwpgd --device "$DEVICE" \\\n      "${EXTRA[@]}" --dry-run | tail -n +2 > "$DRY"\n    python -m json.tool "$DRY" >/dev/null\n\n    echo "[level0-2] Level ${LEVEL}, optimizer=${OPTIMIZER}, layer_lr=${LAYER_LR}, wwpgd=${WWPGD_MODE}" >&2\n    wwgpt run-multiseed \\\n      --level "$LEVEL" --config "$CONFIG" --analysis-plan "$ANALYSIS_PLAN" \\\n      --data-root "$DATA_ROOT" --results-root "$RESULTS_ROOT" \\\n      --token-multiplier "$TOKEN_MULTIPLIER" --seeds "$SEEDS" \\\n      --optimizer "$OPTIMIZER" --extensions none,wwpgd --device "$DEVICE" \\\n      "${EXTRA[@]}" "${RESUME_FLAG[@]}"\n  done\ndone\n\nwwgpt analyze-results "$RESULTS_ROOT" --profile scaling --analysis-plan "$ANALYSIS_PLAN"\nwwgpt health-report --results-root "$RESULTS_ROOT" --output-dir "$RESULTS_ROOT/analysis"\nwwgpt audit-experiment --experiment-root "$RESULTS_ROOT"\n\nexport WWGPT_RESULTS_ROOT="$RESULTS_ROOT"\nexport WWGPT_NOTEBOOK_OUTPUT_DIR="$RESULTS_ROOT/notebook-analysis"\nexport WWGPT_ANALYSIS_PLAN="$ANALYSIS_PLAN"\nexport WWGPT_BASE_OPTIMIZER="${OPT_ARRAY[0]}"\nexport WWGPT_RUN_ANALYSIS=0\nexport WWGPT_REUSE_ANALYSIS=1\n./scripts/run_analysis_notebooks.sh\n\necho "[level0-2] complete: results=$RESULTS_ROOT analysis=$RESULTS_ROOT/analysis notebooks=$RESULTS_ROOT/notebook-analysis" >&2\n''',
)
write(
    "scripts/macbook_setup_and_preflight.sh",
    '''#!/usr/bin/env bash\nset -euo pipefail\nPYTHON_BIN="${PYTHON_BIN:-python3.11}"\nVENV_DIR="${VENV_DIR:-.venv}"\nDEVICE="${WWGPT_DEVICE:-auto}"\nOUTPUT="${1:-$HOME/wwgpt-readiness}"\n\ncommand -v "$PYTHON_BIN" >/dev/null || { echo "$PYTHON_BIN is required (Python 3.10-3.12)" >&2; exit 1; }\nif [ ! -d "$VENV_DIR" ]; then "$PYTHON_BIN" -m venv "$VENV_DIR"; fi\nsource "$VENV_DIR/bin/activate"\npython -m pip install --upgrade pip setuptools wheel\npython -m pip install --no-cache-dir -e ".[notebooks]"\npython -m pip show weightwatcher\npython -m pip show ww-pgd\npython - <<'PY'\nimport platform, torch\nprint('platform:', platform.platform())\nprint('torch:', torch.__version__)\nprint('MPS built:', torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False)\nprint('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)\nPY\nmkdir -p "$OUTPUT"\nwwgpt readiness-check --output "$OUTPUT" --device "$DEVICE" --levels 0,1,2 --optimizers adamw,muon,stableadamw\necho "Mac readiness passed. Report: $OUTPUT/readiness_report.json"\n''',
)

# ---------------------------------------------------------------------------
# Regenerate substantive analysis notebooks around the tested Python modules.
# ---------------------------------------------------------------------------
PARAMETERS = [
    'RESULTS_ROOT = ""', 'OUTPUT_ROOT = ""', 'ANALYSIS_PLAN = ""', 'PROFILE = ""',
    'LEVEL = None', 'TOKEN_MULTIPLIER = None', 'BASE_OPTIMIZER = "adamw"',
    'STRICT = False', 'RUN_ANALYSIS = False', 'REUSE_EXISTING_ANALYSIS = True',
    'FIGURE_FORMAT = "png"', 'RANDOM_SEED = 1729',
]


def notebook(title: str, cells: list[tuple[str, str]]) -> str:
    content = [
        {"cell_type": "markdown", "id": "title", "metadata": {}, "source": [f"# {title}\n"]},
        {"cell_type": "code", "id": "parameters", "metadata": {"tags": ["parameters"]}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in PARAMETERS]},
    ]
    for index, (kind, source) in enumerate(cells, start=1):
        cell = {"cell_type": kind, "id": f"analysis-{index:02d}", "metadata": {}, "source": [line + "\n" for line in source.strip().splitlines()]}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        content.append(cell)
    return json.dumps(
        {
            "cells": content,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        indent=1,
    ) + "\n"

setup = '''from pathlib import Path\nimport json\nimport pandas as pd\nfrom wwgpt.notebook_support import resolve_notebook_parameters, validate_paths, package_provenance\nP = resolve_notebook_parameters(globals())\nvalidate_paths(P)\nprint("Resolved notebook parameters:", json.dumps(P.summary(), indent=2))\nprint("Package provenance:", json.dumps(package_provenance(), indent=2, default=str))'''
write(
    "notebooks/02_compare_single_level.ipynb",
    notebook(
        "02 Compare Single Level Across Seeds",
        [
            ("markdown", "Paired baseline-versus-WWPGD curves. Shaded bands are 95% Student-t confidence intervals; dashed envelopes are mean ± 2 sample standard deviations (Bollinger-style descriptive bands). Accuracy is next-token accuracy."),
            ("code", setup),
            ("code", '''from wwgpt.seed_analysis import analyze_seed_results\noutputs = analyze_seed_results(P.results_root, P.output_root / "tables", figures_dir=P.output_root / "figures", level=P.level, token_multiplier=P.token_multiplier, base_optimizer=P.base_optimizer)\nselected = outputs["selected_checkpoint_effects_by_seed.csv"]\nselected_summary = outputs["selected_checkpoint_effect_summary.csv"]\ncurve_summary = outputs["seed_learning_curve_summary.csv"]\npaired_curve_summary = outputs["paired_learning_curve_effect_summary.csv"]\ndisplay(selected.sort_values(["metric", "seed"]) if not selected.empty else selected)\ndisplay(selected_summary.sort_values("metric") if not selected_summary.empty else selected_summary)\ndisplay(paired_curve_summary.head(30))\nif P.strict and selected.empty: raise RuntimeError("no complete selected-checkpoint pairs")'''),
            ("markdown", "Effects are always WWPGD minus baseline within seed. Negative is favorable for loss/perplexity; positive is favorable for next-token accuracy. Seeds—not layers or evaluation events—are the replicates."),
        ],
    ),
)
write(
    "notebooks/03_weightwatcher_analysis.ipynb",
    notebook(
        "03 WeightWatcher Analysis Across Seeds",
        [
            ("markdown", "Scientific alpha comes only from nonrandomized `alpha_measurements.csv`. Randomized artifacts are used only for correlation-trap diagnostics; `detX_num` is never treated as a trap count."),
            ("code", setup),
            ("code", '''from wwgpt.weightwatcher_analysis import analyze_weightwatcher_results\noutputs = analyze_weightwatcher_results(P.results_root, P.output_root / "tables", figures_dir=P.output_root / "figures", level=P.level, token_multiplier=P.token_multiplier, base_optimizer=P.base_optimizer)\ndisplay(outputs["weightwatcher_alpha_coverage.csv"])\ndisplay(outputs["weightwatcher_alpha_seed_summary.csv"].head(40))\ndisplay(outputs["weightwatcher_exclusion_reasons.csv"])\ndisplay(outputs["weightwatcher_trap_seed_summary.csv"].head(40))\nif P.strict and outputs["weightwatcher_valid_alpha_rows.csv"].empty: raise RuntimeError("no valid projected WeightWatcher alpha rows")'''),
            ("markdown", "The figures report seed-level uncertainty, alpha distance from target, deadband occupancy, fit quality, stable rank, and actual randomized trap-layer fractions."),
        ],
    ),
)
write(
    "notebooks/05_overfitting_and_generalization.ipynb",
    notebook(
        "05 Overfitting and Generalization Across Seeds",
        [
            ("markdown", "This notebook compares train/validation/test behavior, late validation degradation, validation-loss AUC, alpha distance, trap fraction, and actual WWPGD dose. Associations are descriptive and not causal."),
            ("code", setup),
            ("code", '''from wwgpt.generalization_analysis import analyze_generalization_results\noutputs = analyze_generalization_results(P.results_root, P.output_root / "tables", figures_dir=P.output_root / "figures", level=P.level, token_multiplier=P.token_multiplier, base_optimizer=P.base_optimizer)\ndisplay(outputs["generalization_run_summary.csv"])\ndisplay(outputs["generalization_paired_effects_by_seed.csv"].sort_values(["metric", "seed"]) if not outputs["generalization_paired_effects_by_seed.csv"].empty else outputs["generalization_paired_effects_by_seed.csv"])\ndisplay(outputs["generalization_paired_effect_summary.csv"])\ndisplay(outputs["generalization_diagnostic_correlations.csv"])\nif P.strict and outputs["generalization_paired_effects_by_seed.csv"].empty: raise RuntimeError("no paired generalization results")'''),
            ("markdown", "Test results are evaluated once on the validation-selected checkpoint and never select the checkpoint."),
        ],
    ),
)
write(
    "notebooks/06_summary_report.ipynb",
    notebook(
        "06 Experiment Summary Report",
        [
            ("code", setup),
            ("code", '''from wwgpt.run_health import write_experiment_health\nhealth = write_experiment_health(P.results_root, P.output_root / "tables")\ntables = {}\nfor path in sorted((P.output_root / "tables").glob("*.csv")):\n    if path.name == "summary_report.csv" or path.stat().st_size == 0: continue\n    try: tables[path.name] = pd.read_csv(path)\n    except pd.errors.EmptyDataError: pass\nselected = tables.get("selected_checkpoint_effect_summary.csv", pd.DataFrame())\nerrors = int(health.status.eq("ERROR").sum()) if not health.empty and "status" in health else 0\nstatus = "experiment_invalid" if errors else "insufficient_data"\nif not selected.empty:\n    primary = selected[selected.metric.eq("test_loss")]\n    if not primary.empty:\n        mean = float(primary["mean"].mean())\n        ci_low = float(primary["t_ci_low"].min())\n        ci_high = float(primary["t_ci_high"].max())\n        if ci_high < 0: status = "WWPGD_better_descriptively"\n        elif ci_low > 0: status = "baseline_better_descriptively"\n        elif abs(mean) < 1e-12: status = "no_material_difference"\n        else: status = "mixed_result"\nsummary = pd.DataFrame([{"conclusion_status": status, "analysis_label": "exploratory unless an eligible frozen confirmatory plan computes its primary outcome", "health_error_runs": errors, "completed_health_runs": len(health), "available_tables": ";".join(sorted(tables)), "uncertainty": "95% t/bootstrap intervals and mean ± 2 SD descriptive bands", "replicate_unit": "random seed"}])\nsummary.to_csv(P.output_root / "tables" / "summary_report.csv", index=False)\ndisplay(summary); display(health)\nif P.strict and errors: raise RuntimeError("run health contains ERROR")'''),
            ("markdown", "The status is descriptive. It is not an efficacy or scaling-law claim unless the preregistered confirmatory workflow is independently eligible."),
        ],
    ),
)
write(
    "notebooks/07_wwpgd_diagnostics.ipynb",
    notebook(
        "07 WWPGD Dose, Endpoint, and Package Diagnostics",
        [
            ("markdown", "Native package internals are analyzed when exposed. Otherwise compatibility mode reports only WeightWatcher output, candidate movement, adaptive gain, endpoint motion, and dose caps; private midpoint/Cayley/TraceLog fields remain null."),
            ("code", setup),
            ("code", '''from wwgpt.wwpgd_diagnostics_analysis import analyze_wwpgd_diagnostics\noutputs = analyze_wwpgd_diagnostics(P.results_root, P.output_root / "tables", figures_dir=P.output_root / "figures", level=P.level, token_multiplier=P.token_multiplier, base_optimizer=P.base_optimizer)\ndisplay(outputs["wwpgd_diagnostic_capability.csv"])\ndisplay(outputs["wwpgd_diagnostic_health.csv"])\ndisplay(outputs["wwpgd_dose_by_layer.csv"].head(50))\ndisplay(outputs["wwpgd_endpoint_summary.csv"])\ndisplay(outputs["wwpgd_skip_reason_summary.csv"].head(50))\nhealth = outputs["wwpgd_diagnostic_health.csv"]\nif P.strict and not health.empty and health.status.eq("ERROR").any(): raise RuntimeError("WWPGD diagnostic health contains ERROR")'''),
            ("markdown", "A compatibility-only package can still verify that WWPGD ran and quantify the actual candidate/applied displacement. It cannot verify unexposed private internals, and the notebook does not reconstruct them."),
        ],
    ),
)

# ---------------------------------------------------------------------------
# Tests for the final readiness surfaces.
# ---------------------------------------------------------------------------
write(
    "tests/test_full_readiness_pass.py",
    '''from __future__ import annotations\n\nimport json\nfrom dataclasses import replace\nfrom pathlib import Path\n\nimport pandas as pd\nimport pytest\n\nfrom wwgpt.checkpointing import complete_test_checkpoint_state, save_checkpoint\nfrom wwgpt.config import ModelConfig, TrainConfig\nfrom wwgpt.model import GPT\nfrom wwgpt.optim import build_optimizer_bundle\nfrom wwgpt.seed_analysis import analyze_seed_results\nfrom wwgpt.weightwatcher_analysis import analyze_weightwatcher_results\nfrom wwgpt.generalization_analysis import analyze_generalization_results\nfrom wwgpt.wwpgd_diagnostics_analysis import analyze_wwpgd_diagnostics\n\n\ndef small_model() -> GPT:\n    return GPT(ModelConfig(n_layer=1, n_head=1, n_embd=64, block_size=8, vocab_size=32))\n\n\n@pytest.mark.parametrize(("rule", "expected"), [("none", 1.0), ("linear", 0.25), ("sqrt", 0.5)])\ndef test_explicit_token_batch_lr_scaling(rule: str, expected: float) -> None:\n    config = TrainConfig(\n        batch_size=2, gradient_accumulation=1, learning_rate=1e-3,\n        lr_scale_rule=rule, lr_reference_tokens_per_step=64 if rule != "none" else None,\n    )\n    bundle, _ = build_optimizer_bundle(small_model(), config, "adamw")\n    assert bundle.learning_rate_resolution["scale_factor"] == pytest.approx(expected)\n    assert bundle.optimizers[0].param_groups[0]["peak_lr"] == pytest.approx(1e-3 * expected)\n\n\ndef test_unknown_matrix_lr_role_is_rejected() -> None:\n    with pytest.raises(ValueError, match="unknown matrix LR role"):\n        TrainConfig(matrix_lr_multipliers={"typo_role": 1.0})\n\n\ndef test_checkpoint_retention(tmp_path: Path) -> None:\n    run = tmp_path / "run"\n    for step in range(1, 5):\n        save_checkpoint(\n            run,\n            complete_test_checkpoint_state(\n                current_step=step, next_step=step + 1,\n                resolved_config={"train": {"checkpoint_keep_last": 2}},\n            ),\n        )\n    assert len(list((run / "checkpoints").glob("checkpoint_step_*.pt"))) == 2\n    inventory = pd.read_csv(run / "checkpoints" / "checkpoint_inventory.csv")\n    assert len(inventory) == 2\n\n\ndef test_schema_v3_seed_and_diagnostic_analysis(tmp_path: Path) -> None:\n    root = Path("tests/fixtures/schema_v3_results/experiments/level_00/multiplier_1")\n    seed = analyze_seed_results(root, tmp_path / "seed", level=0, token_multiplier=1, base_optimizer="adamw", points=12)\n    assert (tmp_path / "seed" / "selected_checkpoint_effect_summary.csv").is_file()\n    assert "test_loss" in set(seed["selected_checkpoint_effects_by_seed.csv"].get("metric", []))\n    ww = analyze_weightwatcher_results(root, tmp_path / "ww", level=0, token_multiplier=1, base_optimizer="adamw")\n    assert (tmp_path / "ww" / "weightwatcher_alpha_coverage.csv").is_file()\n    gen = analyze_generalization_results(root, tmp_path / "gen", level=0, token_multiplier=1, base_optimizer="adamw")\n    assert (tmp_path / "gen" / "generalization_run_summary.csv").is_file()\n    diag = analyze_wwpgd_diagnostics(root, tmp_path / "diag", level=0, token_multiplier=1, base_optimizer="adamw")\n    assert (tmp_path / "diag" / "wwpgd_diagnostic_health.csv").is_file()\n    assert not diag["wwpgd_diagnostic_capability.csv"].empty\n\n\n@pytest.mark.parametrize("notebook", [\n    "02_compare_single_level.ipynb", "03_weightwatcher_analysis.ipynb",\n    "05_overfitting_and_generalization.ipynb", "06_summary_report.ipynb",\n    "07_wwpgd_diagnostics.ipynb",\n])\ndef test_analysis_notebooks_delegate_to_tested_modules(notebook: str) -> None:\n    content = json.loads((Path("notebooks") / notebook).read_text())\n    source = "\\n".join("".join(cell.get("source", [])) for cell in content["cells"])\n    assert "resolve_notebook_parameters" in source\n    assert "analyze_" in source or "write_experiment_health" in source\n''',
)

# ---------------------------------------------------------------------------
# Mac runbook and README appendix.
# ---------------------------------------------------------------------------
write(
    "docs/MAC_LOCAL_RUNBOOK.md",
    '''# Local macOS Level 0-2 Experiment Runbook\n\n## Installation and real preflight\n\nUse Python 3.10-3.12. The supported clean setup installs WeightWatcher and the unpinned WWPGD dependency through pip; no sibling checkout is used.\n\n```bash\n./scripts/macbook_setup_and_preflight.sh "$HOME/wwgpt-readiness"\n```\n\n`WWGPT_DEVICE=auto` selects MPS when available. Candidate generation automatically uses a detached CPU copy on MPS/XLA because the external projector performs an SVD; the live model and optimizer state remain on the selected accelerator. A readiness failure must be fixed before launching training.\n\n## Full paired Level 0-2 experiment\n\n```bash\nsource .venv/bin/activate\nbash scripts/run_level0_2_experiment.sh \\\n  "$HOME/wwgpt-data" \\\n  "$HOME/wwgpt-results" \\\n  20 \\\n  auto \\\n  1337,2027,4099 \\\n  adamw,muon,stableadamw \\\n  flat \\\n  adaptive\n```\n\nFor a fixed equal WWPGD controller strength across eligible layers:\n\n```bash\nbash scripts/run_level0_2_experiment.sh "$HOME/wwgpt-data" "$HOME/wwgpt-fixed" 20 auto 1337,2027,4099 adamw flat fixed 0.25\n```\n\nFor layerwise learning-rate ablations, replace `flat` with `llrd` or `manual`. `flat` remains the default nanoGPT baseline. Optional global token-batch LR scaling is explicit: `--lr-scale-rule linear|sqrt --lr-reference-tokens-per-step N`; it is never enabled silently.\n\nSet `WWGPT_RESUME=1` to continue compatible incomplete runs. Prepared data is reused by immutable identity. Only the two most recent resumable checkpoints and the current validation-best model are retained by default.\n\n## Outputs\n\nThe workflow writes raw append-only run artifacts, validation-selected train/validation/test loss, perplexity and next-token accuracy, run health, cross-level paired effects, WeightWatcher alpha/trap analysis, WWPGD endpoint/dose diagnostics, and seven executed Papermill notebooks. Confidence bands use the random seed as the replicate: 95% Student-t/bootstrap intervals plus mean ± 2 sample-standard-deviation descriptive bands.\n\nNo workflow can prove improvement in advance. The experiment is designed to measure whether WWPGD improves each base optimizer and whether the paired effect changes with model level.\n''',
)
readme = read("README.md")
appendix = '''\n\n## Complete local Mac readiness and Level 0-2 workflow\n\nRun `./scripts/macbook_setup_and_preflight.sh` before spending compute. It performs real pip-installed WeightWatcher/WWPGD candidate generation for Levels 0-2 and AdamW, Muon, and StableAdamW; checks flat/LLRD/manual parameter-group learning rates; and records the cadence-normalized adaptive schedules. On MPS/XLA, external candidate construction runs on a detached CPU model copy while the live model and optimizer remain on the accelerator.\n\nThe full paired workflow is:\n\n```bash\nbash scripts/run_level0_2_experiment.sh \\\n  "$HOME/wwgpt-data" "$HOME/wwgpt-results" 20 auto \\\n  1337,2027,4099 adamw,muon,stableadamw flat adaptive\n```\n\nUse `fixed 0.25` instead of `adaptive` for equal controller hardness on every eligible layer. Use `llrd` or `manual` instead of `flat` for explicit layerwise-LR ablations. The default fixed nanoGPT learning rate is not silently rescaled; opt-in `linear` and `sqrt` token-batch rules require an explicit reference token batch.\n\n`wwgpt analyze-results` now writes seed-level learning curves, 95% confidence intervals, mean ± 2-SD descriptive bands, validation-selected test loss/perplexity/next-token accuracy, WeightWatcher alpha and trap summaries, WWPGD package/endpoint/dose diagnostics, generalization analyses, cross-level effects, and machine-readable health reports. See `docs/MAC_LOCAL_RUNBOOK.md`.\n'''
if "## Complete local Mac readiness and Level 0-2 workflow" not in readme:
    write("README.md", readme + appendix)

print("full readiness source patch applied")
