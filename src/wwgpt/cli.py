from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from dataclasses import asdict, replace
import yaml

from wwgpt.analysis import analyze_results
from wwgpt.config import DEFAULT_SEEDS
from wwgpt.data import prepare_scientific_data
from wwgpt.scaling import PARAMETER_COUNT_CONVENTIONS, plan_budget, selected_parameter_count
from wwgpt.train import run_multiseed_scientific, run_canonical_trials, smoke
from wwgpt.strength_scan import run_strength_scan, parse_strengths
from wwgpt.strength_scan_analysis import analyze_strength_scan as analyze_strength_scan_cmd
from wwgpt.device import device_summary, save_device_manifest, run_device_preflight
from wwgpt.integrity import audit_experiment, audit_strength_scan
from wwgpt.checkpointing import inspect_checkpoint, validate_resume
from wwgpt.reproducibility import write_reproducibility_report


PROFILE_CONFIGS = {
    "scaling": Path("configs/default.yaml"),
    "reproduction_tiny": Path("configs/reproduction_tiny.yaml"),
    "reproduction_fineweb": Path("configs/reproduction_fineweb.yaml"),
}


def _resolve_config_path(args) -> Path | None:
    profile = getattr(args, "profile", None)
    config = getattr(args, "config", None)
    if profile:
        profile_path = PROFILE_CONFIGS[profile]
        if config is not None and Path(config) != Path("configs/default.yaml") and Path(config) != profile_path:
            raise SystemExit(f"--profile {profile} conflicts with explicit --config {config}; use one configuration source")
        return profile_path
    return config


def _resolved_config(args):
    from wwgpt.config import load_config

    return load_config(_resolve_config_path(args), args.level)


def _budget_summary(cfg, token_multiplier: int) -> dict[str, int]:
    from wwgpt.model import GPT
    from wwgpt.scaling import PARAMETER_COUNT_CONVENTIONS, plan_budget, selected_parameter_count

    report = GPT(cfg.model).parameter_report()
    param_count = selected_parameter_count(report, cfg.parameter_count_convention)
    old_total_count = selected_parameter_count(report, "total")
    budget = plan_budget(param_count, token_multiplier, cfg.train.batch_size, cfg.model.block_size, cfg.train.gradient_accumulation, 10**18)
    old_budget = plan_budget(old_total_count, token_multiplier, cfg.train.batch_size, cfg.model.block_size, cfg.train.gradient_accumulation, 10**18)
    return {
        "parameter_count_convention": cfg.parameter_count_convention,
        "parameter_count_convention_definition": PARAMETER_COUNT_CONVENTIONS[cfg.parameter_count_convention],
        "parameter_count": int(param_count),
        "selected_parameter_count": int(param_count),
        "old_total_parameter_count": int(old_total_count),
        "old_total_requested_tokens": int(old_budget.requested_tokens),
        "old_total_realized_tokens": int(old_budget.realized_tokens),
        "old_vs_selected_realized_token_delta": int(budget.realized_tokens - old_budget.realized_tokens),
        "token_multiplier": int(token_multiplier),
        "requested_tokens": int(budget.requested_tokens),
        "realized_tokens": int(budget.realized_tokens),
        "tokens_per_step": int(budget.tokens_per_step),
        "estimated_optimizer_steps": int(budget.steps),
        "sequence_count": int(budget.sequence_count),
        "optimizer_step_count": int(budget.optimizer_step_count),
        "realized_tokens_per_selected_parameter": float(budget.tokens_per_selected_parameter),
        "parameter_report": report.__dict__,
    }



def _level_multiplier_table(cfg, requested_level: int, requested_multiplier: int) -> list[dict[str, object]]:
    from dataclasses import replace
    from wwgpt.config import ladder
    from wwgpt.model import GPT

    rows = []
    for level, model_cfg in ladder().items():
        resolved_model = replace(model_cfg, vocab_size=cfg.model.vocab_size)
        report = GPT(resolved_model).parameter_report()
        for multiplier in cfg.token_multipliers:
            budget = plan_budget(selected_parameter_count(report, cfg.parameter_count_convention), multiplier, cfg.train.batch_size, resolved_model.block_size, cfg.train.gradient_accumulation, 10**18)
            rows.append({
                "level": level,
                "token_multiplier": multiplier,
                "is_requested_level_multiplier": level == requested_level and multiplier == requested_multiplier,
                "requested_tokens": budget.requested_tokens,
                "realized_tokens": budget.realized_tokens,
                "selected_parameter_count": budget.selected_parameter_count,
                "realized_tokens_per_selected_parameter": budget.tokens_per_selected_parameter,
                "sequence_count": budget.sequence_count,
                "optimizer_step_count": budget.optimizer_step_count,
                "parameter_report": report.__dict__,
            })
    return rows

def _print_resolved_execution(args, *, arms: list[str], seeds: list[int], trials: int, output_dirs: list[Path], dry_run: bool = False) -> None:
    import json
    from dataclasses import asdict

    cfg = _resolved_config(args)
    budget = _budget_summary(cfg, args.token_multiplier)
    payload = {
        "dry_run": dry_run,
        "profile": getattr(args, "profile", None),
        "config_path": str(_resolve_config_path(args)) if _resolve_config_path(args) is not None else None,
        "level": args.level,
        "token_multiplier": args.token_multiplier,
        "number_of_trials": trials,
        "number_of_arms": len(arms),
        "arms": arms,
        "levels": [args.level],
        "seeds": seeds,
        "token_budgets": budget,
        "scaling_law_accounting": {"selected_convention": budget["parameter_count_convention"], "definition": budget["parameter_count_convention_definition"], "comparison_old_total_vs_selected": {"old_total_parameter_count": budget["old_total_parameter_count"], "old_total_requested_tokens": budget["old_total_requested_tokens"], "old_total_realized_tokens": budget["old_total_realized_tokens"], "selected_parameter_count": budget["selected_parameter_count"], "selected_requested_tokens": budget["requested_tokens"], "selected_realized_tokens": budget["realized_tokens"]}},
        "level_multiplier_table": _level_multiplier_table(cfg, args.level, args.token_multiplier),
        "estimated_optimizer_steps": budget["estimated_optimizer_steps"],
        "output_directories": [str(p) for p in output_dirs],
        "dataset_revision": cfg.dataset_revision,
        "resolved_config": asdict(cfg),
    }
    print("Resolved execution configuration:")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _config_with_run_overrides(args):
    from wwgpt.config import load_config
    cfg = load_config(_resolve_config_path(args), args.level)
    model_updates = {}
    train_updates = {}
    for arg, key, target in [
        ("batch_size", "batch_size", train_updates),
        ("gradient_accumulation", "gradient_accumulation", train_updates),
        ("weight_decay", "weight_decay", train_updates),
        ("grad_clip", "grad_clip", train_updates),
        ("eval_batches", "eval_batches", train_updates),
        ("lr_schedule", "lr_schedule", train_updates),
        ("warmup_ratio", "warmup_ratio", train_updates),
        ("warmup_steps", "warmup_steps", train_updates),
        ("lr_decay_steps", "lr_decay_steps", train_updates),
        ("min_lr_ratio", "min_lr_ratio", train_updates),
        ("layer_lr", "layer_lr", train_updates),
        ("llrd_gamma", "llrd_gamma", train_updates),
        ("llrd_min_multiplier", "llrd_min_multiplier", train_updates),
        ("max_train_tokens", "max_train_tokens", train_updates),
        ("max_steps", "max_steps", train_updates),
        ("wwpgd_interval", "wwpgd_interval", train_updates),
        ("dropout", "dropout", model_updates),
    ]:
        value = getattr(args, arg, None)
        if value is not None:
            target[key] = value
    if model_updates:
        cfg = replace(cfg, model=replace(cfg.model, **model_updates))
    if train_updates:
        cfg = replace(cfg, train=replace(cfg.train, **train_updates))
    if not model_updates and not train_updates:
        return _resolve_config_path(args)
    out = args.results_root / "cli_overrides_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(asdict(cfg)))
    return out

def _seeds(s: str | None) -> list[int] | None:
    return None if not s else [int(x) for x in s.split(',') if x]


def main() -> None:
    p=argparse.ArgumentParser(prog="wwgpt", epilog="Supported experiment profiles: reproduction_tiny, reproduction_fineweb, scaling. Primary reproduction uses blend_eta=0.5; strength/blend_eta ablations must be explicit."); sub=p.add_subparsers(dest="cmd", required=True)
    s=sub.add_parser("smoke-test"); s.add_argument("root", type=Path); s.add_argument("--steps", type=int, default=3)
    a=sub.add_parser("analyze-results", help="analyze one isolated profile result root; no composite pooling by default"); a.add_argument("results_root", type=Path); a.add_argument("--profile", choices=["reproduction_tiny", "reproduction_fineweb", "scaling"], help="Profile label for isolation metadata")
    pd=sub.add_parser("prepare-data", help="prepare data for profiles: reproduction_tiny, reproduction_fineweb, scaling"); pd.add_argument("--profile", choices=["reproduction_tiny", "reproduction_fineweb", "scaling"], default="scaling", help="Experiment profile (default: scaling; choices: reproduction_tiny, reproduction_fineweb, scaling)"); pd.add_argument("--level", type=int, required=True); pd.add_argument("--data-root", type=Path, required=True); pd.add_argument("--token-multiplier", type=int, required=True); pd.add_argument("--config", type=Path, default=Path("configs/default.yaml")); pd.add_argument("--docs-file", type=Path, help="newline-delimited local documents for offline data-preparation tests"); pd.add_argument("--dry-run", action="store_true")
    rm=sub.add_parser("run-multiseed", help="run one profile: reproduction_tiny, reproduction_fineweb, or scaling"); rm.add_argument("--profile", choices=["reproduction_tiny", "reproduction_fineweb", "scaling"], default="scaling", help="Experiment profile (default: scaling; choices: reproduction_tiny, reproduction_fineweb, scaling)"); rm.add_argument("--level", type=int, required=True); rm.add_argument("--data-root", type=Path, required=True); rm.add_argument("--results-root", type=Path, required=True); rm.add_argument("--token-multiplier", type=int, required=True); rm.add_argument("--seeds"); rm.add_argument("--device"); rm.add_argument("--precision"); rm.add_argument("--resume", action="store_true"); rm.add_argument("--config", type=Path, default=Path("configs/default.yaml")); rm.add_argument("--ww-interval", type=int); rm.add_argument("--spectral-interval", type=int); rm.add_argument("--eval-interval", type=int); rm.add_argument("--checkpoint-interval", type=int); rm.add_argument("--optimizer", choices=["adamw","muon","stableadamw"], default="adamw"); rm.add_argument("--extensions", default="none,wwpgd"); rm.add_argument("--extension", choices=["none","wwpgd"]); rm.add_argument("--wwpgd-interval", type=int); rm.add_argument("--batch-size", type=int); rm.add_argument("--gradient-accumulation", type=int); rm.add_argument("--weight-decay", type=float); rm.add_argument("--grad-clip", type=float); rm.add_argument("--eval-batches", type=int); rm.add_argument("--dropout", type=float); rm.add_argument("--lr-schedule", choices=["constant","warmup_cosine","warmup_linear"], help="LR schedule; warmup_cosine is the nanoGPT-style default."); rm.add_argument("--warmup-ratio", type=float, help="Derived warmup fraction when --warmup-steps is omitted."); rm.add_argument("--warmup-steps", type=int, help="Explicit linear warmup optimizer steps."); rm.add_argument("--lr-decay-steps", type=int, help="Cosine/linear decay horizon; defaults to the total optimizer-step horizon."); rm.add_argument("--min-lr-ratio", type=float, help="Minimum LR as a ratio of each group peak LR."); rm.add_argument("--layer-lr", choices=["flat","llrd","manual"], help="Layer LR policy: flat is nanoGPT-compatible; llrd and manual are research ablations."); rm.add_argument("--llrd-gamma", type=float); rm.add_argument("--llrd-min-multiplier", type=float); rm.add_argument("--max-train-tokens", type=int); rm.add_argument("--max-steps", type=int); rm.set_defaults(immediate_projection_spectral=False); rm.add_argument("--immediate-projection-spectral", dest="immediate_projection_spectral", action="store_true"); rm.add_argument("--no-immediate-projection-spectral", dest="immediate_projection_spectral", action="store_false"); rm.add_argument("--allow-code-version-mismatch", action="store_true"); rm.add_argument("--dry-run", action="store_true")

    rt=sub.add_parser("run-canonical-trials", help="publication six-arm canonical trials"); rt.add_argument("--profile", choices=["reproduction_tiny", "reproduction_fineweb", "scaling"], default="scaling"); rt.add_argument("--level", type=int, required=True); rt.add_argument("--data-root", type=Path, required=True); rt.add_argument("--results-root", type=Path, required=True); rt.add_argument("--token-multiplier", type=int, required=True); rt.add_argument("--seeds"); rt.add_argument("--device"); rt.add_argument("--precision"); rt.add_argument("--resume", action="store_true"); rt.add_argument("--config", type=Path, default=Path("configs/default.yaml")); rt.add_argument("--ww-interval", type=int); rt.add_argument("--spectral-interval", type=int); rt.add_argument("--eval-interval", type=int); rt.add_argument("--checkpoint-interval", type=int); rt.set_defaults(immediate_projection_spectral=False); rt.add_argument("--immediate-projection-spectral", dest="immediate_projection_spectral", action="store_true"); rt.add_argument("--no-immediate-projection-spectral", dest="immediate_projection_spectral", action="store_false"); rt.add_argument("--allow-code-version-mismatch", action="store_true"); rt.add_argument("--dry-run", action="store_true")
    ss=sub.add_parser("run-strength-scan", help="explicit external blend_eta/strength ablation; not part of primary reproduction") ; ss.add_argument("--level", type=int, required=True); ss.add_argument("--data-root", type=Path, required=True); ss.add_argument("--results-root", type=Path, required=True); ss.add_argument("--token-multiplier", type=int, required=True); ss.add_argument("--seeds", default="1337"); ss.add_argument("--strengths", default="0.1,0.25,0.5,1.0", help="Explicit ablation strengths. The retired 0.02 scan is not part of reproduction and is not a default."); ss.add_argument("--device"); ss.add_argument("--optimizer", choices=["adamw","muon","stableadamw"], default="adamw"); ss.add_argument("--config", type=Path, default=Path("configs/default.yaml")); ss.add_argument("--eval-interval", type=int); ss.add_argument("--spectral-interval", type=int); ss.add_argument("--checkpoint-interval", type=int); ss.set_defaults(immediate_projection_spectral=True); ss.add_argument("--immediate-projection-spectral", dest="immediate_projection_spectral", action="store_true"); ss.add_argument("--no-immediate-projection-spectral", dest="immediate_projection_spectral", action="store_false"); ss.add_argument("--resume", action="store_true"); ss.add_argument("--continue-on-error", action="store_true", default=True); ss.add_argument("--scan-name", default="strength_scan"); ss.add_argument("--instability-loss-threshold", type=float, default=20.0); ss.add_argument("--include-adamw-control", action="store_true", default=True); ss.add_argument("--dry-run", action="store_true")
    ass=sub.add_parser("analyze-strength-scan"); ass.add_argument("--scan-root", type=Path, required=True)
    ic=sub.add_parser("inspect-checkpoint"); ic.add_argument("--checkpoint", type=Path, required=True)
    vr=sub.add_parser("validate-resume"); vr.add_argument("--run-dir", type=Path, required=True)
    dp=sub.add_parser("device-preflight"); dp.add_argument("--device", default="auto"); dp.add_argument("--output", type=Path, default=Path("."))
    ae=sub.add_parser("audit-experiment"); ae.add_argument("--experiment-root", type=Path, required=True)
    ast=sub.add_parser("audit-strength-scan"); ast.add_argument("--scan-root", type=Path, required=True)
    gr=sub.add_parser("generate-reproducibility-report"); gr.add_argument("--experiment-root", type=Path, required=True); gr.add_argument("--strict", action="store_true")
    pl=sub.add_parser("plan-scaling"); pl.add_argument("--params", type=int); pl.add_argument("--level", type=int); pl.add_argument("--token-multiplier", type=int, required=True); pl.add_argument("--available-tokens", type=int, required=True); pl.add_argument("--batch-size", type=int, default=8); pl.add_argument("--block-size", type=int, default=256); pl.add_argument("--grad-accum", type=int, default=1)
    args=p.parse_args()
    if args.cmd=="smoke-test": print(smoke(args.root, args.steps))
    elif args.cmd=="analyze-results": print(analyze_results(args.results_root))
    elif args.cmd=="prepare-data":
        docs = args.docs_file.read_text().splitlines() if args.docs_file else None
        cfg = _resolved_config(args)
        prep_dir = args.data_root / "fineweb_edu" / f"level_{args.level:02d}" / f"multiplier_{args.token_multiplier}"
        _print_resolved_execution(args, arms=["prepare-data"], seeds=cfg.seeds, trials=1, output_dirs=[prep_dir], dry_run=args.dry_run)
        if args.dry_run:
            return
        print(prepare_scientific_data(args.data_root, args.level, args.token_multiplier, _resolve_config_path(args), docs=docs, min_validation_tokens=1 if docs is not None else 100_000).root, flush=True)
        sys.stderr.flush()
        sys.stdout.flush()
        # Some streaming dataset backends can leave non-daemon workers alive after all artifacts
        # have been written. Exit the CLI process deterministically so shell wrappers can finish.
        os._exit(0)
    elif args.cmd=="run-multiseed":
        # CLI overrides are accepted by the schema-v3 interface.
        exts = [args.extension] if args.extension else [x for x in args.extensions.split(",") if x]
        if set(exts) != {"none", "wwpgd"} or args.optimizer != "adamw":
            raise SystemExit("run-multiseed is canonical-only; use run-canonical-trials for six arms or run-strength-scan for ablations")
        args.config = _config_with_run_overrides(args)
        if args.config is not None and Path(args.config).name == "cli_overrides_config.yaml":
            args.profile = None
        seeds = _seeds(args.seeds) or _resolved_config(args).seeds
        out = args.results_root / "experiments" / f"level_{args.level:02d}" / f"multiplier_{args.token_multiplier}"
        _print_resolved_execution(args, arms=["adamw", "adamw_wwpgd", "muon", "muon_wwpgd", "stable_adamw", "stable_adamw_wwpgd"], seeds=seeds, trials=len(seeds), output_dirs=[out], dry_run=args.dry_run)
        if args.dry_run:
            return
        print(run_canonical_trials(args.level,args.data_root,args.results_root,args.token_multiplier,_seeds(args.seeds),args.config,args.device,args.wwpgd_interval or args.ww_interval,args.eval_interval,args.checkpoint_interval,args.spectral_interval,args.precision,args.resume,args.immediate_projection_spectral,args.allow_code_version_mismatch))
    elif args.cmd=="run-canonical-trials":
        seeds = _seeds(args.seeds) or _resolved_config(args).seeds
        out = args.results_root / "experiments" / f"level_{args.level:02d}" / f"multiplier_{args.token_multiplier}"
        _print_resolved_execution(args, arms=["adamw", "adamw_wwpgd", "muon", "muon_wwpgd", "stable_adamw", "stable_adamw_wwpgd"], seeds=seeds, trials=len(seeds), output_dirs=[out], dry_run=args.dry_run)
        if args.dry_run:
            return
        print(run_canonical_trials(args.level,args.data_root,args.results_root,args.token_multiplier,_seeds(args.seeds),_resolve_config_path(args),args.device,args.ww_interval,args.eval_interval,args.checkpoint_interval,args.spectral_interval,args.precision,args.resume,args.immediate_projection_spectral,args.allow_code_version_mismatch))
    elif args.cmd=="run-strength-scan":
        seeds = _seeds(args.seeds) or _resolved_config(args).seeds
        strengths = parse_strengths(args.strengths)
        arms = (["adamw_control"] if args.include_adamw_control else []) + [f"wwpgd_strength_{x:g}" for x in strengths]
        out = args.results_root / args.scan_name / f"level_{args.level:02d}" / f"multiplier_{args.token_multiplier}"
        _print_resolved_execution(args, arms=arms, seeds=seeds, trials=len(seeds), output_dirs=[out], dry_run=args.dry_run)
        if args.dry_run:
            return
        print(run_strength_scan(args.level,args.data_root,args.results_root,args.token_multiplier,_seeds(args.seeds),args.strengths,_resolve_config_path(args),args.device,args.eval_interval,args.spectral_interval,args.checkpoint_interval,args.immediate_projection_spectral,args.resume,args.continue_on_error,args.scan_name,args.instability_loss_threshold,args.include_adamw_control,args.optimizer))
    elif args.cmd=="analyze-strength-scan": print(analyze_strength_scan_cmd(args.scan_root))
    elif args.cmd=="inspect-checkpoint":
        import json; print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2, sort_keys=True, default=str))
    elif args.cmd=="validate-resume":
        import json; res=validate_resume(args.run_dir); print(json.dumps(res, indent=2, sort_keys=True, default=str)); raise SystemExit(0 if res.get("compatible") else 1)
    elif args.cmd=="device-preflight":
        import json; print(json.dumps(run_device_preflight(args.output, args.device), indent=2, sort_keys=True, default=str))
    elif args.cmd=="audit-experiment": print(audit_experiment(args.experiment_root))
    elif args.cmd=="audit-strength-scan": print(audit_strength_scan(args.scan_root))
    elif args.cmd=="generate-reproducibility-report": print(write_reproducibility_report(args.experiment_root))
    elif args.cmd=="plan-scaling":
        params=args.params
        if params is None:
            from wwgpt.config import load_config
            from wwgpt.model import GPT
            if args.level is None: raise SystemExit("--params or --level is required")
            params=GPT(load_config(None,args.level).model).parameter_report().trainable_parameters
        print(plan_budget(params,args.token_multiplier,args.batch_size,args.block_size,args.grad_accum,args.available_tokens))

if __name__ == "__main__": main()
