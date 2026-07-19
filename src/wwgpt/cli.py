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
from wwgpt.scaling import plan_budget
from wwgpt.train import run_multiseed_scientific, run_canonical_trials, smoke
from wwgpt.strength_scan import run_strength_scan, parse_strengths
from wwgpt.strength_scan_analysis import analyze_strength_scan as analyze_strength_scan_cmd
from wwgpt.device import device_summary, save_device_manifest, run_device_preflight
from wwgpt.integrity import audit_experiment, audit_strength_scan
from wwgpt.checkpointing import inspect_checkpoint, validate_resume
from wwgpt.reproducibility import write_reproducibility_report



def _config_with_run_overrides(args):
    from wwgpt.config import load_config
    cfg = load_config(args.config, args.level)
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
        return args.config
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
    pd=sub.add_parser("prepare-data", help="prepare data for profiles: reproduction_tiny, reproduction_fineweb, scaling"); pd.add_argument("--profile", choices=["reproduction_tiny", "reproduction_fineweb", "scaling"], default="scaling", help="Experiment profile (default: scaling; choices: reproduction_tiny, reproduction_fineweb, scaling)"); pd.add_argument("--level", type=int, required=True); pd.add_argument("--data-root", type=Path, required=True); pd.add_argument("--token-multiplier", type=int, required=True); pd.add_argument("--config", type=Path); pd.add_argument("--max-level", type=int); pd.add_argument("--dataset", default="fineweb_edu")
    rm=sub.add_parser("run-multiseed", help="run one profile: reproduction_tiny, reproduction_fineweb, or scaling"); rm.add_argument("--profile", choices=["reproduction_tiny", "reproduction_fineweb", "scaling"], default="scaling", help="Experiment profile (default: scaling; choices: reproduction_tiny, reproduction_fineweb, scaling)"); rm.add_argument("--level", type=int, required=True); rm.add_argument("--data-root", type=Path, required=True); rm.add_argument("--results-root", type=Path, required=True); rm.add_argument("--token-multiplier", type=int, required=True); rm.add_argument("--seeds"); rm.add_argument("--device"); rm.add_argument("--precision"); rm.add_argument("--resume", action="store_true"); rm.add_argument("--config", type=Path); rm.add_argument("--ww-interval", type=int); rm.add_argument("--spectral-interval", type=int); rm.add_argument("--eval-interval", type=int); rm.add_argument("--checkpoint-interval", type=int); rm.add_argument("--optimizer", choices=["adamw","muon","stableadamw"], default="adamw"); rm.add_argument("--extensions", default="none,wwpgd"); rm.add_argument("--extension", choices=["none","wwpgd"]); rm.add_argument("--wwpgd-interval", type=int); rm.add_argument("--batch-size", type=int); rm.add_argument("--gradient-accumulation", type=int); rm.add_argument("--weight-decay", type=float); rm.add_argument("--grad-clip", type=float); rm.add_argument("--eval-batches", type=int); rm.add_argument("--dropout", type=float); rm.add_argument("--lr-schedule", choices=["constant","warmup_cosine","warmup_linear"], help="LR schedule; warmup_cosine is the nanoGPT-style default."); rm.add_argument("--warmup-ratio", type=float, help="Derived warmup fraction when --warmup-steps is omitted."); rm.add_argument("--warmup-steps", type=int, help="Explicit linear warmup optimizer steps."); rm.add_argument("--lr-decay-steps", type=int, help="Cosine/linear decay horizon; defaults to the total optimizer-step horizon."); rm.add_argument("--min-lr-ratio", type=float, help="Minimum LR as a ratio of each group peak LR."); rm.add_argument("--layer-lr", choices=["flat","llrd","manual"], help="Layer LR policy: flat is nanoGPT-compatible; llrd and manual are research ablations."); rm.add_argument("--llrd-gamma", type=float); rm.add_argument("--llrd-min-multiplier", type=float); rm.add_argument("--max-train-tokens", type=int); rm.add_argument("--max-steps", type=int); rm.set_defaults(immediate_projection_spectral=False); rm.add_argument("--immediate-projection-spectral", dest="immediate_projection_spectral", action="store_true"); rm.add_argument("--no-immediate-projection-spectral", dest="immediate_projection_spectral", action="store_false"); rm.add_argument("--allow-code-version-mismatch", action="store_true")

    rt=sub.add_parser("run-canonical-trials", help="publication six-arm canonical trials"); rt.add_argument("--profile", choices=["reproduction_tiny", "reproduction_fineweb", "scaling"], default="scaling"); rt.add_argument("--level", type=int, required=True); rt.add_argument("--data-root", type=Path, required=True); rt.add_argument("--results-root", type=Path, required=True); rt.add_argument("--token-multiplier", type=int, required=True); rt.add_argument("--seeds"); rt.add_argument("--device"); rt.add_argument("--precision"); rt.add_argument("--resume", action="store_true"); rt.add_argument("--config", type=Path); rt.add_argument("--ww-interval", type=int); rt.add_argument("--spectral-interval", type=int); rt.add_argument("--eval-interval", type=int); rt.add_argument("--checkpoint-interval", type=int); rt.set_defaults(immediate_projection_spectral=False); rt.add_argument("--immediate-projection-spectral", dest="immediate_projection_spectral", action="store_true"); rt.add_argument("--no-immediate-projection-spectral", dest="immediate_projection_spectral", action="store_false"); rt.add_argument("--allow-code-version-mismatch", action="store_true")
    ss=sub.add_parser("run-strength-scan", help="explicit external blend_eta/strength ablation; not part of primary reproduction") ; ss.add_argument("--level", type=int, required=True); ss.add_argument("--data-root", type=Path, required=True); ss.add_argument("--results-root", type=Path, required=True); ss.add_argument("--token-multiplier", type=int, required=True); ss.add_argument("--seeds", default="1337"); ss.add_argument("--strengths", default="0.1,0.25,0.5,1.0", help="Explicit ablation strengths. The retired 0.02 scan is not part of reproduction and is not a default."); ss.add_argument("--device"); ss.add_argument("--optimizer", choices=["adamw","muon","stableadamw"], default="adamw"); ss.add_argument("--config", type=Path); ss.add_argument("--eval-interval", type=int); ss.add_argument("--spectral-interval", type=int); ss.add_argument("--checkpoint-interval", type=int); ss.set_defaults(immediate_projection_spectral=True); ss.add_argument("--immediate-projection-spectral", dest="immediate_projection_spectral", action="store_true"); ss.add_argument("--no-immediate-projection-spectral", dest="immediate_projection_spectral", action="store_false"); ss.add_argument("--resume", action="store_true"); ss.add_argument("--continue-on-error", action="store_true", default=True); ss.add_argument("--scan-name", default="strength_scan"); ss.add_argument("--instability-loss-threshold", type=float, default=20.0); ss.add_argument("--include-adamw-control", action="store_true", default=True)
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
        print(prepare_scientific_data(args.data_root, args.level, args.token_multiplier, args.config).root, flush=True)
        sys.stderr.flush()
        sys.stdout.flush()
        # Some streaming dataset backends can leave non-daemon workers alive after all artifacts
        # have been written. Exit the CLI process deterministically so shell wrappers can finish.
        os._exit(0)
    elif args.cmd=="run-multiseed":
        # CLI overrides are accepted by the schema-v3 interface.
        exts = [args.extension] if args.extension else [x for x in args.extensions.split(",") if x]
        print(run_multiseed_scientific(args.level,args.data_root,args.results_root,args.token_multiplier,_seeds(args.seeds),_config_with_run_overrides(args),args.device,args.wwpgd_interval or args.ww_interval,args.eval_interval,args.checkpoint_interval,args.spectral_interval,args.precision,args.resume,args.optimizer,exts,args.immediate_projection_spectral,args.allow_code_version_mismatch))
    elif args.cmd=="run-canonical-trials": print(run_canonical_trials(args.level,args.data_root,args.results_root,args.token_multiplier,_seeds(args.seeds),args.config,args.device,args.ww_interval,args.eval_interval,args.checkpoint_interval,args.spectral_interval,args.precision,args.resume,args.immediate_projection_spectral,args.allow_code_version_mismatch))
    elif args.cmd=="run-strength-scan": print(run_strength_scan(args.level,args.data_root,args.results_root,args.token_multiplier,_seeds(args.seeds),args.strengths,args.config,args.device,args.eval_interval,args.spectral_interval,args.checkpoint_interval,args.immediate_projection_spectral,args.resume,args.continue_on_error,args.scan_name,args.instability_loss_threshold,args.include_adamw_control,args.optimizer))
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
