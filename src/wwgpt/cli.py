from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from wwgpt.analysis import analyze_results
from wwgpt.config import DEFAULT_SEEDS
from wwgpt.data import prepare_scientific_data
from wwgpt.scaling import plan_budget
from wwgpt.train import run_multiseed_scientific, smoke
from wwgpt.strength_scan import run_strength_scan, parse_strengths
from wwgpt.strength_scan_analysis import analyze_strength_scan as analyze_strength_scan_cmd


def _seeds(s: str | None) -> list[int] | None:
    return None if not s else [int(x) for x in s.split(',') if x]


def main() -> None:
    p=argparse.ArgumentParser(prog="wwgpt"); sub=p.add_subparsers(dest="cmd", required=True)
    s=sub.add_parser("smoke-test"); s.add_argument("root", type=Path); s.add_argument("--steps", type=int, default=3)
    a=sub.add_parser("analyze-results"); a.add_argument("results_root", type=Path)
    pd=sub.add_parser("prepare-data"); pd.add_argument("--level", type=int, required=True); pd.add_argument("--data-root", type=Path, required=True); pd.add_argument("--token-multiplier", type=int, required=True); pd.add_argument("--config", type=Path); pd.add_argument("--max-level", type=int); pd.add_argument("--dataset", default="fineweb_edu")
    rm=sub.add_parser("run-multiseed"); rm.add_argument("--level", type=int, required=True); rm.add_argument("--data-root", type=Path, required=True); rm.add_argument("--results-root", type=Path, required=True); rm.add_argument("--token-multiplier", type=int, required=True); rm.add_argument("--seeds"); rm.add_argument("--device"); rm.add_argument("--precision"); rm.add_argument("--resume", action="store_true"); rm.add_argument("--config", type=Path); rm.add_argument("--ww-interval", type=int); rm.add_argument("--spectral-interval", type=int); rm.add_argument("--eval-interval", type=int); rm.add_argument("--checkpoint-interval", type=int)
    ss=sub.add_parser("run-strength-scan"); ss.add_argument("--level", type=int, required=True); ss.add_argument("--data-root", type=Path, required=True); ss.add_argument("--results-root", type=Path, required=True); ss.add_argument("--token-multiplier", type=int, required=True); ss.add_argument("--seeds", default="1337"); ss.add_argument("--strengths", default="0.02,0.1,0.25,0.5,1.0"); ss.add_argument("--device"); ss.add_argument("--config", type=Path); ss.add_argument("--eval-interval", type=int); ss.add_argument("--spectral-interval", type=int); ss.add_argument("--checkpoint-interval", type=int); ss.add_argument("--immediate-projection-spectral", action="store_true", default=True); ss.add_argument("--resume", action="store_true"); ss.add_argument("--continue-on-error", action="store_true", default=True); ss.add_argument("--scan-name", default="strength_scan"); ss.add_argument("--instability-loss-threshold", type=float, default=20.0); ss.add_argument("--include-adamw-control", action="store_true", default=True)
    ass=sub.add_parser("analyze-strength-scan"); ass.add_argument("--scan-root", type=Path, required=True)
    pl=sub.add_parser("plan-scaling"); pl.add_argument("--params", type=int, required=True); pl.add_argument("--token-multiplier", type=int, required=True); pl.add_argument("--available-tokens", type=int, required=True); pl.add_argument("--batch-size", type=int, default=8); pl.add_argument("--block-size", type=int, default=256); pl.add_argument("--grad-accum", type=int, default=1)
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
    elif args.cmd=="run-multiseed": print(run_multiseed_scientific(args.level,args.data_root,args.results_root,args.token_multiplier,_seeds(args.seeds),args.config,args.device,args.ww_interval,args.eval_interval,args.checkpoint_interval,args.spectral_interval,args.precision,args.resume))
    elif args.cmd=="run-strength-scan": print(run_strength_scan(args.level,args.data_root,args.results_root,args.token_multiplier,_seeds(args.seeds),args.strengths,args.config,args.device,args.eval_interval,args.spectral_interval,args.checkpoint_interval,args.immediate_projection_spectral,args.resume,args.continue_on_error,args.scan_name,args.instability_loss_threshold,args.include_adamw_control))
    elif args.cmd=="analyze-strength-scan": print(analyze_strength_scan_cmd(args.scan_root))
    elif args.cmd=="plan-scaling": print(plan_budget(args.params,args.token_multiplier,args.batch_size,args.block_size,args.grad_accum,args.available_tokens))

if __name__ == "__main__": main()
