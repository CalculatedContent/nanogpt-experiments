from __future__ import annotations

import argparse
from pathlib import Path

from wwgpt.analysis import analyze_results
from wwgpt.config import DEFAULT_SEEDS
from wwgpt.data import prepare_local_text
from wwgpt.scaling import plan_budget
from wwgpt.train import smoke


def main() -> None:
    p=argparse.ArgumentParser(prog="wwgpt")
    sub=p.add_subparsers(dest="cmd", required=True)
    s=sub.add_parser("smoke-test"); s.add_argument("root", type=Path); s.add_argument("--steps", type=int, default=3)
    a=sub.add_parser("analyze-results"); a.add_argument("results_root", type=Path)
    pd=sub.add_parser("prepare-data"); pd.add_argument("--max-level", type=int, required=True); pd.add_argument("--data-root", type=Path, required=True); pd.add_argument("--dataset", default="fineweb_edu"); pd.add_argument("--token-multiplier", type=int, required=True); pd.add_argument("--local-text", type=Path)
    sub.add_parser("run-multiseed").add_argument("results_root", type=Path)
    pl=sub.add_parser("plan-scaling"); pl.add_argument("--params", type=int, required=True); pl.add_argument("--token-multiplier", type=int, required=True); pl.add_argument("--available-tokens", type=int, required=True); pl.add_argument("--batch-size", type=int, default=8); pl.add_argument("--block-size", type=int, default=256); pl.add_argument("--grad-accum", type=int, default=1)
    args=p.parse_args()
    if args.cmd=="smoke-test": print(smoke(args.root, args.steps))
    elif args.cmd=="analyze-results": print(analyze_results(args.results_root))
    elif args.cmd=="prepare-data":
        texts=["local smoke data only "+str(i) for i in range(1000)] if args.local_text is None else args.local_text.read_text().splitlines()
        d=prepare_local_text(args.data_root, texts, 1); print(args.data_root/"prepared_local_text"); print(d.corpus_hash)
    elif args.cmd=="run-multiseed":
        root=smoke(args.results_root, 2, DEFAULT_SEEDS[:1]); print(root)
    elif args.cmd=="plan-scaling": print(plan_budget(args.params,args.token_multiplier,args.batch_size,args.block_size,args.grad_accum,args.available_tokens))

if __name__ == "__main__": main()
