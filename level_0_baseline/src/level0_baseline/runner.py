from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from .analysis import run_directory, run_is_complete, run_status_table
from .config import SUPPORTED_OPTIMIZERS, canonical_seeds, load_config, roots
from .generate import generate_from_checkpoint, write_samples


def validate_data_root(data_root: str | Path) -> None:
    data_root = Path(data_root)
    required = [data_root / "meta.json"] + [
        data_root / f"{split}.bin" for split in ("train", "val", "test")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "prepared Level Zero data are incomplete; missing: " + ", ".join(missing)
        )
    metadata = json.loads((data_root / "meta.json").read_text(encoding="utf-8"))
    if metadata.get("tokenizer") != "gpt2":
        raise RuntimeError("Level Zero runner requires GPT-2 BPE data")


def run_suite(
    *,
    config_path: str | Path,
    data_root: str | Path,
    results_root: str | Path,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] | None = None,
    device: str = "auto",
    resume: bool = True,
    overwrite: bool = False,
    generate: bool = False,
    dry_run: bool = False,
):
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    cfg = load_config(config_path)
    selected_seeds = tuple(int(seed) for seed in (seeds or canonical_seeds(cfg)))
    selected_optimizers = tuple(str(name) for name in optimizers)
    unsupported = set(selected_optimizers).difference(SUPPORTED_OPTIMIZERS)
    if unsupported:
        raise ValueError(f"unsupported optimizers: {sorted(unsupported)}")
    validate_data_root(data_root)
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")

    for optimizer in selected_optimizers:
        for seed in selected_seeds:
            run_dir = run_directory(results_root, optimizer, seed)
            if run_is_complete(results_root, optimizer, seed) and not overwrite:
                print(
                    f"[level0-runner] skip completed optimizer={optimizer} seed={seed}",
                    flush=True,
                )
                continue
            command = [
                sys.executable,
                "-m",
                "level0_baseline.train",
                "--config",
                str(Path(config_path).resolve()),
                "--data-root",
                str(Path(data_root).resolve()),
                "--results-root",
                str(results_root.resolve()),
                "--optimizer",
                optimizer,
                "--seed",
                str(seed),
                "--device",
                device,
            ]
            if overwrite:
                command.append("--overwrite")
            elif resume and run_dir.exists():
                command.append("--resume")
            print("[level0-runner] " + " ".join(command), flush=True)
            if not dry_run:
                subprocess.run(command, check=True, env=environment)
                if not run_is_complete(results_root, optimizer, seed):
                    raise RuntimeError(
                        f"training exited without completion marker: {run_dir}"
                    )

            if generate and not dry_run:
                sampling = cfg["sampling"]
                checkpoint = run_dir / "checkpoint_final.pt"
                samples = generate_from_checkpoint(
                    checkpoint,
                    prompt=str(sampling["prompt"]),
                    num_samples=int(sampling["num_samples"]),
                    max_new_tokens=int(sampling["max_new_tokens"]),
                    temperature=float(sampling["temperature"]),
                    top_k=int(sampling["top_k"]),
                    seed=int(sampling["seed_offset"]) + int(seed),
                    device=device,
                )
                write_samples(
                    run_dir,
                    samples,
                    prompt=str(sampling["prompt"]),
                    checkpoint=checkpoint,
                    settings={
                        "num_samples": int(sampling["num_samples"]),
                        "max_new_tokens": int(sampling["max_new_tokens"]),
                        "temperature": float(sampling["temperature"]),
                        "top_k": int(sampling["top_k"]),
                        "seed": int(sampling["seed_offset"]) + int(seed),
                        "device": device,
                    },
                )

    return run_status_table(
        results_root, optimizers=selected_optimizers, seeds=selected_seeds
    )


def main() -> None:
    defaults = roots()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/level0.yaml")
    parser.add_argument("--data-root", default=str(defaults["data"]))
    parser.add_argument("--results-root", default=str(defaults["results"]))
    parser.add_argument(
        "--optimizers",
        default=",".join(SUPPORTED_OPTIMIZERS),
        help="comma-separated optimizer names",
    )
    parser.add_argument("--seeds", help="comma-separated seeds; default comes from config")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    optimizers = tuple(item.strip() for item in args.optimizers.split(",") if item.strip())
    seeds = (
        tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
        if args.seeds
        else None
    )
    status = run_suite(
        config_path=args.config,
        data_root=args.data_root,
        results_root=args.results_root,
        optimizers=optimizers,
        seeds=seeds,
        device=args.device,
        resume=not args.no_resume,
        overwrite=args.overwrite,
        generate=args.generate,
        dry_run=args.dry_run,
    )
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
