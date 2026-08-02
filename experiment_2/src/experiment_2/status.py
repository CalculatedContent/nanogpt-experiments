from __future__ import annotations

import argparse
import json
from pathlib import Path


def completed(path: Path) -> bool:
    try:
        return path.is_file() and json.loads(path.read_text()).get("completed") is True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--seeds", required=True)
    args = parser.parse_args()

    root = Path(args.pair_root)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    baseline = [
        seed
        for seed in seeds
        if completed(root / f"baseline/results/adamw_seed_{seed}/run_complete.json")
    ]
    adaptive = [
        seed
        for seed in seeds
        if completed(
            root
            / f"adaptive/results/adamw_adaptive_wwpgd_seed_{seed}/run_complete.json"
        )
    ]
    print(f"Pair root: {root}")
    print(f"Baseline complete ({len(baseline)}/{len(seeds)}): {baseline}")
    print(f"Adaptive complete ({len(adaptive)}/{len(seeds)}): {adaptive}")


if __name__ == "__main__":
    main()
