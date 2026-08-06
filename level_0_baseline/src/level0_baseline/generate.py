from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .model import GPT, GPTConfig
from .runtime import device_auto, seed_all


def generate_from_checkpoint(
    checkpoint: str | Path,
    *,
    prompt: str,
    num_samples: int = 3,
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_k: int | None = 50,
    seed: int = 9000,
    device: str = "auto",
) -> list[str]:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "Text generation requires tiktoken; run scripts/setup_mac.sh"
        ) from exc

    resolved_device = device_auto() if device == "auto" else torch.device(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_cfg = GPTConfig(**payload["config"]["model"])
    model = GPT(model_cfg).to(resolved_device)
    model.load_state_dict(payload["model"])
    model.eval()
    encoder = tiktoken.get_encoding("gpt2")
    prompt_tokens = encoder.encode(prompt, allowed_special=set())
    if not prompt_tokens:
        prompt_tokens = [encoder.eot_token]
    context = torch.tensor(prompt_tokens, dtype=torch.long, device=resolved_device)[
        None, :
    ]

    outputs: list[str] = []
    for sample_index in range(int(num_samples)):
        sample_seed = int(seed) + sample_index
        seed_all(sample_seed)
        generator = (
            torch.Generator(device="cpu").manual_seed(sample_seed)
            if resolved_device.type == "cpu"
            else None
        )
        generated = model.generate(
            context.clone(),
            int(max_new_tokens),
            temperature=float(temperature),
            top_k=None if top_k is None else int(top_k),
            generator=generator,
        )
        outputs.append(encoder.decode(generated[0].tolist()))
    return outputs


def write_samples(
    run_dir: str | Path,
    samples: list[str],
    *,
    prompt: str,
    checkpoint: str | Path,
    settings: dict[str, Any],
) -> Path:
    run_dir = Path(run_dir)
    output = run_dir / "generated_samples.json"
    payload = {
        "checkpoint": str(Path(checkpoint)),
        "prompt": prompt,
        "settings": settings,
        "samples": samples,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = run_dir / "generated_samples.md"
    sections = [
        "# Final-checkpoint samples",
        "",
        f"Prompt: `{prompt}`",
        "",
    ]
    for index, sample in enumerate(samples, start=1):
        sections.extend([f"## Sample {index}", "", sample, ""])
    markdown.write_text("\n".join(sections), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="The future of artificial intelligence")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    samples = generate_from_checkpoint(
        args.checkpoint,
        prompt=args.prompt,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        device=args.device,
    )
    for index, sample in enumerate(samples, start=1):
        print(f"\n--- sample {index} ---\n{sample}")
    if args.output_dir:
        write_samples(
            args.output_dir,
            samples,
            prompt=args.prompt,
            checkpoint=args.checkpoint,
            settings={
                "num_samples": args.num_samples,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "seed": args.seed,
                "device": args.device,
            },
        )


if __name__ == "__main__":
    main()
