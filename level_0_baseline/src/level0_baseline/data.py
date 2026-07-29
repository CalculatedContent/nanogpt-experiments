from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .config import roots

def encode(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8)

def write_splits(texts, out: Path, train_bytes: int, val_bytes: int, test_bytes: int):
    out.mkdir(parents=True, exist_ok=True)
    need = train_bytes + val_bytes + test_bytes
    chunks, total = [], 0
    for text in texts:
        x = encode(text + "\n")
        chunks.append(x)
        total += len(x)
        if total >= need:
            break
    if total < need:
        raise RuntimeError(f"corpus supplied {total:,} bytes; need {need:,}")
    all_tokens = np.concatenate(chunks)[:need]
    boundaries = {"train": (0, train_bytes), "val": (train_bytes, train_bytes + val_bytes), "test": (train_bytes + val_bytes, need)}
    for name, (a, b) in boundaries.items():
        all_tokens[a:b].tofile(out / f"{name}.bin")
    (out / "meta.json").write_text(json.dumps({"tokenizer": "utf8-byte", "vocab_size": 256, "sizes": {k: b-a for k, (a, b) in boundaries.items()}}, indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fineweb-edu")
    p.add_argument("--output-dir")
    p.add_argument("--train-bytes", type=int, default=50_000_000)
    p.add_argument("--val-bytes", type=int, default=2_000_000)
    p.add_argument("--test-bytes", type=int, default=2_000_000)
    p.add_argument("--local-text")
    a = p.parse_args()
    out = Path(a.output_dir) if a.output_dir else roots()["data"]
    if a.local_text:
        text = Path(a.local_text).read_text(encoding="utf-8")
        def repeat():
            while True:
                yield text
        texts = repeat()
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise SystemExit("Install data support: pip install -e '.[data]'") from e
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        texts = (row["text"] for row in ds)
    write_splits(texts, out, a.train_bytes, a.val_bytes, a.test_bytes)
    print(out)

if __name__ == "__main__":
    main()
