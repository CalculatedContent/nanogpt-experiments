from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wwgpt.utils import sha256_bytes, write_json


@dataclass(frozen=True)
class TokenData:
    train: list[int]
    val: list[int]
    vocab_size: int
    corpus_hash: str


def split_for_doc(text: str, val_fraction: float = 0.1) -> str:
    h = int(sha256_bytes(" ".join(text.split()).encode()), 16)
    return "val" if (h % 10_000) < int(val_fraction * 10_000) else "train"


def encode_chars(texts: list[str]) -> tuple[list[int], dict[str, int]]:
    chars = sorted(set("".join(texts)))
    vocab = {ch: i for i, ch in enumerate(chars)}
    return [vocab[ch] for text in texts for ch in text], vocab


def prepare_local_text(data_root: Path, texts: list[str], min_train_tokens: int = 1) -> TokenData:
    prep = data_root / "prepared_local_text"
    prep.mkdir(parents=True, exist_ok=True)
    train_docs = [t for t in texts if split_for_doc(t) == "train"] or texts[:-1]
    val_docs = [t for t in texts if split_for_doc(t) == "val"] or texts[-1:]
    train_tokens, vocab = encode_chars(train_docs)
    val_tokens, _ = encode_chars(val_docs + train_docs)
    val_tokens = [vocab.get(ch, 0) for text in val_docs for ch in text]
    if len(train_tokens) < min_train_tokens:
        raise ValueError(f"insufficient unique training tokens: {len(train_tokens)} < {min_train_tokens}")
    np.save(prep / f"train_{sha256_bytes(bytes(train_tokens)).hex()[:0] if False else 'tokens'}.npy", np.array(train_tokens, dtype=np.int64))
    np.save(prep / "val_tokens.npy", np.array(val_tokens, dtype=np.int64))
    corpus_hash = sha256_bytes("\n".join(texts).encode())
    manifest = {"dataset": "local_text", "corpus_hash": corpus_hash, "train_tokens": len(train_tokens), "val_tokens": len(val_tokens), "valid_for_science": False}
    tok = {"tokenizer": "char-smoke", "vocab_size": len(vocab), "hash": sha256_bytes(json.dumps(vocab, sort_keys=True).encode()), "special_tokens": {}}
    for name, obj in [("data_manifest.json", manifest), ("tokenizer_manifest.json", tok)]:
        p = prep / name
        if not p.exists():
            write_json(p, obj)
    return TokenData(train_tokens, val_tokens, len(vocab), corpus_hash)


class NonRepeatingTokenReader:
    def __init__(self, tokens: list[int], block_size: int):
        self.tokens = tokens
        self.block_size = block_size
        self.pos = 0

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        need = batch_size * self.block_size + 1
        if self.pos + need > len(self.tokens):
            raise ValueError("token stream exhausted; refusing to wrap or repeat")
        chunk = self.tokens[self.pos:self.pos + need]
        self.pos += batch_size * self.block_size
        x = np.array(chunk[:-1], dtype=np.int64).reshape(batch_size, self.block_size)
        y = np.array(chunk[1:], dtype=np.int64).reshape(batch_size, self.block_size)
        return x, y
