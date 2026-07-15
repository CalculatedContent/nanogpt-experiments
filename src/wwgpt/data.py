from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from wwgpt.config import ExperimentConfig, load_config
from wwgpt.model import GPT
from wwgpt.scaling import plan_budget
from wwgpt.utils import sha256_bytes, unique_dir, write_json


@dataclass(frozen=True)
class TokenData:
    train: list[int]
    val: list[int]
    vocab_size: int
    corpus_hash: str
    root: Path | None = None
    data_manifest: dict[str, object] | None = None
    tokenizer_manifest: dict[str, object] | None = None


def split_for_doc(text: str, val_fraction: float = 0.1) -> str:
    h = int(sha256_bytes(" ".join(text.split()).encode()), 16)
    return "val" if (h % 10_000) < int(val_fraction * 10_000) else "train"


def encode_chars(texts: list[str]) -> tuple[list[int], dict[str, int]]:
    chars = sorted(set("".join(texts)))
    vocab = {ch: i for i, ch in enumerate(chars)}
    return [vocab[ch] for text in texts for ch in text], vocab


def prepare_local_text(data_root: Path, texts: list[str], min_train_tokens: int = 1) -> TokenData:
    prep = data_root / "prepared_local_text"
    if prep.exists():
        prep = unique_dir(data_root, "prepared_local_text")
    else:
        prep.mkdir(parents=True, exist_ok=False)
    train_docs = [t for t in texts if split_for_doc(t) == "train"] or texts[:-1]
    val_docs = [t for t in texts if split_for_doc(t) == "val"] or texts[-1:]
    train_tokens, vocab = encode_chars(train_docs)
    if len(train_tokens) < min_train_tokens:
        raise ValueError(f"insufficient unique training tokens: {len(train_tokens)} < {min_train_tokens}")
    val_tokens = [vocab.get(ch, 0) for text in val_docs for ch in text]
    np.save(prep / "train_tokens.npy", np.array(train_tokens, dtype=np.int64))
    np.save(prep / "val_tokens.npy", np.array(val_tokens, dtype=np.int64))
    corpus_hash = sha256_bytes("\n".join(texts).encode())
    manifest = {"dataset": "local_text", "dataset_name": "local_text", "corpus_hash": corpus_hash, "train_tokens": len(train_tokens), "val_tokens": len(val_tokens), "smoke_test": True, "valid_for_science": False, "repeated_stream": False}
    tok = {"tokenizer": "char-smoke", "tokenizer_type": "char-smoke", "vocab_size": len(vocab), "hash": sha256_bytes(json.dumps(vocab, sort_keys=True).encode()), "special_tokens": {}}
    write_json(prep / "data_manifest.json", manifest); write_json(prep / "tokenizer_manifest.json", tok)
    return TokenData(train_tokens, val_tokens, len(vocab), corpus_hash, prep, manifest, tok)


def _iter_fineweb(cfg: ExperimentConfig) -> Iterable[str]:
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train", revision=cfg.dataset_revision, streaming=True)
    for row in ds:
        text = " ".join(str(row.get("text", "")).split())
        if text:
            yield text


def _train_bpe(train_docs: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="<unk>")); tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"])
    tok.train_from_iterator(train_docs, trainer=trainer)
    return tok


def _hash_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def prepare_scientific_data(data_root: Path, level: int, token_multiplier: int, config_path: Path | None = None, docs: Iterable[str] | None = None, min_validation_tokens: int = 1024) -> TokenData:
    cfg = load_config(config_path, level)
    model = GPT(cfg.model)
    report = model.parameter_report()
    param_count = getattr(report, f"{cfg.parameter_count_convention}_parameters", report.total_parameters)
    tokens_per_step = cfg.train.batch_size * cfg.model.block_size * cfg.train.gradient_accumulation
    requested = param_count * token_multiplier
    budget = plan_budget(param_count, token_multiplier, cfg.train.batch_size, cfg.model.block_size, cfg.train.gradient_accumulation, 10**18)
    realized = budget.realized_tokens
    prep = unique_dir(data_root / "fineweb_edu" / f"level_{level:02d}" / f"multiplier_{token_multiplier}", "prepared")
    train_docs: list[str] = []; val_docs: list[str] = []
    corpus = []
    source = docs if docs is not None else _iter_fineweb(cfg)
    tok: Tokenizer | None = None; train_tokens: list[int] = []; val_tokens: list[int] = []
    for text in source:
        norm = " ".join(text.split())
        corpus.append(norm)
        if split_for_doc(norm) == "val": val_docs.append(norm)
        else: train_docs.append(norm)
        if tok is None and len(train_docs) >= 128:
            tok = _train_bpe(train_docs, cfg.model.vocab_size)
        if tok is not None and len(train_tokens) < realized:
            # Tokenize each whole document at most once; never wrap.
            if split_for_doc(norm) == "train": train_tokens.extend(tok.encode(norm).ids)
            elif len(val_tokens) < min_validation_tokens: val_tokens.extend(tok.encode(norm).ids)
        if len(train_tokens) >= realized and len(val_tokens) >= min_validation_tokens:
            break
    if tok is None:
        if not train_docs:
            raise ValueError("insufficient unique training documents to train BPE tokenizer")
        tok = _train_bpe(train_docs, cfg.model.vocab_size)
        train_tokens = [i for d in train_docs for i in tok.encode(d).ids]
        val_tokens = [i for d in val_docs for i in tok.encode(d).ids]
    needed_train = realized + 1
    if not val_tokens and val_docs:
        val_tokens = [i for d in val_docs for i in tok.encode(d).ids]
    if len(train_tokens) < needed_train:
        raise ValueError(f"insufficient unique training tokens: {len(train_tokens)} < {needed_train}; refusing to wrap or repeat")
    if not val_tokens:
        raise ValueError("insufficient validation tokens")
    train_tokens = train_tokens[:needed_train]
    np.save(prep / "train_tokens.npy", np.array(train_tokens, dtype=np.int64)); np.save(prep / "val_tokens.npy", np.array(val_tokens, dtype=np.int64))
    tok_path = prep / "tokenizer.json"; tok.save(str(tok_path)); tokenizer_hash = _hash_file(tok_path)
    corpus_hash = sha256_bytes("\n".join(corpus).encode())
    data_manifest = {"dataset_name": cfg.dataset_name, "dataset_config": cfg.dataset_config, "dataset_revision": cfg.dataset_revision, "split": "train", "train_document_count": len(train_docs), "validation_document_count": len(val_docs), "unique_train_tokens": len(train_tokens), "validation_tokens": len(val_tokens), "requested_tokens": requested, "realized_tokens": realized, "tokens_per_optimizer_step": tokens_per_step, "optimizer_steps": realized // tokens_per_step, "tokenizer_hash": tokenizer_hash, "corpus_hash": corpus_hash, "valid_for_science": True, "repeated_stream": False, "smoke_test": False, "parameter_report": model.report_dict(), "parameter_count_convention": cfg.parameter_count_convention}
    tokenizer_manifest = {"tokenizer_type": "BPE", "vocabulary_size": cfg.model.vocab_size, "vocab_size": cfg.model.vocab_size, "tokenizer_hash": tokenizer_hash, "special_token_ids": {s: tok.token_to_id(s) for s in ["<unk>", "<bos>", "<eos>", "<pad>"]}, "training_document_partition": "sha256-normalized-content", "dataset_revision": cfg.dataset_revision}
    write_json(prep / "data_manifest.json", data_manifest); write_json(prep / "tokenizer_manifest.json", tokenizer_manifest)
    return TokenData(train_tokens, val_tokens, cfg.model.vocab_size, corpus_hash, prep, data_manifest, tokenizer_manifest)


def load_prepared_scientific_data(data_root: Path, level: int, token_multiplier: int) -> TokenData:
    roots = sorted((data_root / "fineweb_edu" / f"level_{level:02d}" / f"multiplier_{token_multiplier}").glob("prepared_*"))
    for prep in reversed(roots):
        dm = json.loads((prep / "data_manifest.json").read_text()); tm = json.loads((prep / "tokenizer_manifest.json").read_text())
        if dm.get("valid_for_science") is True and dm.get("smoke_test") is False and tm.get("tokenizer_type") == "BPE":
            return TokenData(np.load(prep / "train_tokens.npy").astype(int).tolist(), np.load(prep / "val_tokens.npy").astype(int).tolist(), int(tm.get("vocabulary_size", tm.get("vocab_size", 8192))), str(dm["corpus_hash"]), prep, dm, tm)
    raise FileNotFoundError("no compatible scientific prepared data found; run scripts/download_data.sh first")


class NonRepeatingTokenReader:
    def __init__(self, tokens: list[int], block_size: int):
        self.tokens = tokens; self.block_size = block_size; self.pos = 0
    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        need = batch_size * self.block_size + 1
        if self.pos + need > len(self.tokens):
            raise ValueError("token stream exhausted; refusing to wrap or repeat")
        chunk = self.tokens[self.pos:self.pos + need]; self.pos += batch_size * self.block_size
        return np.array(chunk[:-1], dtype=np.int64).reshape(batch_size, self.block_size), np.array(chunk[1:], dtype=np.int64).reshape(batch_size, self.block_size)
