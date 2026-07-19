from __future__ import annotations

import json
import subprocess
import sys
import time
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


def _preparation_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except Exception:
        return "unknown"


def _source_identifiers(cfg: ExperimentConfig, documents: list[str] | None = None) -> dict[str, object]:
    ids: dict[str, object] = {
        "dataset_name": cfg.dataset_name,
        "dataset_subset": cfg.dataset_subset or cfg.dataset_config,
        "dataset_config": cfg.dataset_config,
        "dataset_split": cfg.dataset_split,
        "dataset_revision": cfg.dataset_revision,
    }
    if documents is not None:
        ids["document_sha256"] = [sha256_bytes(d.encode()) for d in documents]
    return ids


@dataclass(frozen=True)
class TokenData:
    train: list[int]
    val: list[int]
    vocab_size: int
    corpus_hash: str
    root: Path | None = None
    data_manifest: dict[str, object] | None = None
    tokenizer_manifest: dict[str, object] | None = None


TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FINEWEB_GPT2_EOT = 50256


def _first_90_final_10(tokens: list[int]) -> tuple[list[int], list[int]]:
    split = int(len(tokens) * 0.9)
    return tokens[:split], tokens[split:]


def prepare_tiny_shakespeare_char_reproduction(data_root: Path, text: str | None = None) -> TokenData:
    if text is None:
        import urllib.request
        text = urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=30).read().decode("utf-8")
    prep = unique_dir(data_root / "tiny_shakespeare_char_reproduction", "prepared")
    all_tokens, vocab = encode_chars([text])
    train_tokens, val_tokens = _first_90_final_10(all_tokens)
    np.save(prep / "train_tokens.npy", np.array(train_tokens, dtype=np.int64))
    np.save(prep / "val_tokens.npy", np.array(val_tokens, dtype=np.int64))
    corpus_hash = sha256_bytes(text.encode())
    vocab_hash = sha256_bytes(json.dumps(vocab, sort_keys=True).encode())
    data_manifest = {"scientific_schema_version": 3, "data_mode": "tiny_shakespeare_char_reproduction", "dataset_name": "tiny_shakespeare", "source_text": "original Tiny Shakespeare text", "split_method": "first_90_percent_train_final_10_percent_validation", "train_tokens": len(train_tokens), "val_tokens": len(val_tokens), "corpus_hash": corpus_hash, "valid_for_science": True, "smoke_test": False, "repeated_stream": False}
    tokenizer_manifest = {"tokenizer_type": "character", "vocab_size": len(vocab), "vocabulary_size": len(vocab), "tokenizer_hash": vocab_hash, "hash": vocab_hash, "special_tokens": {}}
    write_json(prep / "data_manifest.json", data_manifest); write_json(prep / "tokenizer_manifest.json", tokenizer_manifest)
    return TokenData(train_tokens, val_tokens, len(vocab), corpus_hash, prep, data_manifest, tokenizer_manifest)


def prepare_fineweb_gpt2_reproduction(data_root: Path, cfg: ExperimentConfig, docs: Iterable[str] | None = None) -> TokenData:
    try:
        import tiktoken
        tokenizer_name = cfg.tokenizer or "gpt2"
        if tokenizer_name != "gpt2":
            raise ValueError(f"fineweb_gpt2_reproduction requires tokenizer=gpt2, got {tokenizer_name!r}")
        enc = tiktoken.get_encoding(tokenizer_name)
        encode = enc.encode_ordinary
        n_vocab = enc.n_vocab
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load configured tokenizer {cfg.tokenizer or 'gpt2'!r} "
            f"(revision={cfg.tokenizer_revision or 'builtin-tiktoken-gpt2'!r}). "
            "Install tiktoken data/cache for this tokenizer or choose a loadable tokenizer in the experiment config; "
            "refusing to change tokenization silently."
        ) from exc
    prep = unique_dir(data_root / "fineweb_gpt2_reproduction", "prepared")
    source = docs if docs is not None else _iter_fineweb(cfg)
    train_tokens: list[int] = []; val_tokens: list[int] = []; corpus: list[str] = []
    train_docs = val_docs = 0
    for text in source:
        norm = " ".join(str(text).split())
        if not norm:
            continue
        corpus.append(norm)
        ids = encode(norm) + [FINEWEB_GPT2_EOT]
        if split_for_doc(f"{cfg.seeds[0]}:{norm}") == "val":
            val_tokens.extend(ids); val_docs += 1
        else:
            train_tokens.extend(ids); train_docs += 1
    if not train_tokens or not val_tokens:
        raise ValueError("fineweb_gpt2_reproduction requires both train and validation documents")
    np.save(prep / "train_tokens.npy", np.array(train_tokens, dtype=np.int64)); np.save(prep / "val_tokens.npy", np.array(val_tokens, dtype=np.int64))
    corpus_hash = sha256_bytes("\n".join(corpus).encode())
    tok_hash = sha256_bytes(b"tiktoken:gpt2:eot:50256")
    data_manifest = {"scientific_schema_version": 3, "data_mode": "fineweb_gpt2_reproduction", "dataset_name": cfg.dataset_name, "dataset_subset": cfg.dataset_subset or cfg.dataset_config, "dataset_config": cfg.dataset_config, "dataset_revision": cfg.dataset_revision, "split": cfg.dataset_split, "source_file_identifiers": _source_identifiers(cfg, corpus), "preparation_code_git_commit": _preparation_git_commit(), "document_assignment": "sha256-normalized-content-with-configured-seed", "eot_between_documents": FINEWEB_GPT2_EOT, "train_document_count": train_docs, "validation_document_count": val_docs, "train_tokens": len(train_tokens), "val_tokens": len(val_tokens), "corpus_hash": corpus_hash, "valid_for_science": True, "smoke_test": False, "repeated_stream": False}
    tokenizer_manifest = {"tokenizer_type": "tiktoken_gpt2", "tokenizer_name": cfg.tokenizer or "gpt2", "tokenizer_revision": cfg.tokenizer_revision or "builtin-tiktoken-gpt2", "vocab_size": n_vocab, "vocabulary_size": n_vocab, "vocabulary_hash": tok_hash, "tokenizer_hash": tok_hash, "special_token_ids": {"<|endoftext|>": FINEWEB_GPT2_EOT}, "dataset_revision": cfg.dataset_revision}
    write_json(prep / "data_manifest.json", data_manifest); write_json(prep / "tokenizer_manifest.json", tokenizer_manifest)
    return TokenData(train_tokens, val_tokens, n_vocab, corpus_hash, prep, data_manifest, tokenizer_manifest)


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
    try:
        ds = load_dataset(cfg.dataset_name, cfg.dataset_subset or cfg.dataset_config, split=cfg.dataset_split, revision=cfg.dataset_revision, streaming=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve dataset revision {cfg.dataset_revision!r} for "
            f"{cfg.dataset_name!r} subset={cfg.dataset_subset or cfg.dataset_config!r} split={cfg.dataset_split!r}; "
            "refusing to fall back to main. Check the pinned revision in the experiment config."
        ) from exc
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


def _log_prepare_progress(message: str) -> None:
    print(f"[wwgpt prepare-data] {message}", file=sys.stderr, flush=True)


def prepare_scientific_data(data_root: Path, level: int, token_multiplier: int, config_path: Path | None = None, docs: Iterable[str] | None = None, min_validation_tokens: int = 100_000) -> TokenData:
    cfg = load_config(config_path, level)
    model = GPT(cfg.model)
    report = model.parameter_report()
    param_count = getattr(report, f"{cfg.parameter_count_convention}_parameters", report.total_parameters)
    tokens_per_step = cfg.train.batch_size * cfg.model.block_size * cfg.train.gradient_accumulation
    requested = param_count * token_multiplier
    budget = plan_budget(param_count, token_multiplier, cfg.train.batch_size, cfg.model.block_size, cfg.train.gradient_accumulation, 10**18)
    realized = budget.realized_tokens
    needed_train = realized + 1
    prep = unique_dir(data_root / "fineweb_edu" / f"level_{level:02d}" / f"multiplier_{token_multiplier}", "prepared")
    _log_prepare_progress(f"starting level={level} token_multiplier={token_multiplier} requested_tokens={requested} realized_tokens={realized} output={prep}")
    train_docs: list[str] = []; val_docs: list[str] = []
    corpus = []
    source = docs if docs is not None else _iter_fineweb(cfg)
    tok: Tokenizer | None = None; train_tokens: list[int] = []; val_tokens: list[int] = []
    start_time = time.monotonic(); last_log_time = start_time; last_log_docs = 0
    for text in source:
        norm = " ".join(text.split())
        corpus.append(norm)
        if split_for_doc(norm) == "val": val_docs.append(norm)
        else: train_docs.append(norm)
        if tok is None and len(train_docs) >= 128:
            _log_prepare_progress(f"training BPE tokenizer after {len(train_docs)} train docs and {len(val_docs)} validation docs")
            tok = _train_bpe(train_docs, cfg.model.vocab_size)
            _log_prepare_progress(f"BPE tokenizer ready vocab_size={cfg.model.vocab_size}; collecting tokens")
        if tok is not None and not train_tokens:
            train_tokens = [i for d in train_docs for i in tok.encode(d).ids]
            val_tokens = [i for d in val_docs for i in tok.encode(d).ids]
        elif tok is not None and len(train_tokens) < realized:
            # Tokenize each whole document at most once; never wrap.
            if split_for_doc(norm) == "train": train_tokens.extend(tok.encode(norm).ids)
            elif len(val_tokens) < min_validation_tokens: val_tokens.extend(tok.encode(norm).ids)
        now = time.monotonic()
        docs_seen = len(train_docs) + len(val_docs)
        if now - last_log_time >= 30 or docs_seen - last_log_docs >= 10_000:
            elapsed = max(now - start_time, 1e-9)
            _log_prepare_progress(f"progress docs={docs_seen} train_docs={len(train_docs)} val_docs={len(val_docs)} train_tokens={len(train_tokens)}/{needed_train} val_tokens={len(val_tokens)}/{min_validation_tokens} elapsed_s={elapsed:.1f} docs_per_s={docs_seen / elapsed:.1f}")
            last_log_time = now; last_log_docs = docs_seen
        if len(train_tokens) >= realized and len(val_tokens) >= min_validation_tokens:
            _log_prepare_progress(f"collected enough tokens after docs={docs_seen}: train_tokens={len(train_tokens)} val_tokens={len(val_tokens)}")
            break
    if tok is None:
        if not train_docs:
            raise ValueError("insufficient unique training documents to train BPE tokenizer")
        tok = _train_bpe(train_docs, cfg.model.vocab_size)
        train_tokens = [i for d in train_docs for i in tok.encode(d).ids]
        val_tokens = [i for d in val_docs for i in tok.encode(d).ids]
    _log_prepare_progress(f"finished streaming docs={len(train_docs) + len(val_docs)} train_docs={len(train_docs)} val_docs={len(val_docs)} train_tokens={len(train_tokens)} val_tokens={len(val_tokens)}")
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
    data_manifest = {"scientific_schema_version": 3, "data_mode": "fineweb_custom_bpe_scaling", "model_architecture_version": cfg.model.model_architecture_version, "dataset_name": cfg.dataset_name, "dataset_subset": cfg.dataset_subset or cfg.dataset_config, "dataset_config": cfg.dataset_config, "dataset_revision": cfg.dataset_revision, "split": cfg.dataset_split, "source_file_identifiers": _source_identifiers(cfg, corpus), "preparation_code_git_commit": _preparation_git_commit(), "train_document_count": len(train_docs), "validation_document_count": len(val_docs), "unique_train_tokens": len(train_tokens), "validation_tokens": len(val_tokens), "min_validation_tokens": min_validation_tokens, "requested_tokens": requested, "realized_tokens": realized, "tokens_per_optimizer_step": tokens_per_step, "optimizer_steps": realized // tokens_per_step, "tokenizer_hash": tokenizer_hash, "corpus_hash": corpus_hash, "valid_for_science": True, "repeated_stream": False, "smoke_test": False, "parameter_report": model.report_dict(), "parameter_count_convention": cfg.parameter_count_convention}
    tokenizer_manifest = {"tokenizer_type": "custom_bpe_scaling", "tokenizer_name": "tokenizers.ByteLevelBPE-trained-from-configured-training-split", "tokenizer_revision": "prepared-locally", "experiment_label": "fineweb_custom_bpe_scaling", "not_reproduction_of_uploaded_fineweb_experiment": True, "vocabulary_size": cfg.model.vocab_size, "vocab_size": cfg.model.vocab_size, "vocabulary_hash": tokenizer_hash, "tokenizer_hash": tokenizer_hash, "special_token_ids": {s: tok.token_to_id(s) for s in ["<unk>", "<bos>", "<eos>", "<pad>"]}, "training_document_partition": "sha256-normalized-content", "dataset_name": cfg.dataset_name, "dataset_subset": cfg.dataset_subset or cfg.dataset_config, "dataset_config": cfg.dataset_config, "dataset_split": cfg.dataset_split, "dataset_revision": cfg.dataset_revision, "preparation_code_git_commit": _preparation_git_commit()}
    write_json(prep / "data_manifest.json", data_manifest); write_json(prep / "tokenizer_manifest.json", tokenizer_manifest)
    _log_prepare_progress(f"wrote train_tokens.npy, val_tokens.npy, tokenizer.json, and manifests under {prep}")
    return TokenData(train_tokens, val_tokens, cfg.model.vocab_size, corpus_hash, prep, data_manifest, tokenizer_manifest)


def load_prepared_scientific_data(data_root: Path, level: int, token_multiplier: int) -> TokenData:
    roots = sorted((data_root / "fineweb_edu" / f"level_{level:02d}" / f"multiplier_{token_multiplier}").glob("prepared_*"))
    for prep in reversed(roots):
        dm = json.loads((prep / "data_manifest.json").read_text()); tm = json.loads((prep / "tokenizer_manifest.json").read_text())
        if dm.get("valid_for_science") is True and dm.get("smoke_test") is False and tm.get("tokenizer_type") in {"BPE", "custom_bpe_scaling"} and (int(dm.get("scientific_schema_version", 2)) < 3 or dm.get("model_architecture_version") == load_config(None, level).model.model_architecture_version):
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


class RandomWindowTokenReader:
    def __init__(self, tokens: list[int], block_size: int, seed: int):
        self.tokens = tokens
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if len(self.tokens) < self.block_size + 1:
            raise ValueError("insufficient tokens for random windows")
        starts = self.rng.integers(0, len(self.tokens) - self.block_size - 1, size=batch_size)
        x = np.empty((batch_size, self.block_size), dtype=np.int64)
        y = np.empty((batch_size, self.block_size), dtype=np.int64)
        for i, st in enumerate(starts):
            arr = np.array(self.tokens[int(st):int(st) + self.block_size + 1], dtype=np.int64)
            x[i] = arr[:-1]
            y[i] = arr[1:]
        return x, y

    def state_dict(self) -> dict[str, object]:
        return {"bit_generator": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.rng.bit_generator.state = state["bit_generator"]

    @property
    def pos(self) -> int:
        return 0

    @pos.setter
    def pos(self, value: int) -> None:
        if value not in (0, None):
            raise ValueError("RandomWindowTokenReader has no sequential position")


def fixed_probe(tokens: list[int], block_size: int, batch_size: int, eval_batches: int) -> tuple[np.ndarray, np.ndarray, str]:
    need = batch_size * block_size * eval_batches + 1
    if len(tokens) < need:
        raise ValueError(f"insufficient probe tokens: {len(tokens)} < {need}")
    arr = np.array(tokens[:need], dtype=np.int64)
    x = arr[:-1].reshape(eval_batches, batch_size, block_size)
    y = arr[1:].reshape(eval_batches, batch_size, block_size)
    return x, y, sha256_bytes(arr.tobytes())


def stable_seed(*parts: object) -> int:
    return int(sha256_bytes("|".join(map(str, parts)).encode())[:16], 16) % (2**63 - 1)


def random_probe(tokens: list[int], block_size: int, batch_size: int, eval_batches: int, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    need = block_size + 1
    if len(tokens) < need:
        raise ValueError(f"insufficient probe tokens: {len(tokens)} < {need}")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - block_size - 1, size=batch_size * eval_batches)
    x = np.empty((eval_batches, batch_size, block_size), dtype=np.int64)
    y = np.empty((eval_batches, batch_size, block_size), dtype=np.int64)
    used = bytearray()
    for j, st in enumerate(starts):
        arr = np.array(tokens[int(st):int(st) + block_size + 1], dtype=np.int64)
        used.extend(arr.tobytes())
        e, b = divmod(j, batch_size)
        x[e, b] = arr[:-1]; y[e, b] = arr[1:]
    return x, y, sha256_bytes(bytes(used))
