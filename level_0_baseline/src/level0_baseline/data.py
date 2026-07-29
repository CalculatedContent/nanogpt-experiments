from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

import numpy as np

from .config import load_config, roots

DATA_SCHEMA_VERSION = 2
UINT16_MAX = np.iinfo(np.uint16).max


class Tokenizer(Protocol):
    name: str
    n_vocab: int
    eot_token: int

    def encode_ordinary(self, text: str) -> list[int]: ...


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _progress_message(
    *,
    collected_tokens: int,
    required_tokens: int,
    documents: int,
    elapsed_seconds: float,
    stalled_seconds: float,
) -> str:
    elapsed_seconds = max(float(elapsed_seconds), 1e-9)
    rate = collected_tokens / elapsed_seconds
    remaining = max(required_tokens - collected_tokens, 0)
    eta = remaining / rate if rate > 0 else None
    percent = 100.0 * collected_tokens / max(required_tokens, 1)
    return (
        "[level0-prepare-data] progress "
        f"documents={documents:,} "
        f"tokens={collected_tokens:,}/{required_tokens:,} "
        f"percent={percent:5.1f}% "
        f"elapsed={_format_duration(elapsed_seconds)} "
        f"speed={rate:,.0f} tokens/s "
        f"eta={_format_duration(eta)} "
        f"no_new_tokens_for={_format_duration(stalled_seconds)}"
    )


class _ProgressReporter:
    """Emit heartbeats even while dataset metadata or streaming is blocked."""

    def __init__(self, required_tokens: int, interval_seconds: float):
        self.required_tokens = int(required_tokens)
        self.interval_seconds = float(interval_seconds)
        self.started_at = time.monotonic()
        self.last_progress_at = self.started_at
        self.collected_tokens = 0
        self.documents = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="level0-data-progress",
            daemon=True,
        )

    def start(self, output_dir: Path) -> None:
        print(
            "[level0-prepare-data] starting "
            f"required_tokens={self.required_tokens:,} output={output_dir}",
            file=sys.stderr,
            flush=True,
        )
        self._thread.start()

    def update(self, documents: int, collected_tokens: int) -> None:
        now = time.monotonic()
        with self._lock:
            if collected_tokens > self.collected_tokens:
                self.last_progress_at = now
            self.documents = int(documents)
            self.collected_tokens = int(collected_tokens)

    def _snapshot(self) -> tuple[int, int, float, float]:
        now = time.monotonic()
        with self._lock:
            return (
                self.documents,
                self.collected_tokens,
                now - self.started_at,
                now - self.last_progress_at,
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            documents, tokens, elapsed, stalled = self._snapshot()
            print(
                _progress_message(
                    collected_tokens=tokens,
                    required_tokens=self.required_tokens,
                    documents=documents,
                    elapsed_seconds=elapsed,
                    stalled_seconds=stalled,
                ),
                file=sys.stderr,
                flush=True,
            )

    def stop(self) -> tuple[int, int, float, float]:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        return self._snapshot()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tokenizer(name: str) -> Tokenizer:
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised by CLI users.
        raise SystemExit(
            "Install data support from level_0_baseline: "
            "python -m pip install -e '.[data]'"
        ) from exc
    tokenizer = tiktoken.get_encoding(name)
    if tokenizer.n_vocab - 1 > UINT16_MAX:
        raise ValueError(f"tokenizer {name!r} has IDs that do not fit in uint16")
    return tokenizer


def _fineweb_texts(
    load_dataset,
    *,
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    dataset_revision: str,
    verbose: bool,
) -> Iterator[str]:
    if verbose:
        print(
            "[level0-prepare-data] resolving streamed dataset "
            f"{dataset_name} {dataset_config} {dataset_split} "
            f"revision={dataset_revision}",
            file=sys.stderr,
            flush=True,
        )
    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=dataset_split,
        revision=dataset_revision,
        streaming=True,
    )
    if verbose:
        print(
            "[level0-prepare-data] dataset stream ready; tokenizing documents",
            file=sys.stderr,
            flush=True,
        )
    try:
        for row in dataset:
            text = row.get("text")
            if isinstance(text, str) and text:
                yield text
    finally:
        del dataset
        gc.collect()


def _compatible_metadata(
    metadata: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    keys = (
        "data_schema_version",
        "dataset_name",
        "dataset_config",
        "dataset_split",
        "dataset_revision",
        "tokenizer",
        "vocab_size",
        "dtype",
    )
    if any(metadata.get(key) != expected.get(key) for key in keys):
        return False
    return metadata.get("split_tokens") == expected.get("split_tokens")


def validate_prepared_data(
    output_dir: str | Path,
    *,
    expected: dict[str, Any] | None = None,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    metadata_path = output / "meta.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing prepared-data metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("data_schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError(
            "prepared data is not the corrected GPT-2-BPE Level 0 format; "
            "prepare a fresh dataset in a new directory"
        )
    if metadata.get("dtype") != "uint16":
        raise ValueError("prepared data must use uint16 GPT-2 token IDs")
    if expected is not None and not _compatible_metadata(metadata, expected):
        raise ValueError("prepared data does not match the requested identity")

    split_tokens = metadata.get("split_tokens") or {}
    hashes = metadata.get("sha256") or {}
    for split in ("train", "val", "test"):
        path = output / f"{split}.bin"
        if not path.is_file():
            raise FileNotFoundError(f"missing prepared split: {path}")
        expected_bytes = int(split_tokens.get(split, 0)) * np.dtype(np.uint16).itemsize
        if expected_bytes <= 0 or path.stat().st_size != expected_bytes:
            raise ValueError(
                f"prepared split {split} has {path.stat().st_size:,} bytes; "
                f"expected {expected_bytes:,}"
            )
        if verify_hashes and _sha256_file(path) != hashes.get(split):
            raise ValueError(f"prepared split hash mismatch: {split}")
    return metadata


def write_token_splits(
    texts: Iterable[str],
    output_dir: str | Path,
    *,
    tokenizer: Tokenizer,
    train_tokens: int,
    val_tokens: int,
    test_tokens: int,
    dataset_metadata: dict[str, str],
    verbose: bool = False,
    log_interval_seconds: float = 10.0,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_sizes = {
        "train": int(train_tokens),
        "val": int(val_tokens),
        "test": int(test_tokens),
    }
    required_tokens = sum(split_sizes.values())
    if required_tokens <= 0:
        raise ValueError("the requested token count must be positive")

    reporter = (
        _ProgressReporter(required_tokens, log_interval_seconds)
        if verbose
        else None
    )
    if reporter is not None:
        reporter.start(output)

    split_order = ("train", "val", "test")
    split_chunks: dict[str, list[np.ndarray]] = {name: [] for name in split_order}
    split_collected = {name: 0 for name in split_order}
    split_documents = {name: 0 for name in split_order}
    split_index = 0
    used_tokens = 0
    encoded_tokens = 0
    discarded_boundary_tokens = 0
    documents = 0
    snapshot: tuple[int, int, float, float] | None = None
    try:
        for text in texts:
            ids = tokenizer.encode_ordinary(text)
            ids.append(int(tokenizer.eot_token))
            if ids and max(ids) > UINT16_MAX:
                raise ValueError("token ID exceeds uint16 capacity")
            chunk = np.asarray(ids, dtype=np.uint16)
            encoded_tokens += int(chunk.size)
            documents += 1

            if split_index >= len(split_order):
                break
            split = split_order[split_index]
            remaining = split_sizes[split] - split_collected[split]
            take = min(remaining, int(chunk.size))
            if take > 0:
                split_chunks[split].append(chunk[:take])
                split_collected[split] += take
                split_documents[split] += 1
                used_tokens += take
            if take < int(chunk.size):
                # Never let one source document cross a scientific split. The
                # unused suffix is deliberately discarded at the boundary.
                discarded_boundary_tokens += int(chunk.size) - take
            if split_collected[split] == split_sizes[split]:
                split_index += 1

            if reporter is not None:
                reporter.update(documents, used_tokens)
            if split_index >= len(split_order):
                break
    finally:
        close = getattr(texts, "close", None)
        if callable(close):
            close()
        if reporter is not None:
            snapshot = reporter.stop()

    if split_collected != split_sizes:
        raise RuntimeError(
            "corpus did not supply all requested split tokens: "
            f"collected={split_collected}, requested={split_sizes}"
        )
    if verbose and snapshot is not None:
        print(
            "[level0-prepare-data] tokenization complete "
            f"documents={documents:,} used_tokens={used_tokens:,} "
            f"elapsed={_format_duration(snapshot[2])}; writing splits",
            file=sys.stderr,
            flush=True,
        )

    temporary_paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for split in split_order:
        temporary_path = output / f".{split}.bin.tmp"
        np.concatenate(split_chunks[split]).tofile(temporary_path)
        temporary_paths[split] = temporary_path
        hashes[split] = _sha256_file(temporary_path)
    for split in split_order:
        os.replace(temporary_paths[split], output / f"{split}.bin")

    metadata: dict[str, Any] = {
        "data_schema_version": DATA_SCHEMA_VERSION,
        **dataset_metadata,
        "tokenizer": tokenizer.name,
        "vocab_size": int(tokenizer.n_vocab),
        "eot_token": int(tokenizer.eot_token),
        "dtype": "uint16",
        "split_tokens": split_sizes,
        "split_documents": split_documents,
        "document_disjoint_splits": True,
        "documents_consumed": documents,
        "tokens_encoded": encoded_tokens,
        "tokens_used": used_tokens,
        "boundary_tokens_discarded": discarded_boundary_tokens,
        "sha256": hashes,
        "created_unix_time": time.time(),
    }
    temporary_metadata = output / ".meta.json.tmp"
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary_metadata, output / "meta.json")

    if verbose:
        print(
            "[level0-prepare-data] complete "
            f"output={output} "
            + " ".join(
                f"{split}={count:,}" for split, count in split_sizes.items()
            ),
            file=sys.stderr,
            flush=True,
        )
    return metadata


def expected_metadata(config: dict[str, Any], tokenizer: Tokenizer) -> dict[str, Any]:
    data = config["data"]
    return {
        "data_schema_version": DATA_SCHEMA_VERSION,
        "dataset_name": data["dataset_name"],
        "dataset_config": data["dataset_config"],
        "dataset_split": data["dataset_split"],
        "dataset_revision": data["dataset_revision"],
        "tokenizer": tokenizer.name,
        "vocab_size": int(tokenizer.n_vocab),
        "dtype": data["dtype"],
        "split_tokens": {
            "train": int(data["train_tokens"]),
            "val": int(data["val_tokens"]),
            "test": int(data["test_tokens"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/level0.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--local-text")
    parser.add_argument("--train-tokens", type=int)
    parser.add_argument("--val-tokens", type=int)
    parser.add_argument("--test-tokens", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.log_interval_seconds <= 0:
        parser.error("--log-interval-seconds must be greater than zero")

    config = load_config(args.config)
    data = config["data"]
    if args.dataset not in (None, "fineweb-edu"):
        parser.error("the isolated baseline supports only --dataset fineweb-edu")
    for argument, key in (
        (args.train_tokens, "train_tokens"),
        (args.val_tokens, "val_tokens"),
        (args.test_tokens, "test_tokens"),
    ):
        if argument is not None:
            data[key] = int(argument)

    resolved_roots = roots()
    output = Path(args.output_dir) if args.output_dir else resolved_roots["data"]
    cache_root = resolved_roots["cache"]
    cache_root.mkdir(parents=True, exist_ok=True)
    huggingface_root = cache_root / "huggingface"
    os.environ.setdefault("HF_HOME", str(huggingface_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(huggingface_root / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(huggingface_root / "hub"))
    os.environ.setdefault(
        "HUGGINGFACE_HUB_CACHE",
        str(huggingface_root / "hub"),
    )
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache_root / "tiktoken"))
    Path(os.environ["TIKTOKEN_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(data["tokenizer"])
    expected = expected_metadata(config, tokenizer)
    if not args.force and (output / "meta.json").is_file():
        try:
            validate_prepared_data(
                output,
                expected=expected,
                verify_hashes=True,
            )
        except (FileNotFoundError, ValueError):
            pass
        else:
            print(
                f"[level0-prepare-data] compatible data already exists: {output}",
                file=sys.stderr,
                flush=True,
            )
            print(output)
            return

    if args.local_text:
        text = Path(args.local_text).read_text(encoding="utf-8")

        def repeat_text() -> Iterator[str]:
            while True:
                yield text

        texts: Iterable[str] = repeat_text()
        dataset_metadata = {
            "dataset_name": "local-text",
            "dataset_config": "local-text",
            "dataset_split": "local-text",
            "dataset_revision": _sha256_file(Path(args.local_text)),
        }
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - exercised by CLI users.
            raise SystemExit(
                "Install data support from level_0_baseline: "
                "python -m pip install -e '.[data]'"
            ) from exc
        dataset_metadata = {
            "dataset_name": data["dataset_name"],
            "dataset_config": data["dataset_config"],
            "dataset_split": data["dataset_split"],
            "dataset_revision": data["dataset_revision"],
        }
        texts = _fineweb_texts(
            load_dataset,
            dataset_name=data["dataset_name"],
            dataset_config=data["dataset_config"],
            dataset_split=data["dataset_split"],
            dataset_revision=data["dataset_revision"],
            verbose=args.verbose,
        )

    write_token_splits(
        texts,
        output,
        tokenizer=tokenizer,
        train_tokens=int(data["train_tokens"]),
        val_tokens=int(data["val_tokens"]),
        test_tokens=int(data["test_tokens"]),
        dataset_metadata=dataset_metadata,
        verbose=args.verbose,
        log_interval_seconds=args.log_interval_seconds,
    )
    print(output)


if __name__ == "__main__":
    main()
