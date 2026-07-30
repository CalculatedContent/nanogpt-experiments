from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path
from collections.abc import Iterable
from typing import Protocol

import numpy as np

from .config import roots


DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "593b3a867298afb8ce42625a270ef20ddcad28f9"
TOKENIZER_NAME = "gpt2"
TOKEN_DTYPE = np.dtype(np.uint16)


class Encoder(Protocol):
    n_vocab: int
    eot_token: int

    def encode_ordinary(self, text: str) -> list[int]: ...


def _format_duration(seconds: float) -> str:
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
    elapsed_seconds = max(elapsed_seconds, 1e-9)
    rate = collected_tokens / elapsed_seconds
    remaining = max(required_tokens - collected_tokens, 0)
    eta = remaining / rate if rate > 0 else None
    percent = 100.0 * collected_tokens / max(required_tokens, 1)
    eta_text = _format_duration(eta) if eta is not None else "unknown"
    return (
        "[level0-prepare-data] progress "
        f"documents={documents:,} "
        f"tokens={collected_tokens:,}/{required_tokens:,} "
        f"percent={percent:5.1f}% "
        f"elapsed={_format_duration(elapsed_seconds)} "
        f"speed={rate:,.0f} tok/s "
        f"eta={eta_text} "
        f"no_new_tokens_for={_format_duration(stalled_seconds)}"
    )


class _ProgressReporter:
    """Emit a heartbeat even when the remote streaming iterator is blocked."""

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
            documents, collected_tokens, elapsed, stalled = self._snapshot()
            print(
                _progress_message(
                    collected_tokens=collected_tokens,
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


def _encode_document(text: str, encoder: Encoder) -> np.ndarray:
    token_ids = encoder.encode_ordinary(text)
    token_ids.append(int(encoder.eot_token))
    if token_ids and max(token_ids) > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError("token id exceeds uint16 storage capacity")
    return np.asarray(token_ids, dtype=TOKEN_DTYPE)


def write_token_splits(
    texts: Iterable[str],
    encoder: Encoder,
    out: Path,
    train_tokens: int,
    val_tokens: int,
    test_tokens: int,
    *,
    verbose: bool = False,
    log_interval_seconds: float = 10.0,
    reporter: _ProgressReporter | None = None,
    dataset_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    targets = {
        "train": int(train_tokens),
        "val": int(val_tokens),
        "test": int(test_tokens),
    }
    if any(value <= 0 for value in targets.values()):
        raise ValueError("all split token counts must be positive")
    required_tokens = sum(targets.values())
    own_reporter = reporter is None and verbose
    active_reporter = reporter
    if own_reporter:
        active_reporter = _ProgressReporter(required_tokens, log_interval_seconds)
        active_reporter.start(out)

    partial_paths = {name: out / f"{name}.bin.partial" for name in targets}
    final_paths = {name: out / f"{name}.bin" for name in targets}
    handles = {name: path.open("wb") for name, path in partial_paths.items()}
    written = {name: 0 for name in targets}
    split_document_counts = {name: 0 for name in targets}
    split_names = list(targets)
    split_index = 0
    documents = 0
    total_tokens = 0
    source_utf8_bytes = 0

    try:
        for text in texts:
            if split_index >= len(split_names):
                break
            documents += 1
            source_utf8_bytes += len(text.encode("utf-8", errors="replace"))
            encoded = _encode_document(text, encoder)
            split = split_names[split_index]
            remaining = targets[split] - written[split]
            take = min(remaining, len(encoded))
            if take > 0:
                encoded[:take].tofile(handles[split])
                written[split] += take
                total_tokens += take
                split_document_counts[split] += 1
            # Never carry the remainder of a document into another split.
            # This keeps train, validation, and test document-disjoint.
            if written[split] == targets[split]:
                split_index += 1
            if active_reporter is not None:
                active_reporter.update(documents, total_tokens)
            if total_tokens >= required_tokens:
                break
    finally:
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        if own_reporter and active_reporter is not None:
            snapshot = active_reporter.stop()
        else:
            snapshot = active_reporter._snapshot() if active_reporter else None

    if total_tokens < required_tokens:
        for path in partial_paths.values():
            path.unlink(missing_ok=True)
        raise RuntimeError(
            f"corpus supplied {total_tokens:,} tokens; need {required_tokens:,}"
        )

    for name in split_names:
        os.replace(partial_paths[name], final_paths[name])

    elapsed = snapshot[2] if snapshot is not None else 0.0
    metadata: dict[str, object] = {
        "schema_version": 2,
        "tokenizer": TOKENIZER_NAME,
        "vocab_size": int(encoder.n_vocab),
        "eot_token": int(encoder.eot_token),
        "dtype": TOKEN_DTYPE.name,
        "splits": written,
        "split_document_counts": split_document_counts,
        "document_disjoint_splits": True,
        "documents_consumed": documents,
        "source_utf8_bytes": source_utf8_bytes,
        "elapsed_seconds": elapsed,
    }
    if dataset_metadata:
        metadata.update(dataset_metadata)
    (out / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    if verbose:
        print(
            "[level0-prepare-data] complete "
            f"documents={documents:,} tokens={total_tokens:,} "
            f"elapsed={_format_duration(elapsed)} output={out}",
            file=sys.stderr,
            flush=True,
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fineweb-edu")
    parser.add_argument("--output-dir")
    parser.add_argument("--train-tokens", type=int, default=10_000_000)
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--test-tokens", type=int, default=1_000_000)
    parser.add_argument("--local-text")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print token progress, elapsed time, throughput, ETA, and stall heartbeats",
    )
    parser.add_argument(
        "--log-interval-seconds",
        type=float,
        default=10.0,
        help="heartbeat interval used with --verbose (default: 10 seconds)",
    )
    args = parser.parse_args()
    if args.log_interval_seconds <= 0:
        parser.error("--log-interval-seconds must be greater than zero")
    if args.dataset != "fineweb-edu" and not args.local_text:
        parser.error("the isolated baseline currently supports --dataset fineweb-edu")

    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit("Install data support: pip install -e '.[data]'") from exc

    encoder = tiktoken.get_encoding(TOKENIZER_NAME)
    if encoder.n_vocab > np.iinfo(TOKEN_DTYPE).max:
        raise RuntimeError("GPT-2 vocabulary no longer fits uint16 storage")

    out = Path(args.output_dir) if args.output_dir else roots()["data"]
    required_tokens = args.train_tokens + args.val_tokens + args.test_tokens
    reporter = (
        _ProgressReporter(required_tokens, args.log_interval_seconds)
        if args.verbose
        else None
    )
    if reporter is not None:
        reporter.start(out)

    dataset = None
    texts = None
    try:
        if args.local_text:
            text = Path(args.local_text).read_text(encoding="utf-8")

            def repeat() -> Iterable[str]:
                while True:
                    yield text

            texts = repeat()
            dataset_metadata = {
                "dataset_name": "local_text",
                "dataset_config": None,
                "dataset_revision": None,
            }
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise SystemExit(
                    "Install data support: pip install -e '.[data]'"
                ) from exc
            if args.verbose:
                print(
                    "[level0-prepare-data] resolving pinned FineWeb-Edu stream",
                    file=sys.stderr,
                    flush=True,
                )
            dataset = load_dataset(
                DATASET_NAME,
                name=DATASET_CONFIG,
                split="train",
                revision=DATASET_REVISION,
                streaming=True,
            )
            if args.verbose:
                print(
                    "[level0-prepare-data] stream ready; GPT-2 BPE tokenization started",
                    file=sys.stderr,
                    flush=True,
                )
            texts = (row["text"] for row in dataset)
            dataset_metadata = {
                "dataset_name": DATASET_NAME,
                "dataset_config": DATASET_CONFIG,
                "dataset_revision": DATASET_REVISION,
            }

        write_token_splits(
            texts,
            encoder,
            out,
            args.train_tokens,
            args.val_tokens,
            args.test_tokens,
            verbose=args.verbose,
            log_interval_seconds=args.log_interval_seconds,
            reporter=reporter,
            dataset_metadata=dataset_metadata,
        )
    finally:
        if reporter is not None:
            reporter.stop()
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        texts = None
        dataset = None
        gc.collect()

    print(out, flush=True)


if __name__ == "__main__":
    main()
