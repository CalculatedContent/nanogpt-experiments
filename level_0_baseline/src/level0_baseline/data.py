from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from .config import roots

FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "593b3a867298afb8ce42625a270ef20ddcad28f9"
TOKEN_DTYPE = np.dtype("<u2")


class Encoder(Protocol):
    name: str
    eot_token: int
    n_vocab: int

    def encode_ordinary(self, text: str) -> list[int]: ...


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _progress_message(
    *,
    written_tokens: int,
    required_tokens: int,
    documents: int,
    elapsed_seconds: float,
    stalled_seconds: float,
    phase: str,
    split: str,
) -> str:
    elapsed_seconds = max(elapsed_seconds, 1e-9)
    rate = written_tokens / elapsed_seconds
    remaining = max(required_tokens - written_tokens, 0)
    eta = remaining / rate if rate > 0 else None
    percent = 100.0 * written_tokens / max(required_tokens, 1)
    return (
        "[level0-prepare-data] progress "
        f"phase={phase} split={split} documents={documents:,} "
        f"tokens={written_tokens:,}/{required_tokens:,} "
        f"percent={percent:5.1f}% elapsed={_format_duration(elapsed_seconds)} "
        f"speed={rate:,.0f} tok/s eta={_format_duration(eta)} "
        f"no_new_tokens_for={_format_duration(stalled_seconds)}"
    )


class ProgressReporter:
    """Emit heartbeats even when dataset resolution or streaming blocks."""

    def __init__(self, required_tokens: int, interval_seconds: float):
        self.required_tokens = int(required_tokens)
        self.interval_seconds = float(interval_seconds)
        self.started_at = time.monotonic()
        self.last_progress_at = self.started_at
        self.written_tokens = 0
        self.documents = 0
        self.phase = "starting"
        self.split = "none"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="level0-bpe-data-progress",
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

    def set_phase(self, phase: str, split: str | None = None) -> None:
        with self._lock:
            self.phase = str(phase)
            if split is not None:
                self.split = str(split)

    def update(self, *, documents: int, written_tokens: int, split: str) -> None:
        now = time.monotonic()
        with self._lock:
            if written_tokens > self.written_tokens:
                self.last_progress_at = now
            self.documents = int(documents)
            self.written_tokens = int(written_tokens)
            self.phase = "tokenizing"
            self.split = str(split)

    def snapshot(self) -> tuple[int, int, float, float, str, str]:
        now = time.monotonic()
        with self._lock:
            return (
                self.documents,
                self.written_tokens,
                now - self.started_at,
                now - self.last_progress_at,
                self.phase,
                self.split,
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            documents, written, elapsed, stalled, phase, split = self.snapshot()
            print(
                _progress_message(
                    written_tokens=written,
                    required_tokens=self.required_tokens,
                    documents=documents,
                    elapsed_seconds=elapsed,
                    stalled_seconds=stalled,
                    phase=phase,
                    split=split,
                ),
                file=sys.stderr,
                flush=True,
            )

    def stop(self) -> tuple[int, int, float, float, str, str]:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        return self.snapshot()


def load_tokenizer(name: str) -> Encoder:
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit(
            "Install BPE data support with: pip install -e '.[data]'"
        ) from exc
    encoding = tiktoken.get_encoding(name)
    return encoding


def _validate_targets(split_targets: dict[str, int]) -> OrderedDict[str, int]:
    expected = ("train", "val", "test")
    if tuple(split_targets) != expected:
        raise ValueError(f"split targets must be ordered as {expected}")
    out: OrderedDict[str, int] = OrderedDict()
    for name, value in split_targets.items():
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{name} token target must be positive")
        out[name] = parsed
    return out


def prepare_token_splits(
    texts: Iterable[str],
    output_dir: Path,
    split_targets: dict[str, int],
    tokenizer: Encoder,
    *,
    model_vocab_size: int,
    reporter: ProgressReporter | None = None,
    dataset_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Tokenize streamed documents into non-overlapping fixed BPE splits.

    A document is never shared across splits. If a document crosses a split
    boundary, only the prefix needed to finish the current split is retained and
    the remainder is discarded; the next split starts from the next document.
    """

    targets = _validate_targets(split_targets)
    if int(model_vocab_size) < int(tokenizer.n_vocab):
        raise ValueError("model_vocab_size must be at least tokenizer.n_vocab")
    if int(model_vocab_size) > 65_535:
        raise ValueError("model_vocab_size must fit in uint16 token files")

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        name: output_dir / f".{name}.bin.tmp" for name in targets
    }
    final_paths = {name: output_dir / f"{name}.bin" for name in targets}
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    handles = {
        name: open(path, "wb") for name, path in temporary_paths.items()
    }
    hashers = {name: hashlib.sha256() for name in targets}
    written = {name: 0 for name in targets}
    documents_by_split = {name: 0 for name in targets}
    discarded_boundary_tokens = 0
    documents_seen = 0
    split_names = list(targets)
    split_index = 0

    try:
        for text in texts:
            if split_index >= len(split_names):
                break
            documents_seen += 1
            token_ids = list(tokenizer.encode_ordinary(str(text)))
            token_ids.append(int(tokenizer.eot_token))
            if not token_ids:
                continue
            minimum = min(token_ids)
            maximum = max(token_ids)
            if minimum < 0 or maximum >= int(model_vocab_size):
                raise ValueError(
                    "tokenizer emitted an id outside the configured model vocabulary"
                )

            split = split_names[split_index]
            remaining = targets[split] - written[split]
            take = min(remaining, len(token_ids))
            if take:
                values = np.asarray(token_ids[:take], dtype=TOKEN_DTYPE)
                payload = values.tobytes(order="C")
                handles[split].write(payload)
                hashers[split].update(payload)
                written[split] += take
                documents_by_split[split] += 1
            if take < len(token_ids):
                discarded_boundary_tokens += len(token_ids) - take

            total_written = sum(written.values())
            if reporter is not None:
                reporter.update(
                    documents=documents_seen,
                    written_tokens=total_written,
                    split=split,
                )

            if written[split] == targets[split]:
                split_index += 1
                if split_index < len(split_names) and reporter is not None:
                    reporter.set_phase("tokenizing", split_names[split_index])
    finally:
        for handle in handles.values():
            handle.flush()
            handle.close()

    incomplete = {
        name: targets[name] - written[name]
        for name in targets
        if written[name] != targets[name]
    }
    if incomplete:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise RuntimeError(f"stream ended before fixed splits were filled: {incomplete}")

    for name in targets:
        final_paths[name].unlink(missing_ok=True)
        temporary_paths[name].replace(final_paths[name])

    metadata: dict[str, object] = {
        "format_version": 2,
        "tokenizer": f"tiktoken:{tokenizer.name}",
        "tokenizer_vocab_size": int(tokenizer.n_vocab),
        "model_vocab_size": int(model_vocab_size),
        "eot_token": int(tokenizer.eot_token),
        "dtype": TOKEN_DTYPE.str,
        "split_tokens": written,
        "split_documents": documents_by_split,
        "documents_seen": documents_seen,
        "discarded_boundary_tokens": discarded_boundary_tokens,
        "sha256": {name: hashers[name].hexdigest() for name in targets},
    }
    if dataset_metadata:
        metadata["dataset"] = dict(dataset_metadata)
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def _local_text_stream(path: Path) -> Iterable[str]:
    text = path.read_text(encoding="utf-8")
    while True:
        yield text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare fixed GPT-2-BPE FineWeb-Edu splits for Level 0"
    )
    parser.add_argument("--dataset", default="fineweb-edu")
    parser.add_argument("--output-dir")
    parser.add_argument("--train-tokens", type=int, default=20_000_000)
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--test-tokens", type=int, default=1_000_000)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--model-vocab-size", type=int, default=50_304)
    parser.add_argument("--local-text")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()

    if args.log_interval_seconds <= 0:
        parser.error("--log-interval-seconds must be greater than zero")
    if args.dataset != "fineweb-edu" and not args.local_text:
        parser.error("the isolated baseline currently supports --dataset fineweb-edu")

    output_dir = Path(args.output_dir) if args.output_dir else roots()["data"]
    targets = OrderedDict(
        train=args.train_tokens,
        val=args.val_tokens,
        test=args.test_tokens,
    )
    required_tokens = sum(targets.values())
    reporter = (
        ProgressReporter(required_tokens, args.log_interval_seconds)
        if args.verbose
        else None
    )
    if reporter is not None:
        reporter.start(output_dir)

    dataset = None
    try:
        if reporter is not None:
            reporter.set_phase("loading_tokenizer", "none")
        tokenizer = load_tokenizer(args.tokenizer)
        if args.verbose:
            print(
                "[level0-prepare-data] tokenizer ready "
                f"name={tokenizer.name} vocab={tokenizer.n_vocab:,} "
                f"eot={tokenizer.eot_token}",
                file=sys.stderr,
                flush=True,
            )

        if args.local_text:
            texts = _local_text_stream(Path(args.local_text))
            dataset_metadata = {
                "name": "local-text",
                "path": str(Path(args.local_text).resolve()),
            }
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise SystemExit(
                    "Install BPE data support with: pip install -e '.[data]'"
                ) from exc
            if reporter is not None:
                reporter.set_phase("resolving_dataset", "none")
            if args.verbose:
                print(
                    "[level0-prepare-data] resolving streamed dataset "
                    f"{FINEWEB_DATASET} {FINEWEB_CONFIG} train "
                    f"revision={FINEWEB_REVISION}",
                    file=sys.stderr,
                    flush=True,
                )
            dataset = load_dataset(
                FINEWEB_DATASET,
                name=FINEWEB_CONFIG,
                split="train",
                revision=FINEWEB_REVISION,
                streaming=True,
            )
            texts = (row["text"] for row in dataset)
            dataset_metadata = {
                "name": FINEWEB_DATASET,
                "config": FINEWEB_CONFIG,
                "split": "train",
                "revision": FINEWEB_REVISION,
                "streaming": True,
            }
            if args.verbose:
                print(
                    "[level0-prepare-data] dataset stream ready; tokenizing documents",
                    file=sys.stderr,
                    flush=True,
                )

        if reporter is not None:
            reporter.set_phase("tokenizing", "train")
        metadata = prepare_token_splits(
            texts,
            output_dir,
            targets,
            tokenizer,
            model_vocab_size=args.model_vocab_size,
            reporter=reporter,
            dataset_metadata=dataset_metadata,
        )
        if args.verbose:
            print(
                "[level0-prepare-data] complete "
                f"output={output_dir} "
                + " ".join(
                    f"{name}={count:,}"
                    for name, count in metadata["split_tokens"].items()
                ),
                file=sys.stderr,
                flush=True,
            )
    finally:
        if reporter is not None:
            reporter.stop()
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        dataset = None
        gc.collect()

    print(output_dir, flush=True)


if __name__ == "__main__":
    main()
