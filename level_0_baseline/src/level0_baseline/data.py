from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .config import roots


def encode(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8)


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
    collected_bytes: int,
    required_bytes: int,
    documents: int,
    elapsed_seconds: float,
    stalled_seconds: float,
) -> str:
    elapsed_seconds = max(elapsed_seconds, 1e-9)
    rate = collected_bytes / elapsed_seconds
    remaining = max(required_bytes - collected_bytes, 0)
    eta = remaining / rate if rate > 0 else None
    percent = 100.0 * collected_bytes / max(required_bytes, 1)
    speed_mib = rate / (1024 * 1024)
    eta_text = _format_duration(eta) if eta is not None else "unknown"
    stall_text = _format_duration(stalled_seconds)
    return (
        "[level0-prepare-data] progress "
        f"documents={documents:,} "
        f"bytes={collected_bytes:,}/{required_bytes:,} "
        f"percent={percent:5.1f}% "
        f"elapsed={_format_duration(elapsed_seconds)} "
        f"speed={speed_mib:.2f} MiB/s "
        f"eta={eta_text} "
        f"no_new_bytes_for={stall_text}"
    )


class _ProgressReporter:
    """Emit heartbeat logs even while the streaming iterator is blocked."""

    def __init__(self, required_bytes: int, interval_seconds: float):
        self.required_bytes = int(required_bytes)
        self.interval_seconds = float(interval_seconds)
        self.started_at = time.monotonic()
        self.last_progress_at = self.started_at
        self.collected_bytes = 0
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
            f"required_bytes={self.required_bytes:,} output={output_dir}",
            file=sys.stderr,
            flush=True,
        )
        self._thread.start()

    def update(self, documents: int, collected_bytes: int) -> None:
        now = time.monotonic()
        with self._lock:
            if collected_bytes > self.collected_bytes:
                self.last_progress_at = now
            self.documents = int(documents)
            self.collected_bytes = int(collected_bytes)

    def _snapshot(self) -> tuple[int, int, float, float]:
        now = time.monotonic()
        with self._lock:
            return (
                self.documents,
                self.collected_bytes,
                now - self.started_at,
                now - self.last_progress_at,
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            documents, collected_bytes, elapsed, stalled = self._snapshot()
            print(
                _progress_message(
                    collected_bytes=collected_bytes,
                    required_bytes=self.required_bytes,
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


def write_splits(
    texts,
    out: Path,
    train_bytes: int,
    val_bytes: int,
    test_bytes: int,
    *,
    verbose: bool = False,
    log_interval_seconds: float = 10.0,
):
    out.mkdir(parents=True, exist_ok=True)
    need = train_bytes + val_bytes + test_bytes
    chunks = []
    total = 0
    documents = 0
    reporter = (
        _ProgressReporter(need, log_interval_seconds) if verbose else None
    )
    if reporter is not None:
        reporter.start(out)

    try:
        for text in texts:
            documents += 1
            x = encode(text + "\n")
            chunks.append(x)
            total += len(x)
            if reporter is not None:
                reporter.update(documents, total)
            if total >= need:
                break
    finally:
        snapshot = reporter.stop() if reporter is not None else None

    if total < need:
        raise RuntimeError(f"corpus supplied {total:,} bytes; need {need:,}")

    if verbose and snapshot is not None:
        _, _, elapsed, _ = snapshot
        print(
            "[level0-prepare-data] collection complete "
            f"documents={documents:,} collected_bytes={total:,} "
            f"elapsed={_format_duration(elapsed)}; writing fixed splits",
            file=sys.stderr,
            flush=True,
        )

    all_tokens = np.concatenate(chunks)[:need]
    boundaries = {
        "train": (0, train_bytes),
        "val": (train_bytes, train_bytes + val_bytes),
        "test": (train_bytes + val_bytes, need),
    }
    for name, (start, end) in boundaries.items():
        all_tokens[start:end].tofile(out / f"{name}.bin")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "tokenizer": "utf8-byte",
                "vocab_size": 256,
                "sizes": {
                    name: end - start
                    for name, (start, end) in boundaries.items()
                },
            },
            indent=2,
        )
    )

    if verbose:
        sizes = ", ".join(
            f"{name}={end - start:,}"
            for name, (start, end) in boundaries.items()
        )
        print(
            f"[level0-prepare-data] complete output={out} {sizes}",
            file=sys.stderr,
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fineweb-edu")
    parser.add_argument("--output-dir")
    parser.add_argument("--train-bytes", type=int, default=50_000_000)
    parser.add_argument("--val-bytes", type=int, default=2_000_000)
    parser.add_argument("--test-bytes", type=int, default=2_000_000)
    parser.add_argument("--local-text")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print streaming progress, elapsed time, throughput, ETA, and stall heartbeats",
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

    out = Path(args.output_dir) if args.output_dir else roots()["data"]
    if args.local_text:
        text = Path(args.local_text).read_text(encoding="utf-8")

        def repeat():
            while True:
                yield text

        texts = repeat()
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("Install data support: pip install -e '.[data]'") from exc
        if args.verbose:
            print(
                "[level0-prepare-data] resolving streamed dataset "
                "HuggingFaceFW/fineweb-edu sample-10BT train",
                file=sys.stderr,
                flush=True,
            )
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        if args.verbose:
            print(
                "[level0-prepare-data] dataset stream ready; collecting documents",
                file=sys.stderr,
                flush=True,
            )
        texts = (row["text"] for row in dataset)

    write_splits(
        texts,
        out,
        args.train_bytes,
        args.val_bytes,
        args.test_bytes,
        verbose=args.verbose,
        log_interval_seconds=args.log_interval_seconds,
    )
    print(out)


if __name__ == "__main__":
    main()
