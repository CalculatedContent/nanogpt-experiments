from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wwgpt.data import (
    RandomWindowTokenReader,
    load_prepared_scientific_data,
    prepare_scientific_data,
    split_for_doc3,
    token_dtype_for_vocab,
)


def _cfg(tmp_path: Path, vocab_size: int = 64) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(f"""
model:
  n_layer: 1
  n_head: 1
  n_embd: 64
  block_size: 8
  vocab_size: {vocab_size}
train:
  batch_size: 2
  gradient_accumulation: 1
""")
    return p


def _docs() -> list[str]:
    docs = []
    i = 0
    while True:
        d = f"memmap synthetic document {i} " + ("alpha beta gamma delta " * 80)
        docs.append(d)
        counts = {sp: sum(1 for x in docs if split_for_doc3(x) == sp) for sp in ("train", "val", "test")}
        if counts["train"] >= 240 and counts["val"] >= 8 and counts["test"] >= 4:
            return docs
        i += 1


def test_dtype_selection():
    assert token_dtype_for_vocab(100) == np.dtype(np.uint16)
    assert token_dtype_for_vocab(65536) == np.dtype(np.uint16)
    assert token_dtype_for_vocab(65537) == np.dtype(np.uint32)


def test_memmap_manifest_reopen_and_no_document_overlap(tmp_path: Path):
    data = prepare_scientific_data(tmp_path, 0, 1, _cfg(tmp_path), _docs(), min_validation_tokens=1)
    assert isinstance(data.train, np.memmap)
    assert isinstance(data.val, np.memmap)
    assert isinstance(data.test, np.memmap)
    dm = data.data_manifest
    assert dm["dtype"] == "uint16"
    assert set(dm["splits"]) == {"train", "val", "test"}
    train_docs = set(dm["splits"]["train"]["document_sha256"])
    val_docs = set(dm["splits"]["val"]["document_sha256"])
    test_docs = set(dm["splits"]["test"]["document_sha256"])
    assert train_docs.isdisjoint(val_docs)
    assert train_docs.isdisjoint(test_docs)
    assert val_docs.isdisjoint(test_docs)
    reopened = load_prepared_scientific_data(tmp_path, 0, 1)
    assert isinstance(reopened.train, np.memmap)
    assert reopened.train.shape == tuple(dm["splits"]["train"]["shape"])
    np.testing.assert_array_equal(reopened.train[:20], data.train[:20])


def test_random_window_can_start_at_last_valid_index():
    tokens = np.arange(6, dtype=np.uint16)
    seen = set()
    r = RandomWindowTokenReader(tokens, block_size=4, seed=0)
    for _ in range(200):
        x, _ = r.next_batch(1)
        seen.add(int(x[0, 0]))
    assert seen == {0, 1}


def test_obsolete_prepared_format_errors(tmp_path: Path):
    prep = tmp_path / "fineweb_edu" / "level_00" / "multiplier_1" / "prepared_old"
    prep.mkdir(parents=True)
    np.save(prep / "train_tokens.npy", np.arange(10))
    np.save(prep / "val_tokens.npy", np.arange(10))
    (prep / "data_manifest.json").write_text(json.dumps({"valid_for_science": True, "smoke_test": False, "scientific_schema_version": 2, "corpus_hash": "x"}))
    (prep / "tokenizer_manifest.json").write_text(json.dumps({"tokenizer_type": "custom_bpe_scaling", "vocab_size": 64}))
    with pytest.raises(RuntimeError, match="obsolete prepared-data format"):
        load_prepared_scientific_data(tmp_path, 0, 1)


def test_manifest_validation_detects_corrupt_token_file(tmp_path: Path):
    data = prepare_scientific_data(tmp_path, 0, 1, _cfg(tmp_path), _docs(), min_validation_tokens=1)
    with (data.root / "val_tokens.bin").open("r+b") as f:
        f.write(b"xxxx")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        load_prepared_scientific_data(tmp_path, 0, 1)
