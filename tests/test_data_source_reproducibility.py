from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from wwgpt.config import load_config
from wwgpt.data import prepare_fineweb_gpt2_reproduction, prepare_scientific_data


def _tiny_config(path: Path, revision: str = "rev-fixture-123", split: str = "train[:1%]") -> Path:
    path.write_text(
        f"""
dataset_name: fixture/dataset
dataset_config: fixture-subset
dataset_subset: fixture-subset
dataset_split: {split}
dataset_revision: {revision}
model:
  n_layer: 1
  n_head: 1
  n_embd: 64
  block_size: 4
  vocab_size: 64
train:
  batch_size: 1
  gradient_accumulation: 1
  eval_batches: 1
""".lstrip()
    )
    return path


def _docs() -> list[str]:
    return [(f"configured revision document {i} " + ("alpha beta gamma " * 200)) for i in range(220)]


def test_prepare_scientific_data_persists_configured_source_and_tokenizer_identity(tmp_path):
    cfg = _tiny_config(tmp_path / "config.yaml", revision="pinned-rev")
    prepared = prepare_scientific_data(tmp_path, 0, 1, config_path=cfg, docs=_docs(), min_validation_tokens=1)

    dm = prepared.data_manifest
    tm = prepared.tokenizer_manifest
    assert dm["dataset_name"] == "fixture/dataset"
    assert dm["dataset_subset"] == "fixture-subset"
    assert dm["split"] == "train[:1%]"
    assert dm["dataset_revision"] == "pinned-rev"
    assert dm["source_file_identifiers"]["dataset_revision"] == "pinned-rev"
    assert dm["source_file_identifiers"]["document_sha256"]
    assert dm["preparation_code_git_commit"]
    assert tm["tokenizer_name"] == "tokenizers.ByteLevelBPE-trained-from-configured-training-split"
    assert tm["tokenizer_revision"] == "prepared-locally"
    assert tm["vocabulary_hash"] == tm["tokenizer_hash"]


def test_cli_prepare_data_propagates_configured_revision_without_full_download(tmp_path):
    cfg = _tiny_config(tmp_path / "config.yaml", revision="cli-rev")
    docs = tmp_path / "docs.txt"
    docs.write_text("\n".join(_docs()))
    data_root = tmp_path / "data"

    subprocess.run(
        [sys.executable, "-m", "wwgpt.cli", "prepare-data", "--config", str(cfg), "--level", "0", "--data-root", str(data_root), "--token-multiplier", "1", "--docs-file", str(docs)],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
    )
    manifest_path = next(data_root.glob("fineweb_edu/level_00/multiplier_1/prepared_*/data_manifest.json"))
    assert json.loads(manifest_path.read_text())["dataset_revision"] == "cli-rev"


@pytest.mark.parametrize(
    "script,args,subcommand",
    [
        ("scripts/download_data.sh", ["0", "DATA", "1", "custom.yaml"], "prepare-data"),
        ("scripts/run_five_seeds.sh", ["0", "DATA", "RESULTS", "1", "custom.yaml"], "run-canonical-trials"),
    ],
)
def test_shell_entry_points_pass_selected_config(tmp_path, script, args, subcommand):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "argv.json"
    stub = bin_dir / "wwgpt"
    stub.write_text(f"#!/usr/bin/env python3\nimport json,sys\nopen({str(log)!r},'w').write(json.dumps(sys.argv[1:]))\n")
    stub.chmod(0o755)
    subprocess.run([str(Path.cwd() / script), *args], check=True, env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    argv = json.loads(log.read_text())
    assert argv[0] == subcommand
    assert argv[argv.index("--config") + 1] == "custom.yaml"


def test_unresolvable_revision_fails_without_main_fallback(monkeypatch):
    cfg = replace(load_config(Path("configs/default.yaml"), level=0), dataset_revision="missing-rev")
    calls = []

    def fake_load_dataset(name, subset, *, split, revision, streaming):
        calls.append({"name": name, "subset": subset, "split": split, "revision": revision, "streaming": streaming})
        raise FileNotFoundError("revision missing")

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    with pytest.raises(RuntimeError, match="missing-rev.*refusing to fall back to main"):
        next(__import__("wwgpt.data", fromlist=["_iter_fineweb"]). _iter_fineweb(cfg))
    assert calls == [{"name": cfg.dataset_name, "subset": cfg.dataset_subset or cfg.dataset_config, "split": cfg.dataset_split, "revision": "missing-rev", "streaming": True}]


def test_tokenizer_load_failure_does_not_silently_change_tokenization(monkeypatch, tmp_path):
    cfg = replace(load_config(Path("configs/reproduction_fineweb.yaml"), level=0), tokenizer="not-gpt2")
    with pytest.raises(RuntimeError, match="Failed to load configured tokenizer.*refusing to change tokenization silently"):
        prepare_fineweb_gpt2_reproduction(tmp_path, cfg, docs=["a train doc", "a validation doc"])
