from __future__ import annotations
import json, os, time, random, hashlib
from pathlib import Path
from typing import Any
import numpy as np
import torch

SCIENTIFIC_CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_COMPAT=("configuration_hash","data_hash","tokenizer_hash","initialization_hash","model_configuration_hash","training_configuration_hash","wwpgd_configuration_hash","validation_probe_hash","training_probe_hash","scientific_schema_version")


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()

def rng_state() -> dict[str, Any]:
    out={"python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(), "torch_cpu_rng_state": torch.get_rng_state()}
    out["torch_cuda_rng_states"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return out

def restore_rng_state(state: dict[str, Any]) -> None:
    if "python_random_state" in state: random.setstate(state["python_random_state"])
    if "numpy_random_state" in state: np.random.set_state(state["numpy_random_state"])
    if "torch_cpu_rng_state" in state: torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng_states"):
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_states"])

def atomic_torch_save(obj, path: Path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    with tmp.open('wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path); return path

def save_checkpoint(run_dir: Path, state: dict):
    ck=Path(run_dir)/"checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    step=int(state.get("current_step", state.get("step",0)))
    full={**state,"step":step,"current_step":step,"next_step":int(state.get("next_step", step+1)),"checkpoint_schema_version":SCIENTIFIC_CHECKPOINT_SCHEMA_VERSION,"saved_at":time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    path=ck/f"checkpoint_step_{step:06d}.pt"
    atomic_torch_save(full,path)
    latest=ck/"latest.json"; tmp=latest.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({"checkpoint":path.name,"step":step,"next_step":full["next_step"],"saved_at":full["saved_at"]}, indent=2)+"\n"); os.replace(tmp, latest)
    return path

def load_latest_checkpoint(run_dir: Path):
    ck=Path(run_dir)/"checkpoints"; latest=ck/"latest.json"
    if not latest.exists(): raise FileNotFoundError(f"missing latest checkpoint pointer: {latest}")
    meta=json.loads(latest.read_text()); return torch.load(ck/meta["checkpoint"], map_location="cpu", weights_only=False)

def compatibility_mismatches(checkpoint: dict, expected: dict):
    got=checkpoint.get("compatibility",{})
    return {k:{"checkpoint":got.get(k),"expected":expected.get(k)} for k in REQUIRED_COMPAT if expected.get(k) is not None and got.get(k)!=expected.get(k)}

def assert_checkpoint_compatible(checkpoint: dict, expected: dict) -> None:
    mm=compatibility_mismatches(checkpoint, expected)
    if mm:
        raise RuntimeError("checkpoint compatibility validation failed: "+json.dumps(mm, sort_keys=True, default=str))

def inspect_checkpoint(path: Path):
    obj=torch.load(Path(path), map_location="cpu", weights_only=False)
    return {k: obj.get(k) for k in ("step","current_step","next_step","tokens_processed","training_reader_position","reader_position","gradient_accumulation_position","next_projection_event_index","completed_projection_event_indexes","compatibility","saved_at")}

def validate_resume(run_dir: Path, expected: dict|None=None):
    ck=load_latest_checkpoint(run_dir); exp=expected or ck.get("compatibility",{}); mm=compatibility_mismatches(ck, exp)
    return {"compatible":not mm,"mismatches":mm,"next_step":int(ck.get("next_step", int(ck.get("step",0))+1)),"token_position":ck.get("training_reader_position", ck.get("reader_position")),"checkpoint_step":ck.get("step")}
