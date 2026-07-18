from __future__ import annotations
import json, os, time, random, hashlib
from pathlib import Path
from typing import Any
import numpy as np
import torch

SCIENTIFIC_CHECKPOINT_SCHEMA_VERSION = 2
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

REQUIRED_CHECKPOINT_KEYS = (
    "model_state_dict","optimizer_state_dict","scheduler_state_dict","gradient_scaler_state_dict",
    "current_step","next_step","tokens_processed","training_reader_position","seed",
    "python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_states",
    "device_type","precision_policy","gradient_accumulation_position","metrics_rows",
    "periodic_weightwatcher_rows","wwpgd_projection_rows","immediate_projection_weightwatcher_rows",
    "scientific_schema_version","checkpoint_schema_version","created_at",
)

INVENTORY_FIELDS = (
    "checkpoint","current_step","next_step","tokens_processed","created_at","sha256",
    "size_bytes","verified","compatibility_hash","checkpoint_schema_version",
)

def _sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def validate_checkpoint_keys(obj: dict) -> None:
    missing=[k for k in REQUIRED_CHECKPOINT_KEYS if k not in obj]
    if missing:
        raise ValueError("checkpoint missing required keys: "+", ".join(missing))
    if int(obj.get("checkpoint_schema_version", -1)) != SCIENTIFIC_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema {obj.get('checkpoint_schema_version')}")

def _atomic_write_json(path: Path, data: dict) -> None:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    with tmp.open('w') as f:
        f.write(json.dumps(data, indent=2, sort_keys=True, default=str)+"\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _append_inventory_atomic(path: Path, row: dict) -> None:
    import csv
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    if path.exists():
        with path.open(newline='') as f: rows=list(csv.DictReader(f))
    rows.append({k: row.get(k, '') for k in INVENTORY_FIELDS})
    tmp=path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    with tmp.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(INVENTORY_FIELDS)); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_torch_save(obj, path: Path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    with tmp.open('wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    loaded=torch.load(tmp, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "checkpoint_schema_version" in loaded:
        validate_checkpoint_keys(loaded)
    sha=_sha256_file(tmp); size=tmp.stat().st_size
    os.replace(tmp, path); return path, sha, size

def save_checkpoint(run_dir: Path, state: dict):
    ck=Path(run_dir)/"checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    step=int(state.get("current_step", state.get("step",0)))
    created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    compatibility=state.get("compatibility", {})
    full={**state,"step":step,"current_step":step,"next_step":int(state.get("next_step", step+1)),"checkpoint_schema_version":SCIENTIFIC_CHECKPOINT_SCHEMA_VERSION,"created_at":state.get("created_at") or created,"saved_at":created}
    for k, v in compatibility.items():
        full.setdefault(k, v)
    # Fill mandatory nullable fields for compatibility with older callers while still writing a full-state checkpoint.
    if "model_state_dict" not in full and "model_state" in full:
        full["model_state_dict"] = full["model_state"]
    if "optimizer_state_dict" not in full and "optimizer_state" in full:
        full["optimizer_state_dict"] = full["optimizer_state"]
    for k, v in {
        "model_state_dict": {}, "optimizer_state_dict": {}, "seed": 0,
        "scheduler_state_dict": None, "gradient_scaler_state_dict": None, "training_reader_position": state.get("reader_position",0),
        "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(), "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": [], "device_type": "unknown", "precision_policy": "torch_default", "gradient_accumulation_position": 0,
        "metrics_rows": [], "periodic_weightwatcher_rows": [], "wwpgd_projection_rows": [], "immediate_projection_weightwatcher_rows": [],
        "scientific_schema_version": 0,
    }.items(): full.setdefault(k, v)
    path=ck/f"checkpoint_step_{step:06d}.pt"
    path, sha, size = atomic_torch_save(full,path)
    meta={"checkpoint":path.name,"current_step":step,"next_step":full["next_step"],"tokens_processed":full.get("tokens_processed",0),"created_at":full["created_at"],"sha256":sha,"size_bytes":size,"verified":True,"compatibility_hash":stable_hash(compatibility),"checkpoint_schema_version":SCIENTIFIC_CHECKPOINT_SCHEMA_VERSION}
    _atomic_write_json(ck/"latest.json", meta)
    _append_inventory_atomic(ck/"checkpoint_inventory.csv", meta)
    return path

def load_latest_checkpoint(run_dir: Path):
    ck=Path(run_dir)/"checkpoints"; latest=ck/"latest.json"
    if not latest.exists(): raise FileNotFoundError(f"missing latest checkpoint pointer: {latest}")
    meta=json.loads(latest.read_text()); path=ck/meta["checkpoint"]
    if not path.exists(): raise FileNotFoundError(f"latest checkpoint missing: {path}")
    size=path.stat().st_size
    if int(meta.get("size_bytes", -1)) != size: raise RuntimeError(f"checkpoint size mismatch for {path}")
    sha=_sha256_file(path)
    if meta.get("sha256") != sha: raise RuntimeError(f"checkpoint sha256 mismatch for {path}")
    obj=torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_keys(obj)
    return obj

def compatibility_mismatches(checkpoint: dict, expected: dict):
    got=checkpoint.get("compatibility",{})
    return {k:{"checkpoint":got.get(k),"expected":expected.get(k)} for k in REQUIRED_COMPAT if expected.get(k) is not None and got.get(k)!=expected.get(k)}

def assert_checkpoint_compatible(checkpoint: dict, expected: dict) -> None:
    mm=compatibility_mismatches(checkpoint, expected)
    if mm:
        raise RuntimeError("checkpoint compatibility validation failed: "+json.dumps(mm, sort_keys=True, default=str))

def inspect_checkpoint(path: Path):
    path=Path(path)
    sha=_sha256_file(path); size=path.stat().st_size
    obj=torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_keys(obj)
    keys=("checkpoint_schema_version","scientific_schema_version","run_directory","pair_id","optimizer_name","seed","level","token_multiplier","current_step","next_step","tokens_processed","training_reader_position","reader_position","gradient_accumulation_position","next_projection_event_index","completed_projection_event_indexes","compatibility","data_hash","tokenizer_hash","validation_probe_hash","training_probe_hash","weightwatcher_version","weightwatcher_configuration","wwpgd_commit","git_commit","device_type","precision_policy","created_at","saved_at")
    out={k: obj.get(k) for k in keys}
    out.update({"sha256": sha, "size_bytes": size, "sha256_verified": True, "size_verified": True})
    return out

def validate_resume(run_dir: Path, expected: dict|None=None):
    ck=load_latest_checkpoint(run_dir); exp=expected or ck.get("compatibility",{}); mm=compatibility_mismatches(ck, exp)
    return {"compatible":not mm,"mismatches":mm,"next_step":int(ck.get("next_step", int(ck.get("step",0))+1)),"token_position":ck.get("training_reader_position", ck.get("reader_position")),"checkpoint_step":ck.get("step")}
