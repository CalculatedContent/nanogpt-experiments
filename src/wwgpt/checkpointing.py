from __future__ import annotations
import json, os, time
from pathlib import Path
import torch

REQUIRED_COMPAT=("configuration_hash","data_hash","tokenizer_hash","initialization_hash","scientific_schema_version")

def atomic_torch_save(obj, path: Path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    with tmp.open('wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path); return path

def save_checkpoint(run_dir: Path, state: dict):
    ck=Path(run_dir)/"checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    step=int(state.get("step",0)); path=ck/f"checkpoint_step_{step:06d}.pt"
    state={**state,"saved_at":time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    atomic_torch_save(state,path)
    latest=ck/"latest.json"; tmp=latest.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({"checkpoint":path.name,"step":step}, indent=2)+"\n"); os.replace(tmp, latest)
    return path

def load_latest_checkpoint(run_dir: Path):
    ck=Path(run_dir)/"checkpoints"; latest=ck/"latest.json"
    if not latest.exists(): raise FileNotFoundError(f"missing latest checkpoint pointer: {latest}")
    meta=json.loads(latest.read_text()); return torch.load(ck/meta["checkpoint"], map_location="cpu", weights_only=False)

def compatibility_mismatches(checkpoint: dict, expected: dict):
    return {k:{"checkpoint":checkpoint.get("compatibility",{}).get(k),"expected":expected.get(k)} for k in REQUIRED_COMPAT if checkpoint.get("compatibility",{}).get(k)!=expected.get(k)}

def inspect_checkpoint(path: Path):
    obj=torch.load(Path(path), map_location="cpu", weights_only=False)
    return {k: obj.get(k) for k in ("step","tokens_processed","reader_position","projection_event_index","completed_projection_events","compatibility","saved_at")}

def validate_resume(run_dir: Path, expected: dict|None=None):
    ck=load_latest_checkpoint(run_dir); mm=compatibility_mismatches(ck, expected or ck.get("compatibility",{}))
    return {"compatible":not mm,"mismatches":mm,"next_step":int(ck.get("step",0))+1,"token_position":ck.get("reader_position"),"checkpoint_step":ck.get("step")}
