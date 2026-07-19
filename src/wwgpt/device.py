from __future__ import annotations
import json, os, platform, sys
from pathlib import Path
import torch

VALID_DEVICES={"auto","cuda","mps","xla","cpu"}

def _xla_device():
    try:
        import torch_xla.core.xla_model as xm
        dev=xm.xla_device()
        t=torch.tensor([1.0], device=dev); _=(t+1).cpu()
        return dev
    except Exception:
        return None

def detect_device(requested: str|None="auto"):
    req=requested or "auto"
    if req not in VALID_DEVICES: raise ValueError(f"unsupported device {req}; expected one of {sorted(VALID_DEVICES)}")
    if req=="cpu": return torch.device("cpu")
    if req=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if req=="mps":
        if not (hasattr(torch.backends,"mps") and torch.backends.mps.is_available()): raise RuntimeError("requested MPS but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if req=="xla":
        dev=_xla_device()
        if dev is None: raise RuntimeError("requested XLA/TPU but an actual XLA device could not be initialized")
        return dev
    dev=_xla_device()
    if dev is not None: return dev
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def resolve_device(requested: str|None="auto"):
    return detect_device(requested)

def precision_policy(device=None):
    dev=device or detect_device("auto"); typ=getattr(dev,"type",str(dev).split(':')[0])
    if typ=="cuda":
        bf16=False
        try: bf16=torch.cuda.is_bf16_supported()
        except Exception: bf16=False
        return {"dtype":"bfloat16" if bf16 else "float32", "mixed_precision":"bf16" if bf16 else "none"}
    if typ=="xla": return {"dtype":"bfloat16", "mixed_precision":"bf16_when_configured"}
    return {"dtype":"float32", "mixed_precision":"none"}

def device_summary(requested: str|None="auto"):
    dev=detect_device(requested); typ=getattr(dev,"type",str(dev).split(':')[0]); name=typ; count=1
    if typ=="cuda": name=torch.cuda.get_device_name(0); count=torch.cuda.device_count()
    elif typ=="mps": name="Apple Silicon MPS"
    pol=precision_policy(dev)
    try:
        import torch_xla; xla_ver=getattr(torch_xla,"__version__","unknown")
    except Exception: xla_ver=None
    return {"requested_device":requested or "auto","resolved_device":str(dev),"device_type":typ,"device_name":name,"accelerator_count":count,"pytorch_version":torch.__version__,"cuda_version":torch.version.cuda,"mps_available":bool(hasattr(torch.backends,"mps") and torch.backends.mps.is_available()),"torch_xla_version":xla_ver,"precision_selected":pol["dtype"],"mixed_precision_policy":pol["mixed_precision"],"memory_information":memory_stats(dev),"operating_system":platform.platform(),"python_version":sys.version}

def synchronize_device(device=None):
    dev=device or detect_device("auto"); typ=getattr(dev,"type",str(dev).split(':')[0])
    if typ=="cuda": torch.cuda.synchronize(dev)
    elif typ=="mps" and hasattr(torch,"mps"): torch.mps.synchronize()

def memory_stats(device=None):
    dev=device or torch.device("cpu"); typ=getattr(dev,"type",str(dev).split(':')[0])
    if typ=="cuda": return {"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved()}
    return {}

def save_device_manifest(path: Path, requested: str|None="auto"):
    path=Path(path); path.write_text(json.dumps(device_summary(requested), indent=2, sort_keys=True)+"\n")
    return path

def run_device_preflight(output: Path | None = None, requested: str | None = "auto"):
    import time
    from dataclasses import asdict
    from wwgpt.model import GPT
    from wwgpt.config import ModelConfig
    from wwgpt.checkpointing import save_checkpoint, load_latest_checkpoint, rng_state, restore_rng_state
    from wwgpt.ww import measured_projection_spectral_rows, weightwatcher_details, apply_external_wwpgd
    outdir = Path(output or "."); outdir.mkdir(parents=True, exist_ok=True)
    report = {"requested_device": requested or "auto", "valid_for_science": False, "timings": {}, "warnings": [], "failures": []}
    t0=time.perf_counter()
    dev=detect_device(requested); report.update(device_summary(requested)); report["resolved_device"]=str(dev)
    cfg=ModelConfig(n_layer=1,n_head=1,n_embd=16,block_size=8,vocab_size=32)
    model=GPT(cfg).to(dev); opt=torch.optim.AdamW(model.parameters(), lr=1e-3)
    x=torch.randint(0, cfg.vocab_size, (1, cfg.block_size), device=dev); y=torch.randint(0, cfg.vocab_size, (1, cfg.block_size), device=dev)
    logits, loss=model(x,y); report["forward_success"]=bool(loss is not None)
    loss.backward(); report["backward_success"]=True
    opt.step(); opt.zero_grad(); synchronize_device(dev); report["optimizer_success"]=True
    details=weightwatcher_details(model); report["weightwatcher_success"]=True
    rows=apply_external_wwpgd(model, event_index=0, actual_step=1, actual_tokens_seen=8); report["wwpgd_success"]=True; report["wwpgd_rows"]=len(rows)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    reader_state={"pos": 8}; before_rng=rng_state()
    ck=save_checkpoint(outdir, {"model_state_dict":model.state_dict(),"optimizer_state_dict":{"optimizers":[opt.state_dict()]},"scheduler_state_dict":None,"gradient_scaler_state_dict":None,"current_step":1,"next_step":2,"tokens_processed":8,"training_reader_position":reader_state["pos"],"reader_position":reader_state["pos"],"seed":0,**before_rng,"device_type":getattr(dev,'type',str(dev)),"precision_policy":"torch_default","gradient_accumulation_position":0,"metrics_rows":[],"periodic_weightwatcher_rows":[],"wwpgd_projection_rows":rows,"immediate_projection_weightwatcher_rows":[],"scientific_schema_version":0,"compatibility":{}})
    loaded=load_latest_checkpoint(outdir); model.load_state_dict(loaded["model_state_dict"]); opt.load_state_dict(loaded["optimizer_state_dict"]["optimizers"][0]); restore_rng_state(loaded); report["checkpoint_success"]=True
    logits2, loss2=model(x,y); loss2.backward(); opt.step(); synchronize_device(dev); report["resume_success"]=True
    report["timings"]["total_seconds"]=time.perf_counter()-t0
    (outdir/"device_preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str)+"\n")
    return report
