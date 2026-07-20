from __future__ import annotations
import contextlib
import json, platform, sys
from pathlib import Path
import torch

VALID_DEVICES={"auto","cuda","mps","xla","cpu"}


def _device_type(dev) -> str:
    return getattr(dev, "type", str(dev).split(":")[0])


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _xla_model_module():
    try:
        import torch_xla.core.xla_model as xm
        return xm
    except Exception:
        return None


def _xla_device():
    xm = _xla_model_module()
    if xm is None:
        return None
    try:
        dev=xm.xla_device()
        t=torch.tensor([1.0], device=dev); _=(t+1).cpu()
        return dev
    except Exception:
        return None


def detect_device(requested: str|None="auto", *, explain: bool = False):
    req=(requested or "auto").lower()
    if req not in VALID_DEVICES: raise ValueError(f"unsupported device {req}; expected one of {sorted(VALID_DEVICES)}")
    decisions=[]
    if req=="cpu":
        decisions.append("explicit cpu requested; using CPU")
        dev=torch.device("cpu")
    elif req=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("requested CUDA but torch.cuda.is_available() is false; refusing fallback")
        decisions.append("explicit cuda requested and torch.cuda.is_available() is true")
        dev=torch.device("cuda")
    elif req=="mps":
        if not _mps_available(): raise RuntimeError("requested MPS but torch.backends.mps.is_available() is false; refusing fallback")
        decisions.append("explicit mps requested and torch.backends.mps.is_available() is true")
        dev=torch.device("mps")
    elif req=="xla":
        dev=_xla_device()
        if dev is None: raise RuntimeError("requested XLA/TPU but torch_xla is unavailable or no XLA device could be initialized; refusing fallback")
        decisions.append("explicit xla requested and torch_xla initialized a single XLA device")
    else:
        dev=_xla_device()
        if dev is not None:
            decisions.append("auto selected xla because torch_xla initialized a single XLA device")
        elif torch.cuda.is_available():
            dev=torch.device("cuda"); decisions.append("auto selected cuda because torch.cuda.is_available() is true")
        elif _mps_available():
            dev=torch.device("mps"); decisions.append("auto selected mps because CUDA/XLA were unavailable and MPS is available")
        else:
            dev=torch.device("cpu"); decisions.append("auto selected cpu because XLA, CUDA, and MPS were unavailable")
    if explain:
        return dev, "; ".join(decisions)
    return dev


def resolve_device(requested: str|None="auto"):
    return detect_device(requested)


def precision_policy(device=None, requested_precision: str | None = None):
    dev=device or detect_device("auto"); typ=_device_type(dev)
    requested=(requested_precision or "auto").lower()
    if requested not in {"auto", "fp32", "float32", "bf16", "bfloat16", "fp16", "float16"}:
        raise ValueError(f"unsupported precision {requested_precision}; expected auto, fp32, bf16, or fp16")
    if requested in {"fp32", "float32"} or typ in {"cpu", "mps"}:
        return {"dtype":"float32", "torch_dtype": torch.float32, "mixed_precision":"none", "autocast_enabled": False}
    if requested == "auto" and typ == "cuda":
        if torch.cuda.is_bf16_supported():
            return {"dtype":"bfloat16", "torch_dtype": torch.bfloat16, "mixed_precision":"bf16", "autocast_enabled": True}
        return {"dtype":"float32", "torch_dtype": torch.float32, "mixed_precision":"none", "autocast_enabled": False}
    if requested == "auto" and typ == "xla":
        return {"dtype":"bfloat16", "torch_dtype": torch.bfloat16, "mixed_precision":"bf16", "autocast_enabled": True}
    if requested in {"bf16", "bfloat16"}:
        if typ == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("requested bf16 precision on CUDA but torch.cuda.is_bf16_supported() is false")
        if typ not in {"cuda", "xla"}: raise RuntimeError("bf16 autocast is only supported for CUDA or XLA in this single-device trainer")
        return {"dtype":"bfloat16", "torch_dtype": torch.bfloat16, "mixed_precision":"bf16", "autocast_enabled": True}
    if requested in {"fp16", "float16"}:
        if typ not in {"cuda"}: raise RuntimeError("fp16 autocast is only supported for CUDA in this single-device trainer")
        return {"dtype":"float16", "torch_dtype": torch.float16, "mixed_precision":"fp16", "autocast_enabled": True}
    return {"dtype":"float32", "torch_dtype": torch.float32, "mixed_precision":"none", "autocast_enabled": False}


def autocast_context(device=None, requested_precision: str | None = None):
    dev=device or detect_device("auto"); typ=_device_type(dev); pol=precision_policy(dev, requested_precision)
    if not pol["autocast_enabled"]:
        return contextlib.nullcontext()
    return torch.autocast(device_type=typ, dtype=pol["torch_dtype"])


def optimizer_step(optimizer, device=None) -> None:
    dev=device or torch.device("cpu"); typ=_device_type(dev)
    if typ == "xla":
        xm = _xla_model_module()
        if xm is None: raise RuntimeError("XLA optimizer step requested but torch_xla is unavailable")
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()


def device_summary(requested: str|None="auto"):
    dev, reason=detect_device(requested, explain=True); typ=_device_type(dev); name=typ; count=1
    if typ=="cuda": name=torch.cuda.get_device_name(0); count=torch.cuda.device_count()
    elif typ=="mps": name="Apple Silicon MPS"
    elif typ=="xla": name=str(dev)
    pol=precision_policy(dev)
    try:
        import torch_xla; xla_ver=getattr(torch_xla,"__version__","unknown")
    except Exception: xla_ver=None
    return {"requested_device":requested or "auto","resolved_device":str(dev),"device_type":typ,"device_name":name,"accelerator_count":count,"single_device_only": True,"distributed_training": False,"multi_accelerator_claim":"unsupported; no distributed smoke path is implemented","selection_reason":reason,"pytorch_version":torch.__version__,"cuda_version":torch.version.cuda,"mps_available":_mps_available(),"torch_xla_version":xla_ver,"precision_selected":pol["dtype"],"mixed_precision_policy":pol["mixed_precision"],"memory_information":memory_stats(dev),"operating_system":platform.platform(),"python_version":sys.version}


def synchronize_device(device=None):
    dev=device or detect_device("auto"); typ=_device_type(dev)
    if typ=="cuda": torch.cuda.synchronize(dev)
    elif typ=="mps" and hasattr(torch,"mps"): torch.mps.synchronize()
    elif typ=="xla":
        xm=_xla_model_module()
        if xm is None: raise RuntimeError("XLA synchronization requested but torch_xla is unavailable")
        xm.mark_step()


def memory_stats(device=None):
    dev=device or torch.device("cpu"); typ=_device_type(dev)
    if typ=="cuda": return {"allocated":torch.cuda.memory_allocated(dev),"reserved":torch.cuda.memory_reserved(dev),"max_allocated":torch.cuda.max_memory_allocated(dev)}
    return {}


def save_device_manifest(path: Path, requested: str|None="auto"):
    path=Path(path); path.write_text(json.dumps(device_summary(requested), indent=2, sort_keys=True)+"\n")
    return path

def run_device_preflight(output: Path | None = None, requested: str | None = "auto"):
    import time
    from dataclasses import asdict
    from wwgpt.model import GPT
    from wwgpt.config import ModelConfig
    from wwgpt.checkpointing import save_checkpoint, load_latest_checkpoint, rng_state, restore_rng_state, complete_test_checkpoint_state
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
    optimizer_step(opt, dev); opt.zero_grad(); synchronize_device(dev); report["optimizer_success"]=True
    details=weightwatcher_details(model); report["weightwatcher_success"]=True
    rows=apply_external_wwpgd(model, event_index=0, actual_step=1, actual_tokens_seen=8); report["wwpgd_success"]=True; report["wwpgd_rows"]=len(rows)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    reader_state={"pos": 8}; before_rng=rng_state()
    ck=save_checkpoint(outdir, complete_test_checkpoint_state(model_state_dict=model.state_dict(), optimizer_state_dict={"optimizers":[opt.state_dict()]}, base_optimizer_state_dict=opt.state_dict(), current_step=1, next_step=2, tokens_processed=8, training_reader_position=reader_state["pos"], reader_position=reader_state["pos"], seed=0, **before_rng, device_type=getattr(dev,'type',str(dev)), wwpgd_projection_rows=rows, resolved_config={"device_preflight": True}, optimizer_fingerprint="device-preflight", data_hash="device-preflight", tokenizer_hash="device-preflight"))
    loaded=load_latest_checkpoint(outdir); model.load_state_dict(loaded["model_state_dict"]); opt.load_state_dict(loaded["optimizer_state_dict"]["optimizers"][0]); restore_rng_state(loaded); report["checkpoint_success"]=True
    logits2, loss2=model(x,y); loss2.backward(); optimizer_step(opt, dev); synchronize_device(dev); report["resume_success"]=True
    report["timings"]["total_seconds"]=time.perf_counter()-t0
    (outdir/"device_preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str)+"\n")
    return report
