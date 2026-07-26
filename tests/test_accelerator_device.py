import types

import pytest
import torch

from wwgpt import device as devmod


def test_explicit_unavailable_cuda_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requested CUDA.*refusing fallback"):
        devmod.detect_device("cuda")


def test_auto_logs_cpu_reason_when_no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(devmod, "_mps_available", lambda: False)
    monkeypatch.setattr(devmod, "_xla_device", lambda: None)
    device, reason = devmod.detect_device("auto", explain=True)
    assert str(device) == "cpu"
    assert "auto selected cpu" in reason
    summary = devmod.device_summary("auto")
    assert summary["single_device_only"] is True
    assert summary["distributed_training"] is False


def test_auto_prefers_xla_then_cuda_then_mps(monkeypatch):
    fake_xla = torch.device("xla")
    monkeypatch.setattr(devmod, "_xla_device", lambda: fake_xla)
    assert devmod.detect_device("auto") == fake_xla
    monkeypatch.setattr(devmod, "_xla_device", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert devmod.detect_device("auto").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(devmod, "_mps_available", lambda: True)
    assert devmod.detect_device("auto").type == "mps"


def test_cuda_memory_metrics_only_called_for_cuda(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("cuda memory API should not be called")

    monkeypatch.setattr(torch.cuda, "memory_allocated", boom)
    monkeypatch.setattr(torch.cuda, "memory_reserved", boom)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", boom)
    assert devmod.memory_stats(torch.device("cpu")) == {}


def test_xla_optimizer_step_and_sync_use_xla_apis(monkeypatch):
    calls = []
    xm = types.SimpleNamespace(
        optimizer_step=lambda opt: calls.append(("step", opt)),
        mark_step=lambda: calls.append(("mark", None)),
    )
    monkeypatch.setattr(devmod, "_xla_model_module", lambda: xm)
    opt = object()
    devmod.optimizer_step(opt, torch.device("xla"))
    devmod.synchronize_device(torch.device("xla"))
    assert calls == [("step", opt), ("mark", None)]


def test_precision_policy_is_device_appropriate(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert devmod.precision_policy(torch.device("cuda"))["mixed_precision"] == "bf16"
    assert devmod.precision_policy(torch.device("cpu"))["mixed_precision"] == "none"
    assert devmod.precision_policy(torch.device("mps"))["mixed_precision"] == "none"


# Actual CUDA, MPS, and XLA tensor execution is environment-dependent and is
# exercised by `wwgpt device-preflight --device <device>` on the target machine.
# Keeping those probes inside the portable CPU pytest suite only produced
# guaranteed skips and did not add CI coverage beyond the contract tests above.
