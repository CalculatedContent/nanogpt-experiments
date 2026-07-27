from pathlib import Path

path = Path("scripts/_agent_phase2_health_workflow.py")
text = path.read_text()

model_diag_old = (
    'def _model_diagnostic_metrics(model: torch.nn.Module) -> dict[str, float | int]:\\n'
    '    total_sq = 0.0\\n'
)
model_diag_new = (
    'def _model_diagnostic_metrics(model: torch.nn.Module) -> dict[str, float | int]:\\n'
    '    from wwgpt.ww import is_projected_layer\\n\\n'
    '    total_sq = 0.0\\n'
)
if text.count(model_diag_old) != 1:
    raise RuntimeError(
        f"expected one model-diagnostics helper block, found {text.count(model_diag_old)}"
    )
text = text.replace(model_diag_old, model_diag_new, 1)

completion_old = (
    '    })\\n'
    '    write_json(run_dir / "run_complete.json", common_complete)\\n'
    '    _log_train_progress(\\n'
)
completion_new = (
    '    })\\n'
    '    (run_dir / "run_complete.json").write_text(\\n'
    '        json.dumps(common_complete, indent=2, sort_keys=True, default=str) + "\\\\n"\\n'
    '    )\\n'
    '    _log_train_progress(\\n'
)
if text.count(completion_old) != 1:
    raise RuntimeError(
        f"expected one health completion block, found {text.count(completion_old)}"
    )
text = text.replace(completion_old, completion_new, 1)

path.write_text(text)
