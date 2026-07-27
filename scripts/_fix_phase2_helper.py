from pathlib import Path

path = Path("scripts/_agent_phase2_health_workflow.py")
text = path.read_text()

scheduler_old = '''replace_once(
    train_path,
    '        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,\\n',
    '        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,\\n'
    '        "learning_rate_resolution": dict(learning_rate_resolution or {}),\\n',
)
'''
scheduler_new = '''replace_once(
    train_path,
    '        "model_architecture_version": model.model_architecture_version,\\n'
    '        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,\\n',
    '        "model_architecture_version": model.model_architecture_version,\\n'
    '        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,\\n'
    '        "learning_rate_resolution": dict(learning_rate_resolution or {}),\\n',
)
'''
if text.count(scheduler_old) != 1:
    raise RuntimeError(
        f"expected one ambiguous scheduler helper block, found {text.count(scheduler_old)}"
    )
text = text.replace(scheduler_old, scheduler_new, 1)

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

path.write_text(text)
