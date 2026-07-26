from pathlib import Path

path = Path("scripts/_agent_phase2_health_workflow.py")
text = path.read_text()
old = '''replace_once(
    train_path,
    '        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,\\n',
    '        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,\\n'
    '        "learning_rate_resolution": dict(learning_rate_resolution or {}),\\n',
)
'''
new = '''replace_once(
    train_path,
    '        "model_architecture_version": model.model_architecture_version,\\n'
    '        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,\\n',
    '        "model_architecture_version": model.model_architecture_version,\\n'
    '        "scheduler_implementation": SCHEDULER_IMPLEMENTATION,\\n'
    '        "learning_rate_resolution": dict(learning_rate_resolution or {}),\\n',
)
'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one ambiguous scheduler helper block, found {text.count(old)}")
path.write_text(text.replace(old, new, 1))
