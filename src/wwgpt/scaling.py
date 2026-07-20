from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


PARAMETER_COUNT_CONVENTIONS = {
    "total": "Total unique trainable parameters; tied embedding/head storage is counted once.",
    "trainable": "Alias for total unique trainable parameters.",
    "total_unique_trainable": "Alias for total unique trainable parameters.",
    "non_position": "Unique trainable parameters excluding position embeddings.",
    "non_embedding": "Unique trainable parameters excluding token and position embeddings; for tied heads, the head is not subtracted a second time.",
    "transformer_body": "Transformer blocks plus final layer norm only; excludes token embeddings, position embeddings, and output head.",
}


@dataclass(frozen=True)
class ScalingPlan:
    requested_tokens: int
    realized_tokens: int
    steps: int
    tokens_per_step: int
    estimated_flops: float
    coverage_ratio: float
    scaling_valid: bool
    reason: str
    selected_parameter_count: int
    tokens_per_selected_parameter: float
    sequence_count: int
    optimizer_step_count: int


def selected_parameter_count(report, convention: str) -> int:
    if convention not in PARAMETER_COUNT_CONVENTIONS:
        raise ValueError(f"unknown parameter_count_convention {convention}; choose one of {sorted(PARAMETER_COUNT_CONVENTIONS)}")
    attr = f"{convention}_parameters"
    if convention in {"trainable", "total_unique_trainable"}:
        attr = "total_unique_trainable_parameters"
    value = getattr(report, attr)
    return int(value)


def plan_budget(param_count: int, token_multiplier: int, batch_size: int, block_size: int, grad_accum: int, available_tokens: int) -> ScalingPlan:
    requested = int(param_count * token_multiplier)
    tokens_per_step = batch_size * block_size * grad_accum
    import math
    steps = max(1, math.ceil(requested / tokens_per_step))
    realized = steps * tokens_per_step
    valid = steps > 0 and available_tokens >= realized
    reason = "ok" if valid else "insufficient corpus or zero complete batches"
    seqs = realized // block_size
    return ScalingPlan(requested, realized, steps, tokens_per_step, 6.0 * param_count * realized, realized / max(available_tokens, 1), valid, reason, int(param_count), realized / max(param_count, 1), seqs, steps)


def design_condition_number(ns: list[float], ds: list[float]) -> float:
    x = np.column_stack([np.ones(len(ns)), np.log(ns), np.log(ds)])
    return float(np.linalg.cond(x))


def is_non_collinear(ns: list[float], ds: list[float], threshold: float = 1000.0) -> bool:
    return len(set(ds)) > 1 and len(set(ns)) > 1 and design_condition_number(ns, ds) < threshold


def plan_dict(plan: ScalingPlan) -> dict[str, float | int | bool | str]:
    return asdict(plan)
