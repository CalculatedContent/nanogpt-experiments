from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


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


def plan_budget(param_count: int, token_multiplier: int, batch_size: int, block_size: int, grad_accum: int, available_tokens: int) -> ScalingPlan:
    requested = int(param_count * token_multiplier)
    tokens_per_step = batch_size * block_size * grad_accum
    steps = requested // tokens_per_step
    realized = steps * tokens_per_step
    valid = steps > 0 and available_tokens >= realized
    reason = "ok" if valid else "insufficient corpus or zero complete batches"
    return ScalingPlan(requested, realized, steps, tokens_per_step, 6.0 * param_count * realized, realized / max(available_tokens, 1), valid, reason)


def design_condition_number(ns: list[float], ds: list[float]) -> float:
    x = np.column_stack([np.ones(len(ns)), np.log(ns), np.log(ds)])
    return float(np.linalg.cond(x))


def is_non_collinear(ns: list[float], ds: list[float], threshold: float = 1000.0) -> bool:
    return len(set(ds)) > 1 and len(set(ns)) > 1 and design_condition_number(ns, ds) < threshold


def plan_dict(plan: ScalingPlan) -> dict[str, float | int | bool | str]:
    return asdict(plan)
