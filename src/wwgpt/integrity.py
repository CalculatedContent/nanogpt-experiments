"""Compatibility import for the schema-v3 integrity implementation."""

from wwgpt.integrity_v2 import (
    BASELINE_EXTENSIONS,
    CANONICAL_ARMS,
    CANONICAL_PAIRS,
    audit_arm,
    audit_experiment,
    audit_run,
    audit_trial,
    normalize_arm,
)

__all__ = [
    "BASELINE_EXTENSIONS",
    "CANONICAL_ARMS",
    "CANONICAL_PAIRS",
    "audit_arm",
    "audit_experiment",
    "audit_run",
    "audit_trial",
    "normalize_arm",
]
