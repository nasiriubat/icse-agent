"""Experiment orchestration: matrix, fairness, cost control, and the runner."""

from memcheck.evaluation.cost import CostEstimate, CostLimitExceeded, enforce, estimate
from memcheck.evaluation.fairness import (
    FairnessError,
    FairnessReport,
    validate_across_runs,
    validate_conditions,
)
from memcheck.evaluation.matrix import Condition, expand, fairness_fields, fairness_key
from memcheck.evaluation.runner import (
    ExperimentRunner,
    discover_runs,
    load_manifest,
    verify_integrity,
)

__all__ = [
    "Condition",
    "CostEstimate",
    "CostLimitExceeded",
    "ExperimentRunner",
    "FairnessError",
    "FairnessReport",
    "discover_runs",
    "enforce",
    "estimate",
    "expand",
    "fairness_fields",
    "fairness_key",
    "load_manifest",
    "validate_across_runs",
    "validate_conditions",
    "verify_integrity",
]
