"""Workloads and the factory that builds them from a configuration."""

from __future__ import annotations

from memcheck.config import ExperimentConfig
from memcheck.workloads.base import (
    QUERY_CATEGORIES,
    Checkpoint,
    Query,
    Workload,
    grade_answer,
    query_is_asked,
)
from memcheck.workloads.longmemeval import LongMemEvalUnavailable, LongMemEvalWorkload
from memcheck.workloads.repo_evolution import SCENARIOS, CodingTask, RepoEvolutionWorkload
from memcheck.workloads.synthetic import SyntheticLongitudinalWorkload

__all__ = [
    "QUERY_CATEGORIES",
    "SCENARIOS",
    "Checkpoint",
    "CodingTask",
    "LongMemEvalUnavailable",
    "LongMemEvalWorkload",
    "Query",
    "RepoEvolutionWorkload",
    "SyntheticLongitudinalWorkload",
    "Workload",
    "build_workload",
    "grade_answer",
    "query_is_asked",
]


def build_workload(config: ExperimentConfig, seed: int | None = None) -> Workload:
    """Construct the workload named by the configuration, passing its declared parameters.

    ``seed`` selects the stream for seed-sensitive workloads; it defaults to the first
    configured seed. The runner builds one workload per seed, so seeds are separate samples.
    """
    spec = config.workload
    if seed is None:
        seed = config.seeds[0] if config.seeds else 1
    if spec.name == "synthetic_longitudinal":
        params = dict(spec.params)
        # v3 derives one query set per checkpoint, so it must know the schedule.
        params.setdefault("checkpoint_fractions", list(config.checkpoints))
        return SyntheticLongitudinalWorkload(version=spec.version, seed=seed, **params)
    if spec.name == "repo_evolution":
        return RepoEvolutionWorkload(
            version=spec.version, seed=seed, subset=spec.subset, **spec.params
        )
    if spec.name == "longmemeval":
        return LongMemEvalWorkload(
            version=spec.version,
            subset=spec.subset or "smoke",
            data_path=spec.data_path,
            **spec.params,
        )
    raise KeyError(f"unknown workload {spec.name!r}")
