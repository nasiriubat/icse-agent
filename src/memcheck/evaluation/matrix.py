"""Experiment matrix expansion.

The matrix is the cross product of backends (plus declared ablations), faults, seeds, and
repetitions. Each resulting condition has a stable identifier derived from its content, so the
same configuration yields the same condition ids on any machine and a resumed run can tell
exactly which conditions are already finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memcheck.config import ExperimentConfig
from memcheck.models import content_hash
from memcheck.utilities.hashing import stable_id


@dataclass(frozen=True)
class Condition:
    """One cell of the experiment matrix."""

    condition_id: str
    backend: str
    ablations: tuple[str, ...]
    fault: str
    seed: int
    repetition: int
    fairness_key: str
    label: str
    declared_ablation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "backend": self.backend,
            "ablations": list(self.ablations),
            "fault": self.fault,
            "seed": self.seed,
            "repetition": self.repetition,
            "fairness_key": self.fairness_key,
            "label": self.label,
            "declared_ablation": self.declared_ablation,
        }


def fairness_fields(config: ExperimentConfig) -> dict[str, Any]:
    """Exactly the fields that must be identical across a legitimate comparison (Section 15)."""
    return {
        "main_model": config.model.qualified_name,
        "temperature": config.model.temperature,
        "system_prompt_id": config.model.system_prompt_id,
        "tools": sorted(config.model.tools_enabled),
        "context_limit_tokens": config.model.context_limit_tokens,
        "max_output_tokens": config.model.max_output_tokens,
        "query_set": f"{config.workload.name}@{config.workload.version}:{config.workload.subset}",
        "workload_params": config.workload.params,
        "token_budget": config.retrieval.token_budget,
        "evidence_budget": config.retrieval.evidence_budget,
        "retrieval_mode": config.retrieval.mode,
        "checkpoints": config.checkpoints,
        "grading_method": "deterministic",
    }


def fairness_key(config: ExperimentConfig) -> str:
    return content_hash(fairness_fields(config))


def expand(config: ExperimentConfig) -> list[Condition]:
    """Expand a configuration into its conditions, in a deterministic order."""
    key = fairness_key(config)
    conditions: list[Condition] = []

    # Backend variants: each core backend, plus one guarded variant per declared ablation.
    variants: list[tuple[str, tuple[str, ...], bool]] = [
        (backend, (), False) for backend in config.backends
    ]
    for ablation in config.ablations:
        variants.append(("guarded_memory", (ablation,), True))

    for backend, ablations, declared in variants:
        for fault in config.faults:
            for seed in config.seeds:
                for repetition in range(1, config.repetitions + 1):
                    label_parts = [backend]
                    if ablations:
                        label_parts.append("+".join(ablations))
                    if fault != "none":
                        label_parts.append(fault)
                    label_parts.append(f"seed{seed}")
                    if config.repetitions > 1:
                        label_parts.append(f"rep{repetition}")
                    condition_id = stable_id(
                        "cond",
                        key,
                        backend,
                        sorted(ablations),
                        fault,
                        seed,
                        repetition,
                    )
                    conditions.append(
                        Condition(
                            condition_id=condition_id,
                            backend=backend,
                            ablations=tuple(sorted(ablations)),
                            fault=fault,
                            seed=seed,
                            repetition=repetition,
                            fairness_key=key,
                            label="/".join(label_parts),
                            declared_ablation=declared,
                        )
                    )
    return conditions
