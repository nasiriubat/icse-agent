"""Fairness validator (Section 15).

Its job is to make an unfair comparison *impossible to run and impossible to report*, rather
than merely discouraged. Two checks:

1. **Within a run** — every condition must share one fairness key. Only a declared ablation may
   differ, and only in the one guard it names.
2. **Across runs** — before the reporting layer pools two run directories into one table, their
   manifests must agree on model, temperature, prompts, tools, context limit, query set, token
   budget, checkpoint schedule, and grading method.

The second check is the one that matters in practice: pooling yesterday's run with today's after
a config edit is the easy mistake, and it is silent without this.
"""

from __future__ import annotations

from dataclasses import dataclass

from memcheck.config import ExperimentConfig
from memcheck.evaluation.matrix import Condition, fairness_fields
from memcheck.models import ExperimentRunManifest

#: Manifest fields that must match for two runs to be comparable.
COMPARABLE_FIELDS: tuple[str, ...] = (
    "model_provider",
    "model_identifier",
    "temperature",
    "benchmark_name",
    "benchmark_version",
    "benchmark_parameters_hash",
    "token_budget",
    "evidence_budget",
    "checkpoint_schedule",
    "retrieval_mode",
    "extractor_identity",
    "prompt_hashes",
    "attempt",
    "contract_store",
)


#: The subset of comparability fields that identify *which experiment* a run belongs to, rather
#: than how it was parameterised within one. The reporting layer groups on these, so two runs
#: differing in any of them are never pooled in the first place; the remaining comparable fields
#: are checked *within* a group and reported as problems if they differ.
#:
#: Adding a field to COMPARABLE_FIELDS forces a decision about this list. Getting it wrong is not
#: a cosmetic error: when the extractor became a comparability field and this list was not
#: updated, two runs of the same benchmark under the same reader with *different write policies*
#: were pooled into one table.
GROUPING_FIELDS: tuple[str, ...] = (
    "benchmark_name",
    "benchmark_version",
    "benchmark_parameters_hash",
    "model_provider",
    "model_identifier",
    "extractor_identity",
    # The condition set is part of the experiment's identity too. Two runs of the same workload
    # exploring different ablations or faults share their unablated conditions, and on a
    # deterministic workload those shared conditions are the *same computation performed twice*,
    # not two independent samples. Pooling them would double-count every one of them.
    "ablations",
    "faults",
    # The prompts, judge included. A re-graded copy of a run (D-025) shares every other field
    # with its source, and pooling the two would count each answer twice under two judges.
    "prompt_hashes",
    # Independent repeats of one configuration answer the same queries again; pooling them would
    # count each query once per attempt as if it were a new query.
    "attempt",
    # A run whose contracts were executed on aged snapshots answers the same queries as the
    # fresh-store run of the same configuration; it is a different contract experiment.
    "contract_store",
)


#: Fields that must be *present*, not merely equal. An empty value means the run predates the
#: field, so comparability cannot be established either way.
REQUIRED_FIELDS: tuple[str, ...] = ("benchmark_parameters_hash",)


class FairnessError(ValueError):
    """Raised when a comparison would be unfair and was not declared as an ablation."""


@dataclass
class FairnessReport:
    ok: bool
    problems: list[str]
    checked: int

    def raise_if_unfair(self) -> None:
        if not self.ok:
            raise FairnessError(
                "refusing to proceed with an unfair comparison:\n  - "
                + "\n  - ".join(self.problems)
            )


def validate_conditions(config: ExperimentConfig, conditions: list[Condition]) -> FairnessReport:
    """Every condition in one run must share a fairness key; ablations must be declared."""
    problems: list[str] = []
    keys = {condition.fairness_key for condition in conditions}
    if len(keys) > 1:
        problems.append(
            f"conditions span {len(keys)} distinct fairness keys; a single run must hold "
            "model, prompts, tools, budgets, query set, checkpoints, and grading constant"
        )
    for condition in conditions:
        if condition.ablations and not condition.declared_ablation:
            problems.append(
                f"condition {condition.label} disables {list(condition.ablations)} without "
                "being declared as an ablation in the configuration"
            )
        if condition.ablations and condition.backend != "guarded_memory":
            problems.append(
                f"condition {condition.label} applies an ablation to {condition.backend}, "
                "which has no such guard to disable"
            )
        if len(condition.ablations) > 1:
            problems.append(
                f"condition {condition.label} disables {len(condition.ablations)} guards at "
                "once; an ablation must isolate exactly one difference"
            )
    if not conditions:
        problems.append("the matrix expanded to zero conditions")
    _ = fairness_fields(config)  # documents the fields the key is built from
    return FairnessReport(ok=not problems, problems=problems, checked=len(conditions))


def validate_across_runs(manifests: list[ExperimentRunManifest]) -> FairnessReport:
    """Refuse to pool runs whose comparison-relevant settings differ."""
    problems: list[str] = []
    if len(manifests) < 2:
        return FairnessReport(ok=True, problems=[], checked=len(manifests))
    # A manifest written before a comparability field existed carries an empty value. Two such
    # runs are *unverifiable*, not proven comparable, and treating absence as agreement is how a
    # silent unfair pooling would slip through.
    for manifest in manifests:
        for field_name in REQUIRED_FIELDS:
            if not getattr(manifest, field_name):
                problems.append(
                    f"run {manifest.run_id} records no {field_name}; it predates that field and "
                    "cannot be shown comparable to another run. Re-run it to pool it."
                )

    reference = manifests[0]
    for manifest in manifests[1:]:
        for field_name in COMPARABLE_FIELDS:
            expected = getattr(reference, field_name)
            observed = getattr(manifest, field_name)
            if expected != observed:
                problems.append(
                    f"run {manifest.run_id} differs from {reference.run_id} on {field_name}: "
                    f"{observed!r} != {expected!r}"
                )
        if manifest.prompt_hashes != reference.prompt_hashes:
            differing = sorted(
                name
                for name in set(manifest.prompt_hashes) | set(reference.prompt_hashes)
                if manifest.prompt_hashes.get(name) != reference.prompt_hashes.get(name)
            )
            problems.append(f"run {manifest.run_id} used different prompt text for: {differing}")
    return FairnessReport(ok=not problems, problems=problems, checked=len(manifests))
