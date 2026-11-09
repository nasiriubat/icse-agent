"""Contract 12.2 — restart equivalence.

The equivalence rule is defined once, here, and reused by the fault harness so that the reported
restart-divergence metric has exactly one meaning (DECISIONS D-007).

Two backend states are **equivalent** when both hold:

1. The multiset of ``(tenant_id, canonical_key, version, content, status, valid_from, valid_to)``
   over all non-deleted records is equal.
2. For every probe query, the ordered list of ``(memory_id, version)`` in the bounded retrieval
   result is equal, and the abstention decision is equal.

Deliberately excluded: wall-clock latency, physical row order, database file size, row ids, and
any timestamp the store assigns at write time rather than deriving from the event.
"""

from __future__ import annotations

from memcheck.backends.base import MemoryBackend
from memcheck.contracts.base import Contract, ContractContext, make_event
from memcheck.models import BackendState, ContractViolation, Severity

PROBE_QUERIES: list[tuple[str, str]] = [
    ("What is the primary database?", "tenant_a"),
    ("What is the deployment target?", "tenant_a"),
    ("What is the retention period?", "tenant_a"),
]


def canonical_state(
    backend: MemoryBackend, *, evidence_budget: int, token_budget: int
) -> BackendState:
    """The canonical projection both sides of a restart are compared on."""
    return backend.canonical_state(
        PROBE_QUERIES, evidence_budget=evidence_budget, token_budget=token_budget
    )


def divergence(before: BackendState, after: BackendState) -> list[str]:
    """Human-readable differences between two canonical states; empty means equivalent."""
    differences: list[str] = []
    before_records = sorted(str(record) for record in before.records)
    after_records = sorted(str(record) for record in after.records)
    if before_records != after_records:
        missing = [record for record in before_records if record not in after_records]
        added = [record for record in after_records if record not in before_records]
        if missing:
            differences.append(f"records lost after restart: {missing}")
        if added:
            differences.append(f"records appeared after restart: {added}")
    for probe, expected in before.probe_results.items():
        observed = after.probe_results.get(probe)
        if observed != expected:
            differences.append(f"probe {probe!r}: evidence changed from {expected} to {observed}")
    for probe, expected_abstention in before.probe_abstentions.items():
        observed_abstention = after.probe_abstentions.get(probe)
        if observed_abstention != expected_abstention:
            differences.append(
                f"probe {probe!r}: abstention changed from {expected_abstention} to "
                f"{observed_abstention}"
            )
    return differences


class RestartEquivalence(Contract):
    name = "restart_equivalence"
    description = (
        "A clean restart must not materially change active memory state or bounded retrieval "
        "results, under the canonical-projection equivalence rule."
    )
    severity = Severity.high

    @property
    def preconditions(self) -> list[str]:
        return [
            "the backend persists state durably",
            "the backend implements restart()",
        ]

    @property
    def expected_invariant(self) -> str:
        return (
            "canonical_state(before restart) == canonical_state(after restart), where the "
            "canonical state is the non-deleted record multiset plus the ordered probe results "
            "and abstention decisions."
        )

    def execute(self, context: ContractContext) -> list[ContractViolation]:
        backend = context.fresh_backend("restart")
        try:
            events = [
                make_event(1, "The primary database is postgres."),
                make_event(2, "The deployment target is docker-swarm."),
                make_event(3, "The retention period is thirty days."),
                make_event(4, "The deployment target is kubernetes."),
            ]
            for event in events:
                backend.ingest_event(event)
            backend.checkpoint()
            context.steps.append(f"ingest {len(events)} events, including one fact update")

            before = canonical_state(
                backend,
                evidence_budget=context.evidence_budget,
                token_budget=context.token_budget,
            )
            context.steps.append("snapshot the canonical state")

            backend.restart()
            context.steps.append("restart the backend")

            after = canonical_state(
                backend,
                evidence_budget=context.evidence_budget,
                token_budget=context.token_budget,
            )
            differences = divergence(before, after)
            if not differences:
                return []
            return [
                self.violation(
                    context,
                    expected=self.expected_invariant,
                    observed="; ".join(differences),
                    event_ids=[event.event_id for event in events],
                    metadata={
                        "fingerprint_before": before.fingerprint(),
                        "fingerprint_after": after.fingerprint(),
                        "difference_count": len(differences),
                    },
                )
            ]
        finally:
            backend.close()
