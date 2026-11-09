"""Contract 12.1 — duplicate replay invariance."""

from __future__ import annotations

from memcheck.contracts.base import Contract, ContractContext, make_event
from memcheck.models import ContractViolation, Severity


class DuplicateReplayInvariance(Contract):
    name = "duplicate_replay_invariance"
    description = (
        "Delivering the same event more than once must not change the store's answer or "
        "amplify the number of active memories."
    )
    severity = Severity.high

    @property
    def preconditions(self) -> list[str]:
        return [
            "the backend accepts events with stable event ids",
            "the backend exposes retrieval for a tenant",
        ]

    @property
    def expected_invariant(self) -> str:
        return (
            "For an event stream S and the stream S' obtained by delivering every event of S "
            "twice, the retrieved evidence and the active-record multiset are identical."
        )

    def execute(self, context: ContractContext) -> list[ContractViolation]:
        events = [
            make_event(1, "The primary database is postgres."),
            make_event(2, "The cache layer is redis."),
            make_event(3, "The message broker is rabbitmq."),
        ]
        query = "What is the primary database?"

        # Test generator: the same stream, once and then twice.
        once = context.fresh_backend("once")
        twice = context.fresh_backend("twice")
        try:
            for event in events:
                once.ingest_event(event)
            context.steps.append(f"ingest {len(events)} events once into {context.backend_name}")

            for event in events:
                twice.ingest_event(event)
            for event in events:
                twice.ingest_event(event)
            context.steps.append("re-deliver the identical events a second time")

            once.checkpoint()
            twice.checkpoint()

            baseline = once.retrieve(
                query,
                "tenant_a",
                evidence_budget=context.evidence_budget,
                token_budget=context.token_budget,
            )
            replayed = twice.retrieve(
                query,
                "tenant_a",
                evidence_budget=context.evidence_budget,
                token_budget=context.token_budget,
            )
            context.steps.append(f"query both stores with {query!r}")

            violations: list[ContractViolation] = []

            # Violation detector 1: the answer changed.
            baseline_contents = [item.content for item in baseline.evidence]
            replayed_contents = [item.content for item in replayed.evidence]
            if baseline_contents != replayed_contents or baseline.abstained != replayed.abstained:
                violations.append(
                    self.violation(
                        context,
                        expected=(
                            "bounded retrieval returns the same evidence after duplicate delivery"
                        ),
                        observed=(
                            f"evidence changed from {baseline_contents} (abstained="
                            f"{baseline.abstained}) to {replayed_contents} (abstained="
                            f"{replayed.abstained})"
                        ),
                        event_ids=[event.event_id for event in events],
                        query_id=query,
                        metadata={
                            "baseline_evidence": baseline_contents,
                            "replayed_evidence": replayed_contents,
                        },
                    )
                )

            # Violation detector 2: duplicate amplification of stored state.
            baseline_records = len(once.active_records()) or len(baseline.evidence)
            replayed_records = len(twice.active_records()) or len(replayed.evidence)
            if replayed_records > baseline_records:
                violations.append(
                    self.violation(
                        context,
                        expected="duplicate delivery creates no additional active memories",
                        observed=(
                            f"active memory count grew from {baseline_records} to "
                            f"{replayed_records} after re-delivering the same events"
                        ),
                        event_ids=[event.event_id for event in events],
                        metadata={
                            "amplification_factor": replayed_records / max(1, baseline_records)
                        },
                    )
                )
            return violations
        finally:
            once.close()
            twice.close()
