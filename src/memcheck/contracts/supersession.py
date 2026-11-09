"""Contract 12.3 — supersession correctness."""

from __future__ import annotations

from datetime import timedelta

from memcheck.contracts.base import EPOCH, Contract, ContractContext, make_event
from memcheck.models import ContractViolation, Severity
from memcheck.providers.mock import normalize_value


class SupersessionCorrectness(Contract):
    name = "supersession_correctness"
    description = (
        "A newer valid fact must replace an earlier one for current queries, while the earlier "
        "fact stays reachable for historical (as-of) queries."
    )
    severity = Severity.high

    @property
    def preconditions(self) -> list[str]:
        return [
            "the backend accepts two events asserting different values for the same fact",
            "the second event carries a strictly later event_time",
        ]

    @property
    def expected_invariant(self) -> str:
        return (
            "A current query returns only the newest value; an as-of query positioned inside the "
            "earlier fact's validity interval returns only the earlier value."
        )

    def execute(self, context: ContractContext) -> list[ContractViolation]:
        backend = context.fresh_backend("supersede")
        try:
            old = make_event(1, "The deployment target is docker-swarm.")
            distractor = make_event(2, "The deployment schedule is weekly.")
            new = make_event(3, "The deployment target is kubernetes.")
            for event in (old, distractor, new):
                backend.ingest_event(event)
            backend.checkpoint()
            context.steps.append(
                "ingest an initial fact, an unrelated distractor, then an updated fact"
            )

            query = "What is the deployment target?"
            current = backend.retrieve(
                query,
                "tenant_a",
                evidence_budget=context.evidence_budget,
                token_budget=context.token_budget,
            )
            context.steps.append(f"current query: {query!r}")

            violations: list[ContractViolation] = []
            current_values = {normalize_value(item.content) for item in current.evidence}
            stale = normalize_value(old.content)
            fresh = normalize_value(new.content)

            # Detector 1: the superseded value must not be offered as current evidence.
            if stale in current_values and not any(
                item.valid_to is not None
                for item in current.evidence
                if normalize_value(item.content) == stale
            ):
                violations.append(
                    self.violation(
                        context,
                        expected="a current query returns only the newest value",
                        observed=(
                            "the superseded value 'docker-swarm' was returned as current "
                            "evidence with an open validity interval"
                        ),
                        event_ids=[old.event_id, new.event_id],
                        query_id=query,
                        metadata={"current_evidence": sorted(current_values)},
                    )
                )

            # Detector 2: the update must actually be visible.
            if fresh not in current_values and not current.abstained:
                violations.append(
                    self.violation(
                        context,
                        expected="the newest value is retrievable after the update",
                        observed=f"current evidence {sorted(current_values)} omits the update",
                        event_ids=[new.event_id],
                        query_id=query,
                    )
                )

            # Detector 3: history must survive. Probe inside the earlier interval.
            as_of = EPOCH + timedelta(days=2)
            historical = backend.retrieve(
                query,
                "tenant_a",
                as_of_time=as_of,
                evidence_budget=context.evidence_budget,
                token_budget=context.token_budget,
            )
            context.steps.append(f"historical query at as_of_time={as_of.isoformat()}")
            historical_values = {normalize_value(item.content) for item in historical.evidence}
            if stale not in historical_values:
                violations.append(
                    self.violation(
                        context,
                        expected=(
                            "an as-of query inside the earlier fact's validity interval returns "
                            "the earlier value"
                        ),
                        observed=(
                            f"historical evidence {sorted(historical_values)} does not contain "
                            "the superseded value"
                        ),
                        event_ids=[old.event_id],
                        query_id=f"{query}@{as_of.isoformat()}",
                        severity=Severity.medium,
                    )
                )
            if fresh in historical_values:
                violations.append(
                    self.violation(
                        context,
                        expected="an as-of query does not return facts from its future",
                        observed="the later value leaked into a historical query",
                        event_ids=[new.event_id],
                        query_id=f"{query}@{as_of.isoformat()}",
                    )
                )
            return violations
        finally:
            backend.close()
