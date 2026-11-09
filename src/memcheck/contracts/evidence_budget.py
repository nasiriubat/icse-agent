"""Contract 12.6 — evidence budget compliance."""

from __future__ import annotations

from memcheck.contracts.base import Contract, ContractContext, PreconditionUnmet, make_event
from memcheck.models import ContractViolation, Severity
from memcheck.utilities.tokens import count_tokens


class EvidenceBudgetCompliance(Contract):
    name = "evidence_budget_compliance"
    description = (
        "The backend must never return more evidence items than the configured count budget, "
        "nor more injected tokens than the configured token budget."
    )
    severity = Severity.medium

    @property
    def preconditions(self) -> list[str]:
        return [
            "the store holds more matching content than the budget allows",
            "a count budget and a token budget are configured",
        ]

    @property
    def expected_invariant(self) -> str:
        return (
            "len(result.evidence) <= evidence_budget and "
            "sum(token_count) <= token_budget and result.injected_tokens is accurate."
        )

    def execute(self, context: ContractContext) -> list[ContractViolation]:
        backend = context.fresh_backend("budget")
        try:
            # Generate far more matching content than any budget would admit. Each sentence is
            # a distinct fact in the "the <attribute> is <value>" form, so an extracting backend
            # stores every one of them rather than discarding the sentence.
            events = [
                make_event(
                    index,
                    f"The service level objective for region {index} is tier-{index} availability.",
                    session_id=f"session_{index:02d}",
                )
                for index in range(1, 26)
            ]
            for event in events:
                backend.ingest_event(event)
            backend.checkpoint()
            context.steps.append(f"ingest {len(events)} events all matching the probe query")

            query = "What is the service level objective for region?"
            budgets = ((1, 512), (3, 64), (5, 16))
            # Precondition: with no effective budget the store must return more matching items
            # than the largest budget tested. Otherwise no budget could be exceeded and a pass
            # would mean nothing.
            unbounded = backend.retrieve(
                query, "tenant_a", evidence_budget=len(events) * 2, token_budget=1_000_000
            )
            context.steps.append(f"unbounded probe returned {len(unbounded.evidence)} items")
            largest = max(evidence_budget for evidence_budget, _ in budgets)
            if len(unbounded.evidence) <= largest:
                raise PreconditionUnmet(
                    f"the store returned {len(unbounded.evidence)} matching items, not more than "
                    f"the largest budget tested ({largest})"
                )
            for evidence_budget, token_budget in budgets:
                result = backend.retrieve(
                    query,
                    "tenant_a",
                    evidence_budget=evidence_budget,
                    token_budget=token_budget,
                )
                context.steps.append(
                    f"query with evidence_budget={evidence_budget}, token_budget={token_budget}"
                )
                violations: list[ContractViolation] = []
                if len(result.evidence) > evidence_budget:
                    violations.append(
                        self.violation(
                            context,
                            expected=f"at most {evidence_budget} evidence items",
                            observed=f"returned {len(result.evidence)} evidence items",
                            query_id=query,
                            metadata={
                                "evidence_budget": evidence_budget,
                                "returned": len(result.evidence),
                            },
                        )
                    )
                actual_tokens = sum(count_tokens(item.content) for item in result.evidence)
                if actual_tokens > token_budget:
                    violations.append(
                        self.violation(
                            context,
                            expected=f"at most {token_budget} injected tokens",
                            observed=f"injected {actual_tokens} tokens",
                            query_id=query,
                            metadata={
                                "token_budget": token_budget,
                                "injected_tokens": actual_tokens,
                            },
                        )
                    )
                if result.injected_tokens != actual_tokens:
                    violations.append(
                        self.violation(
                            context,
                            expected="reported injected_tokens matches the evidence actually sent",
                            observed=(
                                f"reported {result.injected_tokens} but the evidence totals "
                                f"{actual_tokens} tokens"
                            ),
                            query_id=query,
                            severity=Severity.low,
                        )
                    )
                if violations:
                    return violations
            return []
        finally:
            backend.close()
