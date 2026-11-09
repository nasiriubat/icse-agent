"""Contract 12.5 — tenant isolation. Any violation is critical."""

from __future__ import annotations

from memcheck.contracts.base import Contract, ContractContext, make_event
from memcheck.models import ContractViolation, Severity

TENANT_A_SECRET = "acme-internal-roadmap"
TENANT_B_SECRET = "globex-internal-roadmap"


class TenantIsolation(Contract):
    name = "tenant_isolation"
    description = "One tenant must never retrieve evidence belonging to another tenant."
    severity = Severity.critical

    @property
    def preconditions(self) -> list[str]:
        return [
            "at least two tenants have written similar-looking facts",
            "the query text matches content in both tenants",
        ]

    @property
    def expected_invariant(self) -> str:
        return (
            "For every query issued as tenant T, every returned evidence item originates from an "
            "event whose tenant_id is T."
        )

    def execute(self, context: ContractContext) -> list[ContractViolation]:
        backend = context.fresh_backend("tenants")
        try:
            a_event = make_event(
                1, f"The roadmap codename is {TENANT_A_SECRET}.", tenant_id="tenant_a"
            )
            b_event = make_event(
                2,
                f"The roadmap codename is {TENANT_B_SECRET}.",
                tenant_id="tenant_b",
                session_id="session_02",
            )
            backend.ingest_event(a_event)
            backend.ingest_event(b_event)
            backend.checkpoint()
            context.steps.append("two tenants write the same fact shape with different values")

            query = "What is the roadmap codename?"
            violations: list[ContractViolation] = []
            for tenant, own_secret, foreign_secret, foreign_event in (
                ("tenant_a", TENANT_A_SECRET, TENANT_B_SECRET, b_event),
                ("tenant_b", TENANT_B_SECRET, TENANT_A_SECRET, a_event),
            ):
                result = backend.retrieve(
                    query,
                    tenant,
                    evidence_budget=context.evidence_budget,
                    token_budget=context.token_budget,
                )
                context.steps.append(f"query as {tenant}")
                leaked = [item for item in result.evidence if foreign_secret in item.content]
                if leaked:
                    violations.append(
                        self.violation(
                            context,
                            expected=f"{tenant} sees only its own evidence",
                            observed=(
                                f"{tenant} retrieved {len(leaked)} evidence item(s) belonging to "
                                "the other tenant"
                            ),
                            event_ids=[foreign_event.event_id],
                            memory_ids=[item.memory_id for item in leaked],
                            query_id=query,
                            severity=Severity.critical,
                            metadata={"leaked_contents": [item.content for item in leaked]},
                        )
                    )
                if own_secret not in " ".join(item.content for item in result.evidence) and not (
                    result.abstained
                ):
                    context.steps.append(
                        f"note: {tenant} did not retrieve its own fact either (recall issue, "
                        "not an isolation failure)"
                    )
            return violations
        finally:
            backend.close()
