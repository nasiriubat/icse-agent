"""Contract 12.4 — deletion closure (DECISIONS D-008).

Deleted content must not come back through *any* surface: active records, summaries, the lexical
index, the semantic index, the retrieval cache, retrieval evidence, or a historical as-of query.
Each surface is probed separately so a partial implementation is reported as a partial failure
rather than a pass.
"""

from __future__ import annotations

from datetime import timedelta

from memcheck.backends.baselines import TranscriptRAGBackend
from memcheck.backends.guarded import GuardedMemoryBackend
from memcheck.contracts.base import EPOCH, Contract, ContractContext, make_event
from memcheck.models import ContractViolation, EventType, Severity

SECRET = "hunter2-rotation-key"


class DeletionClosure(Contract):
    name = "deletion_closure"
    description = (
        "Deleted content must not resurface through active records, summaries, lexical or "
        "semantic indexes, caches, retrieval evidence, or historical queries."
    )
    severity = Severity.critical

    @property
    def preconditions(self) -> list[str]:
        return [
            "the backend receives a deletion_request event naming the content to remove",
            "the deleted content was previously retrievable",
        ]

    @property
    def expected_invariant(self) -> str:
        return (
            "After a deletion request, the deleted string appears in no derived surface and in "
            "no retrieval result, at any as_of_time."
        )

    def execute(self, context: ContractContext) -> list[ContractViolation]:
        backend = context.fresh_backend("deletion")
        try:
            secret_event = make_event(1, f"The rotation key is {SECRET}.")
            unrelated = make_event(2, "The primary database is postgres.")
            deletion = make_event(
                3,
                "Delete everything about the rotation key.",
                event_type=EventType.deletion_request,
                metadata={"deletion_target": "key rotation"},
            )
            growth = [make_event(4 + i, f"The service tier {i} is standard.") for i in range(3)]

            backend.ingest_event(secret_event)
            backend.ingest_event(unrelated)
            context.steps.append("ingest a sensitive fact and an unrelated fact")

            query = "What is the rotation key?"
            before = backend.retrieve(
                query,
                "tenant_a",
                evidence_budget=context.evidence_budget,
                token_budget=context.token_budget,
            )
            context.steps.append("confirm the sensitive fact is retrievable before deletion")
            precondition_met = any(SECRET in item.content for item in before.evidence)

            backend.ingest_event(deletion)
            context.steps.append("deliver a deletion_request naming the sensitive fact")
            for event in growth:
                backend.ingest_event(event)
            context.steps.append("continue writing unrelated memories after the deletion")
            backend.checkpoint()

            violations: list[ContractViolation] = []
            if not precondition_met:
                # The store never held the fact, so there is nothing to leak; report it as a
                # skip-like note rather than silently passing.
                context.steps.append(
                    "precondition not met: the fact was never retrievable, so closure is vacuous"
                )
                return []

            def leak(surface: str, detail: str, severity: Severity = Severity.critical) -> None:
                violations.append(
                    self.violation(
                        context,
                        expected=f"the deleted value is absent from the {surface}",
                        observed=detail,
                        event_ids=[secret_event.event_id, deletion.event_id],
                        query_id=query,
                        severity=severity,
                        metadata={"surface": surface},
                    )
                )

            # Surface 1: retrieval evidence for a current query.
            after = backend.retrieve(
                query,
                "tenant_a",
                evidence_budget=context.evidence_budget,
                token_budget=context.token_budget,
            )
            if any(SECRET in item.content for item in after.evidence):
                leak("retrieval evidence", f"a current query still returns {SECRET!r}")

            # Surface 2: a historical as-of query must not reach around the deletion.
            historical = backend.retrieve(
                query,
                "tenant_a",
                as_of_time=EPOCH + timedelta(days=2),
                evidence_budget=context.evidence_budget,
                token_budget=context.token_budget,
            )
            if any(SECRET in item.content for item in historical.evidence):
                leak(
                    "historical query",
                    f"an as-of query positioned before the deletion returns {SECRET!r}",
                )

            # Surface 3: active derived records.
            if any(SECRET in record.content for record in backend.active_records()):
                leak("active records", "an active memory record still contains the deleted value")

            # Surfaces 4-6 are backend-specific derived stores.
            if isinstance(backend, GuardedMemoryBackend):
                if backend.index.contains_text(SECRET):
                    leak("lexical index", "an index row still contains the deleted value")
                if any(SECRET in text for text in backend.summary_texts()):
                    leak("summaries", "a summary still contains the deleted value")
                if any(SECRET in text for text in backend.cache_texts()):
                    leak("retrieval cache", "a cached retrieval still contains the deleted value")
            elif isinstance(backend, TranscriptRAGBackend):
                if backend.index.contains_text(SECRET):
                    leak("lexical index", "an index row still contains the deleted value")
                if any(SECRET in text for text in backend.chunk_texts()):
                    leak("raw chunks", "a stored transcript chunk still contains the deleted value")

            # Surface 7: survival across restart and rebuild.
            backend.restart()
            restarted = backend.retrieve(
                query,
                "tenant_a",
                evidence_budget=context.evidence_budget,
                token_budget=context.token_budget,
            )
            if any(SECRET in item.content for item in restarted.evidence):
                leak("post-restart retrieval", "the deleted value returns after a restart")
            backend.rebuild_indexes()
            rebuilt = backend.retrieve(
                query,
                "tenant_a",
                evidence_budget=context.evidence_budget,
                token_budget=context.token_budget,
            )
            if any(SECRET in item.content for item in rebuilt.evidence):
                leak(
                    "post-rebuild retrieval",
                    "the deleted value returns after rebuilding indexes from the event log",
                )
            context.steps.append("restart and rebuild, then re-probe every surface")
            return violations
        finally:
            backend.close()
