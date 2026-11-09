"""The six deterministic faults (Section 13)."""

from __future__ import annotations

from datetime import timedelta

from memcheck.backends.base import MemoryBackend
from memcheck.backends.guarded import GuardedMemoryBackend, InterruptedWrite
from memcheck.contracts.base import EPOCH, make_event
from memcheck.faults.base import Fault, FaultContext
from memcheck.models import Event, EventType
from memcheck.providers.mock import normalize_value


class DuplicateEventDelivery(Fault):
    """13.1 — deliver the same event twice."""

    name = "duplicate_event_delivery"
    injection_point = "delivery stream, before ingestion"
    activation_condition = "every event is delivered a second time immediately after the first"
    affected_components = ("event log", "derived memories", "retrieval index")
    expected_recovery_behavior = (
        "the store detects the duplicate on event id or source hash, ignores it, and answers "
        "identically to the un-duplicated stream"
    )

    def wrap_events(self, events: list[Event], context: FaultContext) -> list[Event]:
        doubled: list[Event] = []
        for event in events:
            doubled.append(event)
            doubled.append(event.model_copy(deep=True))
            context.events_affected.append(event.event_id)
        context.metadata["delivered"] = len(doubled)
        return doubled

    def verify(self, backend: MemoryBackend, context: FaultContext) -> tuple[bool, str]:
        records = backend.active_records()
        if not records:
            return True, "backend derives no memories; duplicates cannot amplify stored state"
        # Identity is per tenant and per fact, not global content. Two tenants asserting the same
        # thing are two legitimate memories, not an amplification.
        identities = [
            (record.tenant_id, record.canonical_key, normalize_value(record.content))
            for record in records
        ]
        unique = set(identities)
        if len(identities) == len(unique):
            return True, (
                f"{len(unique)} distinct (tenant, fact, value) memories, no duplicate amplification"
            )
        amplified = len(identities) - len(unique)
        return False, (
            f"{amplified} redundant active memories: duplicate delivery amplified stored state "
            f"({len(identities)} records for {len(unique)} distinct (tenant, fact, value) triples)"
        )


class OutOfOrderDelivery(Fault):
    """13.2 — deliver a newer update before the earlier fact, then replay the correct order."""

    name = "out_of_order_delivery"
    injection_point = "delivery stream ordering"
    activation_condition = "adjacent pairs of events are swapped, then the correct order replayed"
    affected_components = ("event log", "versioning", "validity intervals")
    expected_recovery_behavior = (
        "after replaying the full correct order, the newest fact by event_time is the one a "
        "current query returns, regardless of the order in which events arrived"
    )

    def wrap_events(self, events: list[Event], context: FaultContext) -> list[Event]:
        shuffled = list(events)
        for index in range(0, len(shuffled) - 1, 2):
            shuffled[index], shuffled[index + 1] = shuffled[index + 1], shuffled[index]
            context.events_affected.extend([shuffled[index].event_id, shuffled[index + 1].event_id])
        # Then replay the full correct order, as the specification requires.
        context.metadata["replayed_correct_order"] = True
        return shuffled + list(events)

    def verify(self, backend: MemoryBackend, context: FaultContext) -> tuple[bool, str]:
        records = [r for r in backend.active_records() if r.valid_to is None]
        if not records:
            return True, "backend derives no memories; ordering cannot corrupt derived state"
        # Identity is per tenant *and* per fact. Grouping on the canonical key alone counts one
        # legitimate record per tenant as competing versions of a single fact, which reports a
        # violation for every key that more than one tenant happens to mention.
        by_key: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for record in records:
            by_key.setdefault((record.tenant_id, record.canonical_key), []).append(
                (record.valid_from.isoformat(), record.content)
            )
        unresolved = {key: rows for key, rows in by_key.items() if len(rows) > 1}
        if not unresolved:
            return True, (
                f"each of {len(by_key)} (tenant, fact) pairs has exactly one open active version "
                "after replay"
            )
        return False, (
            f"{len(unresolved)} (tenant, fact) pair(s) retain multiple open active versions after "
            f"replaying the correct order: {sorted(unresolved)[:5]}"
        )


class InterruptedWriteFault(Fault):
    """13.3 — terminate after the event log append but before the derived index completes.

    Modelled as a *logical* abort inside the write sequence rather than a real process kill; a
    real SIGKILL cannot be driven deterministically in-process, and it would make SQLite's own
    WAL recovery the thing under test (DEVIATIONS V-003).
    """

    name = "interrupted_write"
    injection_point = "between the event log append and the derived index update"
    activation_condition = "one designated event aborts after its log append is committed"
    affected_components = ("event log", "derived memories", "retrieval index", "write journal")
    expected_recovery_behavior = (
        "on restart the store notices the incomplete write from its journal, replays the event "
        "from the log, and the interrupted fact becomes retrievable"
    )

    def perturb(self, backend: MemoryBackend, context: FaultContext) -> None:
        if not isinstance(backend, GuardedMemoryBackend):
            context.notes.append(
                f"{context.backend_name} has no separable log/index write phases; the fault "
                "degenerates to a plain ingest, which is itself the finding"
            )
            backend.ingest_event(self._victim())
            return
        try:
            backend.ingest_event(self._victim(), abort_after="log_append")
        except InterruptedWrite as interruption:
            context.notes.append(str(interruption))
            context.events_affected.append(self._victim().event_id)
            context.metadata["aborted"] = True

    def recover(self, backend: MemoryBackend, context: FaultContext) -> None:
        # Recovery is what a process restart does; the backend repairs on the way back up.
        backend.restart()

    def verify(self, backend: MemoryBackend, context: FaultContext) -> tuple[bool, str]:
        victim = self._victim()
        result = backend.retrieve("What is the escalation contact?", victim.tenant_id)
        if any("oncall-team" in item.content for item in result.evidence):
            return True, "the interrupted write was replayed from the log and is now retrievable"
        return False, (
            "the interrupted event is durable in the log but is not retrievable after restart: "
            "derived state is permanently behind the source of truth"
        )

    @staticmethod
    def _victim() -> Event:
        return make_event(
            900,
            "The escalation contact is oncall-team.",
            event_id="evt_interrupted",
            days=900,
        )


class IndexLoss(Fault):
    """13.4 — delete or corrupt a derived retrieval index, then rebuild from the log."""

    name = "index_loss"
    injection_point = "derived retrieval index"
    activation_condition = "half the index rows are deleted at the checkpoint"
    affected_components = ("retrieval index",)
    expected_recovery_behavior = (
        "rebuild_indexes() reconstructs the index from the event log alone and retrieval returns "
        "the same evidence as before the loss"
    )

    def perturb(self, backend: MemoryBackend, context: FaultContext) -> None:
        index = getattr(backend, "index", None)
        if index is None:
            context.notes.append(
                f"{context.backend_name} maintains no derived index; nothing to lose"
            )
            context.metadata["rows_destroyed"] = 0
            return
        self._before = _probe_fingerprint(backend)
        destroyed = index.corrupt(context.rng(), fraction=0.5)
        backend.checkpoint()
        context.metadata["rows_destroyed"] = destroyed
        context.notes.append(f"deleted {destroyed} index row(s) without touching the event log")

    def recover(self, backend: MemoryBackend, context: FaultContext) -> None:
        rebuilt = backend.rebuild_indexes()
        context.metadata["documents_rebuilt"] = rebuilt

    def verify(self, backend: MemoryBackend, context: FaultContext) -> tuple[bool, str]:
        if not context.metadata.get("rows_destroyed"):
            return True, "no derived index to lose; retrieval is unaffected"
        after = _probe_fingerprint(backend)
        before = getattr(self, "_before", after)
        if before == after:
            return True, "the rebuilt index reproduces the pre-loss retrieval results exactly"
        return False, (
            "retrieval results differ after rebuilding from the event log: the log is not "
            f"sufficient to reconstruct the index (before={before}, after={after})"
        )


class ContradictoryUpdate(Fault):
    """13.5 — two incompatible facts whose temporal order is uncertain."""

    name = "contradictory_update"
    injection_point = "delivery stream, mid-session"
    activation_condition = "two events assert different values for one fact at the same event_time"
    affected_components = ("derived memories", "conflict detection", "abstention")
    expected_recovery_behavior = (
        "the store records a conflict and abstains rather than silently selecting one value "
        "without evidence"
    )

    QUERY = "What is the incident severity?"

    def wrap_events(self, events: list[Event], context: FaultContext) -> list[Event]:
        contradiction_time = 800
        first = make_event(
            contradiction_time,
            "The incident severity is critical.",
            event_id="evt_conflict_a",
            days=contradiction_time,
        )
        second = make_event(
            contradiction_time,
            "The incident severity is minor.",
            event_id="evt_conflict_b",
            days=contradiction_time,
        )
        context.events_affected.extend([first.event_id, second.event_id])
        midpoint = len(events) // 2
        return [*events[:midpoint], first, second, *events[midpoint:]]

    def verify(self, backend: MemoryBackend, context: FaultContext) -> tuple[bool, str]:
        result = backend.retrieve(self.QUERY, "tenant_a")
        values = {normalize_value(item.content) for item in result.evidence}
        contains_critical = any("critical" in value for value in values)
        contains_minor = any("minor" in value for value in values)
        if result.abstained:
            return True, f"abstained on the contradiction ({result.abstention_reason})"
        if contains_critical and contains_minor:
            return True, (
                "both contradictory values were surfaced with their provenance, leaving the "
                "conflict visible to the reader rather than resolving it silently"
            )
        if contains_critical or contains_minor:
            chosen = "critical" if contains_critical else "minor"
            return False, (
                f"silently selected {chosen!r} as the answer although nothing in the evidence "
                "establishes which of the two contradictory facts is current"
            )
        return True, "neither contradictory value was retrieved"


class DeletionDuringGrowth(Fault):
    """13.6 — delete a fact after it has influenced summaries and indexes, then keep writing."""

    name = "deletion_during_growth"
    injection_point = "delivery stream, after the fact has been consolidated"
    activation_condition = (
        "a deletion request is delivered mid-stream, followed by unrelated writes"
    )
    affected_components = ("derived memories", "summaries", "retrieval index", "retrieval cache")
    expected_recovery_behavior = (
        "the deleted fact does not resurface through any surface as the store continues to grow"
    )

    SECRET = "classified-merger-target"
    QUERY = "What is the merger target?"

    def wrap_events(self, events: list[Event], context: FaultContext) -> list[Event]:
        secret = make_event(
            700,
            f"The merger target is {self.SECRET}.",
            event_id="evt_growth_secret",
            days=700,
        )
        deletion = make_event(
            701,
            "Delete everything about the merger target.",
            event_id="evt_growth_delete",
            days=701,
            event_type=EventType.deletion_request,
            metadata={"deletion_target": "merger target"},
        )
        follow_up = [
            make_event(
                702 + index,
                f"The quarterly milestone {index} is on track.",
                event_id=f"evt_growth_{index}",
                days=702 + index,
                session_id=f"session_growth_{index:02d}",
            )
            for index in range(5)
        ]
        context.events_affected.extend([secret.event_id, deletion.event_id])
        midpoint = max(1, len(events) // 2)
        return [*events[:midpoint], secret, deletion, *follow_up, *events[midpoint:]]

    def verify(self, backend: MemoryBackend, context: FaultContext) -> tuple[bool, str]:
        surfaces: list[str] = []
        result = backend.retrieve(self.QUERY, "tenant_a")
        if any(self.SECRET in item.content for item in result.evidence):
            surfaces.append("retrieval evidence")
        historical = backend.retrieve(
            self.QUERY, "tenant_a", as_of_time=EPOCH + timedelta(days=700, hours=12)
        )
        if any(self.SECRET in item.content for item in historical.evidence):
            surfaces.append("historical query")
        if any(self.SECRET in record.content for record in backend.active_records()):
            surfaces.append("active records")
        if isinstance(backend, GuardedMemoryBackend):
            if any(self.SECRET in text for text in backend.summary_texts()):
                surfaces.append("summaries")
            if backend.index.contains_text(self.SECRET):
                surfaces.append("lexical index")
            if any(self.SECRET in text for text in backend.cache_texts()):
                surfaces.append("retrieval cache")
        context.metadata["leaking_surfaces"] = surfaces
        if not surfaces:
            return True, "the deleted fact did not resurface on any probed surface"
        return False, f"the deleted fact resurfaced on: {', '.join(surfaces)}"


def _probe_fingerprint(backend: MemoryBackend) -> str:
    """Ordered retrieval results for the standard probe set, as a comparable string."""
    from memcheck.contracts.restart_equivalence import PROBE_QUERIES

    parts: list[str] = []
    for query, tenant in PROBE_QUERIES:
        result = backend.retrieve(query, tenant)
        parts.append(
            f"{tenant}|{query}|{[item.memory_id for item in result.evidence]}|{result.abstained}"
        )
    return " || ".join(parts)
