"""The ``MemoryBackend`` interface every condition implements.

Backends return structured results, never printed text (DECISIONS D-005). Retrieval always
requires an explicit ``tenant_id``; there is no "all tenants" mode.
"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Any

from memcheck.models import (
    BackendState,
    DeleteResult,
    Event,
    MemoryRecord,
    RetrievalResult,
    StoreStatistics,
    WriteResult,
)


class BackendError(RuntimeError):
    """Raised for backend misuse — a missing tenant filter, an unknown record, a closed store."""


class MemoryBackend(abc.ABC):
    """Common surface for no-memory, full-history, transcript-RAG, and guarded memory."""

    name: str = "abstract"

    #: Guards the guarded backend can disable for ablations. Other backends ignore these.
    supports_ablations: tuple[str, ...] = ()

    @abc.abstractmethod
    def initialize(self) -> None:
        """Create storage. Must be idempotent so restart is a no-op on an existing store."""

    @abc.abstractmethod
    def ingest_event(self, event: Event) -> WriteResult:
        """Ingest one event and return what changed."""

    def ingest_events(self, events: list[Event]) -> list[WriteResult]:
        """Ingest a batch in order. Overridden where a backend can batch more cheaply."""
        return [self.ingest_event(event) for event in events]

    @abc.abstractmethod
    def retrieve(
        self,
        query: str,
        tenant_id: str,
        *,
        as_of_time: datetime | None = None,
        token_budget: int | None = None,
        evidence_budget: int | None = None,
        query_id: str | None = None,
    ) -> RetrievalResult:
        """Return bounded evidence, with a reason for every item and an explicit abstention."""

    @abc.abstractmethod
    def delete(self, identifier: str, tenant_id: str) -> DeleteResult:
        """Delete by event id, memory id, or canonical key, across every derived surface."""

    @abc.abstractmethod
    def checkpoint(self) -> None:
        """Flush durable state. Called at every longitudinal checkpoint."""

    @abc.abstractmethod
    def restart(self) -> None:
        """Close and reopen the store from durable state, as a clean process restart would."""

    @abc.abstractmethod
    def rebuild_indexes(self) -> int:
        """Rebuild every derived index from the source of truth. Returns documents rebuilt."""

    @abc.abstractmethod
    def export_state(self) -> dict[str, Any]:
        """Serializable snapshot of durable state."""

    @abc.abstractmethod
    def import_state(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`export_state`."""

    @abc.abstractmethod
    def get_statistics(self, *, tenant_id: str | None = None) -> StoreStatistics:
        """Storage and growth metrics (Section 16.5)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources."""

    # -- introspection used by contracts ------------------------------------------------
    def active_records(self, tenant_id: str | None = None) -> list[MemoryRecord]:
        """Active derived memories. Backends without derived memories return an empty list."""
        return []

    def all_records(self, tenant_id: str | None = None) -> list[MemoryRecord]:
        return []

    def canonical_state(
        self, probes: list[tuple[str, str]], *, evidence_budget: int, token_budget: int
    ) -> BackendState:
        """Canonical projection for restart equivalence (DECISIONS D-007).

        ``probes`` is a list of ``(query_id, tenant_id)`` pairs whose queries are looked up from
        the same workload on both sides of a restart, so the comparison is like-for-like.
        """
        records = [
            (
                record.tenant_id,
                record.canonical_key,
                record.version,
                record.content,
                record.status.value,
                record.valid_from.isoformat(),
                record.valid_to.isoformat() if record.valid_to else None,
            )
            for record in self.all_records()
            if record.status.value != "deleted"
        ]
        probe_results: dict[str, list[tuple[str, int]]] = {}
        probe_abstentions: dict[str, bool] = {}
        version_by_id = {record.memory_id: record.version for record in self.all_records()}
        for query, tenant_id in probes:
            result = self.retrieve(
                query,
                tenant_id,
                token_budget=token_budget,
                evidence_budget=evidence_budget,
            )
            probe_results[f"{tenant_id}::{query}"] = [
                (item.memory_id, version_by_id.get(item.memory_id, 0)) for item in result.evidence
            ]
            probe_abstentions[f"{tenant_id}::{query}"] = result.abstained
        return BackendState(
            records=records, probe_results=probe_results, probe_abstentions=probe_abstentions
        )

    def __enter__(self) -> MemoryBackend:
        self.initialize()
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()
