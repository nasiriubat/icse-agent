"""The three baseline conditions.

They share an event store so that ingestion cost, storage growth, and tenant bookkeeping are
measured the same way across conditions; they differ only in what they do with the events.

None of the baselines interprets a ``deletion_request`` event — that is not an oversight but the
behaviour under test. Deletion handling is a property of the guarded layer, and the deletion
closure contract measures exactly the gap.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from memcheck.backends.base import BackendError, MemoryBackend
from memcheck.backends.retrieval import LexicalIndex
from memcheck.event_log import EventLog
from memcheck.models import (
    AbstentionReason,
    DeleteResult,
    Event,
    Evidence,
    RetrievalResult,
    StoreStatistics,
    WriteOutcome,
    WriteResult,
    utcnow,
)
from memcheck.providers.mock import content_terms
from memcheck.utilities.hashing import stable_id
from memcheck.utilities.tokens import count_tokens

CHUNK_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    event_id    TEXT NOT NULL,
    event_time  TEXT NOT NULL,
    content     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks (tenant_id);

CREATE TABLE IF NOT EXISTS truncations (
    truncation_id TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    query         TEXT NOT NULL,
    dropped_items INTEGER NOT NULL,
    kept_items    INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
"""


class _EventStoreBackend(MemoryBackend):
    """Shared storage plumbing: an event log plus statistics."""

    def __init__(
        self,
        path: str | Path,
        *,
        evidence_budget: int = 5,
        token_budget: int = 512,
    ) -> None:
        self.path = Path(path)
        self.evidence_budget = evidence_budget
        self.token_budget = token_budget
        self._connection: sqlite3.Connection | None = None
        self._log: EventLog | None = None

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(CHUNK_SCHEMA)
        self._log = EventLog(self._connection)
        self._connection.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise BackendError("backend is not initialized; call initialize() first")
        return self._connection

    @property
    def log(self) -> EventLog:
        if self._log is None:
            raise BackendError("backend is not initialized; call initialize() first")
        return self._log

    def checkpoint(self) -> None:
        self.connection.commit()

    def restart(self) -> None:
        self.close()
        self.initialize()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.commit()
            self._connection.close()
            self._connection = None
            self._log = None

    def _record_truncation(self, tenant_id: str, query: str, dropped: int, kept: int) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO truncations (truncation_id, tenant_id, query, dropped_items, "
            "kept_items, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                stable_id("trunc", tenant_id, query, dropped, kept),
                tenant_id,
                query,
                dropped,
                kept,
                utcnow().isoformat(),
            ),
        )

    def truncation_count(self) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(SUM(dropped_items), 0) AS n FROM truncations"
        ).fetchone()
        return int(row["n"])

    def get_statistics(self, *, tenant_id: str | None = None) -> StoreStatistics:
        events = self.log.count(tenant_id)
        duplicates = self.log.duplicate_count(tenant_id)
        sessions = self.log.sessions_ingested(tenant_id)
        database_size = self.path.stat().st_size if self.path.exists() else 0
        return StoreStatistics(
            backend_name=self.name,
            tenant_id=tenant_id,
            database_size_bytes=database_size,
            index_size_bytes=self._index_size(),
            event_count=events,
            memory_record_count=self._record_count(tenant_id),
            active_memory_count=self._record_count(tenant_id),
            superseded_record_count=0,
            deleted_record_count=0,
            conflicted_record_count=0,
            duplicate_ratio=duplicates / max(1, events + duplicates),
            sessions_ingested=sessions,
            storage_growth_per_session=database_size / max(1, sessions),
            metadata={"truncated_items": self.truncation_count()},
        )

    def _index_size(self) -> int:
        return 0

    def _record_count(self, tenant_id: str | None) -> int:
        return self.log.count(tenant_id)

    def export_state(self) -> dict[str, Any]:
        return {"events": [event.model_dump(mode="json") for event in self.log.replay()]}

    def import_state(self, state: dict[str, Any]) -> None:
        for payload in state.get("events", []):
            self.ingest_event(Event.model_validate(payload))


class NoMemoryBackend(_EventStoreBackend):
    """No persistent memory: only the current request and the current session are visible.

    Events are still stored so that ingestion cost and storage growth are measured on the same
    basis as every other condition, but retrieval never reads beyond the active session.
    """

    name = "no_memory"

    def ingest_event(self, event: Event) -> WriteResult:
        started = time.perf_counter()
        append = self.log.append(event)
        self.connection.commit()
        return WriteResult(
            event_id=event.event_id,
            tenant_id=event.tenant_id,
            outcome=WriteOutcome.created if append.appended else WriteOutcome.duplicate_ignored,
            backend_name=self.name,
            latency_ms=(time.perf_counter() - started) * 1000,
            notes="stored without deriving memory",
        )

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
        started = time.perf_counter()
        row = self.connection.execute(
            "SELECT session_id FROM events WHERE tenant_id = ? ORDER BY sequence DESC LIMIT 1",
            (tenant_id,),
        ).fetchone()
        current_session = str(row["session_id"]) if row is not None else None

        evidence: list[Evidence] = []
        injected = 0
        limit_tokens = token_budget if token_budget is not None else self.token_budget
        limit_count = evidence_budget if evidence_budget is not None else self.evidence_budget
        if current_session is not None:
            query_terms = content_terms(query)
            for event in self.log.replay(tenant_id=tenant_id, session_id=current_session):
                if not query_terms or not query_terms.issubset(content_terms(event.content)):
                    continue
                tokens = count_tokens(event.content)
                if len(evidence) >= limit_count or injected + tokens > limit_tokens:
                    break
                injected += tokens
                evidence.append(
                    Evidence(
                        memory_id=f"event:{event.event_id}",
                        source_event_ids=[event.event_id],
                        content=event.content,
                        relevance_score=1.0,
                        confidence=1.0,
                        valid_from=event.event_time,
                        retrieval_reason="current_session_only",
                        token_count=tokens,
                    )
                )
        return RetrievalResult(
            query=query,
            tenant_id=tenant_id,
            evidence=evidence,
            abstained=not evidence,
            abstention_reason=(AbstentionReason.backend_has_no_memory if not evidence else None),
            retrieval_latency_ms=(time.perf_counter() - started) * 1000,
            injected_tokens=injected,
            backend_name=self.name,
            metadata={"current_session": current_session, "query_id": query_id},
        )

    def delete(self, identifier: str, tenant_id: str) -> DeleteResult:
        # Nothing persists beyond the session, so there is nothing to delete.
        return DeleteResult(tenant_id=tenant_id, target=identifier, backend_name=self.name)

    def rebuild_indexes(self) -> int:
        return 0


class FullHistoryBackend(_EventStoreBackend):
    """Replay the chronological history, truncated to the same token limit as everyone else.

    Truncation keeps the most recent events, which is what a context-window-bounded replay does.
    Every truncation decision is recorded so the recall loss is visible in the results rather
    than being hidden inside the backend.
    """

    name = "full_history"

    def ingest_event(self, event: Event) -> WriteResult:
        started = time.perf_counter()
        append = self.log.append(event)
        self.connection.commit()
        return WriteResult(
            event_id=event.event_id,
            tenant_id=event.tenant_id,
            outcome=WriteOutcome.created if append.appended else WriteOutcome.duplicate_ignored,
            backend_name=self.name,
            latency_ms=(time.perf_counter() - started) * 1000,
            notes="appended to replayable history",
        )

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
        started = time.perf_counter()
        limit_tokens = token_budget if token_budget is not None else self.token_budget
        limit_count = evidence_budget if evidence_budget is not None else self.evidence_budget

        history = list(self.log.replay(tenant_id=tenant_id, until=as_of_time))
        evidence: list[Evidence] = []
        injected = 0
        # Most recent first: the window a bounded replay actually keeps.
        for event in reversed(history):
            tokens = count_tokens(event.content)
            if len(evidence) >= limit_count or injected + tokens > limit_tokens:
                break
            injected += tokens
            evidence.append(
                Evidence(
                    memory_id=f"event:{event.event_id}",
                    source_event_ids=[event.event_id],
                    content=event.content,
                    relevance_score=1.0,
                    confidence=1.0,
                    valid_from=event.event_time,
                    retrieval_reason="chronological_replay_window",
                    token_count=tokens,
                )
            )
        dropped = len(history) - len(evidence)
        if dropped > 0:
            self._record_truncation(tenant_id, query, dropped, len(evidence))
            self.connection.commit()

        return RetrievalResult(
            query=query,
            tenant_id=tenant_id,
            evidence=evidence,
            abstained=not evidence,
            abstention_reason=AbstentionReason.no_evidence if not evidence else None,
            retrieval_latency_ms=(time.perf_counter() - started) * 1000,
            injected_tokens=injected,
            backend_name=self.name,
            metadata={
                "history_length": len(history),
                "dropped_by_truncation": dropped,
                "query_id": query_id,
            },
        )

    def delete(self, identifier: str, tenant_id: str) -> DeleteResult:
        # A replayed history has no deletion mechanism: the transcript is the store.
        return DeleteResult(tenant_id=tenant_id, target=identifier, backend_name=self.name)

    def rebuild_indexes(self) -> int:
        return 0


class TranscriptRAGBackend(_EventStoreBackend):
    """Raw transcript chunks retrieved lexically, with no consolidation.

    This is the common production shape: chunk everything, embed or index it, retrieve top-k.
    It has no notion of a fact being superseded and no notion of a fact being deleted.
    """

    name = "transcript_rag"

    def __init__(
        self,
        path: str | Path,
        *,
        evidence_budget: int = 5,
        token_budget: int = 512,
        chunk_tokens: int = 120,
    ) -> None:
        super().__init__(path, evidence_budget=evidence_budget, token_budget=token_budget)
        self.chunk_tokens = chunk_tokens
        self._index: LexicalIndex | None = None

    def initialize(self) -> None:
        super().initialize()
        self._index = LexicalIndex(self.connection)
        self.connection.commit()

    @property
    def index(self) -> LexicalIndex:
        if self._index is None:
            raise BackendError("backend is not initialized; call initialize() first")
        return self._index

    def close(self) -> None:
        super().close()
        self._index = None

    def ingest_event(self, event: Event) -> WriteResult:
        started = time.perf_counter()
        append = self.log.append(event)
        if not append.appended:
            self.connection.commit()
            return WriteResult(
                event_id=event.event_id,
                tenant_id=event.tenant_id,
                outcome=WriteOutcome.duplicate_ignored,
                backend_name=self.name,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        chunk_ids = self._chunk_and_index(event)
        self.connection.commit()
        return WriteResult(
            event_id=event.event_id,
            tenant_id=event.tenant_id,
            outcome=WriteOutcome.created,
            created_memory_ids=chunk_ids,
            backend_name=self.name,
            latency_ms=(time.perf_counter() - started) * 1000,
            notes=f"indexed {len(chunk_ids)} raw chunk(s); no consolidation",
        )

    def _chunk_and_index(self, event: Event) -> list[str]:
        chunks = _split_into_chunks(event.content, self.chunk_tokens)
        chunk_ids: list[str] = []
        for position, chunk in enumerate(chunks):
            chunk_id = stable_id("chunk", event.tenant_id, event.event_id, position)
            self.connection.execute(
                "INSERT OR REPLACE INTO chunks (chunk_id, tenant_id, session_id, event_id, "
                "event_time, content) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    event.tenant_id,
                    event.session_id,
                    event.event_id,
                    event.event_time.isoformat(),
                    chunk,
                ),
            )
            self.index.add(chunk_id, event.tenant_id, chunk, kind="chunk")
            chunk_ids.append(chunk_id)
        return chunk_ids

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
        started = time.perf_counter()
        limit_tokens = token_budget if token_budget is not None else self.token_budget
        limit_count = evidence_budget if evidence_budget is not None else self.evidence_budget

        hits = self.index.search(query, tenant_id, limit=max(limit_count * 4, 20))
        evidence: list[Evidence] = []
        injected = 0
        for hit in hits:
            row = self.connection.execute(
                "SELECT * FROM chunks WHERE chunk_id = ? AND tenant_id = ?",
                (hit.doc_id, tenant_id),
            ).fetchone()
            if row is None:
                continue
            event_time = datetime.fromisoformat(str(row["event_time"]))
            if as_of_time is not None and event_time > as_of_time:
                continue
            tokens = count_tokens(hit.content)
            if len(evidence) >= limit_count or injected + tokens > limit_tokens:
                break
            injected += tokens
            evidence.append(
                Evidence(
                    memory_id=hit.doc_id,
                    source_event_ids=[str(row["event_id"])],
                    content=hit.content,
                    relevance_score=hit.score,
                    confidence=1.0,
                    # A raw chunk has a timestamp but no validity interval: nothing in the store
                    # ever says a chunk has stopped being true.
                    valid_from=event_time,
                    valid_to=None,
                    retrieval_reason=f"bm25={hit.score:.3f}; raw_chunk",
                    token_count=tokens,
                )
            )
        return RetrievalResult(
            query=query,
            tenant_id=tenant_id,
            evidence=evidence,
            abstained=not evidence,
            abstention_reason=AbstentionReason.no_evidence if not evidence else None,
            retrieval_latency_ms=(time.perf_counter() - started) * 1000,
            injected_tokens=injected,
            backend_name=self.name,
            metadata={"candidates_considered": len(hits), "query_id": query_id},
        )

    def delete(self, identifier: str, tenant_id: str) -> DeleteResult:
        """Delete a chunk by chunk id or event id.

        A canonical key or a free-text request cannot be resolved by a store that only holds raw
        chunks; those return zero deletions, which is the leak the closure contract reports.
        """
        rows = self.connection.execute(
            "SELECT chunk_id FROM chunks WHERE tenant_id = ? AND (chunk_id = ? OR event_id = ?)",
            (tenant_id, identifier, identifier),
        ).fetchall()
        deleted: list[str] = []
        purged = 0
        for row in rows:
            chunk_id = str(row["chunk_id"])
            self.connection.execute(
                "DELETE FROM chunks WHERE chunk_id = ? AND tenant_id = ?", (chunk_id, tenant_id)
            )
            purged += self.index.remove(chunk_id, tenant_id)
            deleted.append(chunk_id)
        self.connection.commit()
        return DeleteResult(
            tenant_id=tenant_id,
            target=identifier,
            deleted_memory_ids=deleted,
            purged_index_rows=purged,
            backend_name=self.name,
        )

    def rebuild_indexes(self) -> int:
        self.index.create()
        self.index.clear()
        self.connection.execute("DELETE FROM chunks")
        for event in self.log.replay():
            self._chunk_and_index(event)
        self.connection.commit()
        return self.index.document_count()

    def _index_size(self) -> int:
        return self.index.size_bytes()

    def _record_count(self, tenant_id: str | None) -> int:
        if tenant_id is None:
            row = self.connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
        return int(row["n"])

    def chunk_texts(self, tenant_id: str | None = None) -> list[str]:
        if tenant_id is None:
            rows = self.connection.execute("SELECT content FROM chunks")
        else:
            rows = self.connection.execute(
                "SELECT content FROM chunks WHERE tenant_id = ?", (tenant_id,)
            )
        return [str(row["content"]) for row in rows]

    def export_state(self) -> dict[str, Any]:
        state = super().export_state()
        state["chunks"] = [
            dict(row) for row in self.connection.execute("SELECT * FROM chunks ORDER BY chunk_id")
        ]
        return state


def _split_into_chunks(text: str, chunk_tokens: int) -> list[str]:
    """Split on whitespace into fixed-size token windows, preserving order."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    for start in range(0, len(words), chunk_tokens):
        chunks.append(" ".join(words[start : start + chunk_tokens]))
    return chunks
