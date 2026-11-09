"""GuardedMemoryBackend — the proposed system.

The write lifecycle (Section 10) and the retrieval lifecycle (Section 11) are implemented
literally, each step named in the code so a reviewer can check the implementation against the
specification. Each guard can be disabled individually for the ablation study; the disabled
guard is recorded in the manifest and is the *only* permitted difference in a fair comparison.

None of the individual mechanisms — event sourcing, provenance, versioning, tombstones, temporal
validity — is claimed as novel. The contribution is the contract-based evaluation around them.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from memcheck.backends.base import BackendError, MemoryBackend
from memcheck.backends.extraction import Extractor, RuleBasedExtractor
from memcheck.backends.retrieval import LexicalIndex
from memcheck.event_log import EventLog
from memcheck.models import (
    AbstentionReason,
    DeleteResult,
    Event,
    EventType,
    Evidence,
    MemoryRecord,
    MemoryStatus,
    RetrievalResult,
    StoreStatistics,
    WriteOutcome,
    WriteResult,
    content_hash,
    utcnow,
)
from memcheck.providers.mock import content_terms, extract_value, normalize_value
from memcheck.utilities.hashing import stable_id
from memcheck.utilities.tokens import count_tokens

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id            TEXT PRIMARY KEY,
    tenant_id            TEXT NOT NULL,
    canonical_key        TEXT NOT NULL,
    version              INTEGER NOT NULL,
    content              TEXT NOT NULL,
    status               TEXT NOT NULL,
    valid_from           TEXT NOT NULL,
    valid_to             TEXT,
    confidence           REAL NOT NULL DEFAULT 1.0,
    source_event_ids     TEXT NOT NULL,
    supersedes_memory_id TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    embedding_model      TEXT,
    metadata             TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories (tenant_id, canonical_key, status);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories (tenant_id, status);

CREATE TABLE IF NOT EXISTS tombstones (
    tombstone_id   TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL,
    target         TEXT NOT NULL,
    canonical_key  TEXT,
    content_hash   TEXT,
    created_at     TEXT NOT NULL,
    reason         TEXT NOT NULL DEFAULT 'deletion_request'
);
CREATE INDEX IF NOT EXISTS idx_tombstones_tenant ON tombstones (tenant_id);

CREATE TABLE IF NOT EXISTS summaries (
    summary_id        TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL,
    session_id        TEXT NOT NULL,
    content           TEXT NOT NULL,
    source_memory_ids TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_tenant ON summaries (tenant_id);

CREATE TABLE IF NOT EXISTS retrieval_cache (
    cache_key   TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    query       TEXT NOT NULL,
    payload     TEXT NOT NULL,
    memory_ids  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS store_epoch (
    id    INTEGER PRIMARY KEY CHECK (id = 1),
    epoch INTEGER NOT NULL
);
INSERT OR IGNORE INTO store_epoch (id, epoch) VALUES (1, 0);

CREATE TABLE IF NOT EXISTS write_journal (
    journal_id  TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    event_id    TEXT NOT NULL,
    phase       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


class GuardedMemoryBackend(MemoryBackend):
    """Event-sourced, versioned, tenant-isolated memory with bounded, abstaining retrieval."""

    name = "guarded_memory"
    supports_ablations = (
        "abl_no_idempotency",
        "abl_no_supersession",
        "abl_no_tombstones",
        "abl_no_abstention",
        "abl_no_budget",
        "abl_no_tenant_filter",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        evidence_budget: int = 5,
        token_budget: int = 512,
        min_confidence: float = 0.35,
        abstain_on_conflict: bool = True,
        ablations: Iterable[str] = (),
        use_cache: bool = True,
        extractor: Extractor | None = None,
    ) -> None:
        self.path = Path(path)
        # Rule-based extraction is the default so every existing caller and every contract keeps
        # the behaviour it was written against; a real model is opt-in.
        self.extractor: Extractor = extractor or RuleBasedExtractor()
        self.evidence_budget = evidence_budget
        self.token_budget = token_budget
        self.min_confidence = min_confidence
        self.abstain_on_conflict = abstain_on_conflict
        self.ablations = frozenset(ablations)
        unknown = self.ablations - set(self.supports_ablations)
        if unknown:
            raise BackendError(f"unknown ablation(s) for {self.name}: {sorted(unknown)}")
        self.use_cache = use_cache
        self._connection: sqlite3.Connection | None = None
        self._log: EventLog | None = None
        self._index: LexicalIndex | None = None

    # -- guards -------------------------------------------------------------------------
    @property
    def idempotency_enabled(self) -> bool:
        return "abl_no_idempotency" not in self.ablations

    @property
    def supersession_enabled(self) -> bool:
        return "abl_no_supersession" not in self.ablations

    @property
    def tombstones_enabled(self) -> bool:
        return "abl_no_tombstones" not in self.ablations

    @property
    def abstention_enabled(self) -> bool:
        return "abl_no_abstention" not in self.ablations

    @property
    def budget_enabled(self) -> bool:
        return "abl_no_budget" not in self.ablations

    @property
    def tenant_filter_enabled(self) -> bool:
        return "abl_no_tenant_filter" not in self.ablations

    # -- lifecycle ----------------------------------------------------------------------
    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(MEMORY_SCHEMA)
        self._log = EventLog(self._connection)
        self._index = LexicalIndex(self._connection)
        self._connection.commit()
        self._recover_incomplete_writes()

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

    @property
    def index(self) -> LexicalIndex:
        if self._index is None:
            raise BackendError("backend is not initialized; call initialize() first")
        return self._index

    def checkpoint(self) -> None:
        self.connection.commit()

    def restart(self) -> None:
        """Clean restart: close the connection and reopen from durable state.

        Recovery runs on the way back up, so an interrupted write is repaired here.
        """
        self.close()
        self.initialize()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.commit()
            self._connection.close()
            self._connection = None
            self._log = None
            self._index = None

    # -- Section 10: write lifecycle ----------------------------------------------------
    def ingest_event(self, event: Event, *, abort_after: str | None = None) -> WriteResult:
        """Run the sixteen-step write lifecycle.

        ``abort_after`` is used only by the interrupted-write fault. When set to
        ``"log_append"`` the event log append and the journal intent are committed and the
        method then raises, leaving derived state behind the log — exactly the state a crash
        between the two would produce (DEVIATIONS V-003).
        """
        started = time.perf_counter()
        connection = self.connection

        # 1. Validate the event schema. Pydantic did the structural work at construction; this
        #    catches an event whose hash was tampered with after validation.
        if event.source_hash != event.compute_source_hash():
            return WriteResult(
                event_id=event.event_id,
                tenant_id=event.tenant_id,
                outcome=WriteOutcome.rejected,
                backend_name=self.name,
                notes="source_hash does not match content",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # 2/3. Source hash and duplicate-delivery detection happen inside the log append.
        append = self.log.append(event, detect_duplicates=self.idempotency_enabled)
        if not append.appended:
            connection.commit()
            return WriteResult(
                event_id=event.event_id,
                tenant_id=event.tenant_id,
                outcome=WriteOutcome.duplicate_ignored,
                backend_name=self.name,
                notes=f"duplicate detected on {append.detected_on}",
                latency_ms=(time.perf_counter() - started) * 1000,
                metadata={"duplicate_of": append.duplicate_of_event_id},
            )

        # 15 (first half). Recovery metadata: record the intent before touching derived state.
        journal_id = stable_id("wj", event.tenant_id, event.event_id)
        connection.execute(
            "INSERT OR REPLACE INTO write_journal (journal_id, tenant_id, event_id, phase, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (journal_id, event.tenant_id, event.event_id, "derived_pending", utcnow().isoformat()),
        )

        if abort_after == "log_append":
            # 4. The log append is durable; derived state is not yet written.
            connection.commit()
            raise InterruptedWrite(f"write interrupted after log append for event {event.event_id}")

        try:
            result = self._apply_derived(event)
            # 15 (second half). Clear the intent inside the same transaction as the derived work.
            connection.execute("DELETE FROM write_journal WHERE journal_id = ?", (journal_id,))
            # 13. One transaction covers the log append and every derived change.
            connection.commit()
        except Exception:
            connection.rollback()
            raise

        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    def _apply_derived(self, event: Event) -> WriteResult:
        """Steps 5-14: extraction, comparison, versioning, indexes."""
        if event.event_type is EventType.deletion_request:
            delete_result = self._apply_deletion(event)
            self._bump_epoch()
            return WriteResult(
                event_id=event.event_id,
                tenant_id=event.tenant_id,
                outcome=WriteOutcome.deleted,
                deleted_memory_ids=delete_result.deleted_memory_ids,
                backend_name=self.name,
                notes=f"tombstoned {delete_result.target}",
                metadata={
                    "purged_index_rows": delete_result.purged_index_rows,
                    "invalidated_summaries": delete_result.invalidated_summaries,
                    "evicted_cache_entries": delete_result.evicted_cache_entries,
                },
            )

        # 5. Extract candidate memories. *What* to remember is a separate decision from the
        # lifecycle that manages it, and lives behind the extractor interface so the two can be
        # varied independently.
        candidates = self.extractor.extract(event)
        if not candidates:
            return WriteResult(
                event_id=event.event_id,
                tenant_id=event.tenant_id,
                outcome=WriteOutcome.no_op,
                backend_name=self.name,
                notes="no extractable facts in event",
            )

        created: list[str] = []
        superseded: list[str] = []
        conflicted: list[str] = []
        outcome = WriteOutcome.created

        for candidate in candidates:
            canonical_key = candidate.canonical_key
            statement = candidate.statement
            value = candidate.normalized_value
            # 6. Every candidate must carry provenance; MemoryRecord enforces it too.
            if not event.event_id:
                raise BackendError("candidate memory without source provenance")

            # 7/8. Resolve the key and compare against active versions.
            actives = self._active_records_for_key(event.tenant_id, canonical_key)

            # A tombstoned key must not be resurrected by a later unrelated write.
            if self.tombstones_enabled and self._is_tombstoned(event.tenant_id, canonical_key):
                continue

            # 9. Duplicate, update, or contradiction?
            identical = [r for r in actives if normalize_value(_value_of(r.content)) == value]
            if identical:
                outcome = WriteOutcome.duplicate_ignored if not created else outcome
                continue

            if not actives:
                record = self._insert_record(
                    event, canonical_key, statement, version=1, supersedes=None
                )
                created.append(record.memory_id)
                continue

            newest = max(actives, key=lambda r: (r.valid_from, r.memory_id))
            if event.event_time > newest.valid_from:
                # 10/11/12. A newer fact creates a version and closes the previous interval.
                if self.supersession_enabled:
                    # Close *every* still-open record for this key, not just the newest. If an
                    # earlier contradiction left two open records, a later dated assertion is
                    # exactly the evidence that resolves it; closing only one would leave the
                    # key permanently conflicted and the store permanently abstaining.
                    for stale in actives:
                        if stale.valid_to is None:
                            self._close_record(stale, valid_to=event.event_time)
                            superseded.append(stale.memory_id)
                    record = self._insert_record(
                        event,
                        canonical_key,
                        statement,
                        version=newest.version + 1,
                        supersedes=newest.memory_id,
                    )
                else:
                    # Ablation: append a second active record and leave the old one open, which
                    # is what a store without supersession does.
                    record = self._insert_record(
                        event,
                        canonical_key,
                        statement,
                        version=newest.version + 1,
                        supersedes=None,
                    )
                created.append(record.memory_id)
                outcome = WriteOutcome.version_created
            elif event.event_time < newest.valid_from:
                # A *late-arriving older* fact is not a contradiction: its own timestamp settles
                # the order, it simply reached us out of sequence. Recording it as history keeps
                # the newer fact current and makes the as-of query answerable, which is what
                # replaying the correct order is supposed to achieve. Treating it as a conflict
                # instead left the key permanently unresolved, because idempotency then made the
                # corrective replay a no-op.
                closes_at = min(
                    (
                        candidate.valid_from
                        for candidate in self._records_for_key(event.tenant_id, canonical_key)
                        if candidate.valid_from > event.event_time
                    ),
                    default=newest.valid_from,
                )
                record = self._insert_record(
                    event,
                    canonical_key,
                    statement,
                    version=newest.version + 1,
                    supersedes=None,
                    status=MemoryStatus.superseded,
                    valid_to=closes_at,
                )
                # Superseded records are history, not current evidence, so they stay out of the
                # active index exactly as a normally-superseded version does.
                self.index.remove(record.memory_id, record.tenant_id)
                superseded.append(record.memory_id)
                if outcome is WriteOutcome.created:
                    outcome = WriteOutcome.version_created
            else:
                # Same instant, different values: nothing in the evidence establishes which is
                # current. Recording a conflict is the honest outcome; silently picking one is
                # the failure this contract exists to detect.
                record = self._insert_record(
                    event,
                    canonical_key,
                    statement,
                    version=newest.version + 1,
                    supersedes=None,
                    status=MemoryStatus.conflicted,
                    confidence=0.4,
                )
                self._mark_conflicted(newest)
                conflicted.extend([record.memory_id, newest.memory_id])
                created.append(record.memory_id)
                outcome = WriteOutcome.conflict_recorded

        # 14. Update derived indexes and summaries, then advance the epoch so any retrieval
        # cached against the previous state is no longer served.
        self._refresh_summary(event.tenant_id, event.session_id)
        self._bump_epoch()

        return WriteResult(
            event_id=event.event_id,
            tenant_id=event.tenant_id,
            outcome=outcome if created or superseded or conflicted else WriteOutcome.no_op,
            created_memory_ids=created,
            superseded_memory_ids=superseded,
            conflicted_memory_ids=sorted(set(conflicted)),
            backend_name=self.name,
        )

    # -- extraction ---------------------------------------------------------------------
    def _extract_candidates(self, event: Event) -> list[tuple[str, str, str]]:
        """Deprecated tuple view of the extractor, retained for callers that predate it."""
        return [
            (c.canonical_key, c.statement, c.normalized_value)
            for c in self.extractor.extract(event)
        ]

    def _active_records_for_key(self, tenant_id: str, canonical_key: str) -> list[MemoryRecord]:
        """Records for a key that are still asserting something: active or conflicted."""
        rows = self.connection.execute(
            "SELECT * FROM memories WHERE tenant_id = ? AND canonical_key = ? "
            "AND status IN (?, ?) ORDER BY memory_id",
            (
                tenant_id,
                canonical_key,
                MemoryStatus.active.value,
                MemoryStatus.conflicted.value,
            ),
        )
        return [_row_to_record(row) for row in rows]

    def _records_for_key(self, tenant_id: str, canonical_key: str) -> list[MemoryRecord]:
        """Every record for a key, whatever its status — used to place a late arrival's
        validity interval against the versions that already exist."""
        rows = self.connection.execute(
            "SELECT * FROM memories WHERE tenant_id = ? AND canonical_key = ? ORDER BY valid_from",
            (tenant_id, canonical_key),
        )
        return [_row_to_record(row) for row in rows]

    def _insert_record(
        self,
        event: Event,
        canonical_key: str,
        statement: str,
        *,
        version: int,
        supersedes: str | None,
        status: MemoryStatus = MemoryStatus.active,
        confidence: float = 1.0,
        valid_to: datetime | None = None,
    ) -> MemoryRecord:
        memory_id = stable_id("mem", event.tenant_id, canonical_key, version, statement)
        record = MemoryRecord(
            memory_id=memory_id,
            tenant_id=event.tenant_id,
            canonical_key=canonical_key,
            version=version,
            content=statement,
            status=status,
            valid_from=event.event_time,
            valid_to=valid_to,
            confidence=confidence,
            source_event_ids=[event.event_id],
            supersedes_memory_id=supersedes,
            metadata={"session_id": event.session_id, "event_type": event.event_type.value},
        )
        self._write_record(record)
        self.index.add(record.memory_id, record.tenant_id, record.content)
        return record

    def _write_record(self, record: MemoryRecord) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO memories (memory_id, tenant_id, canonical_key, version, "
            "content, status, valid_from, valid_to, confidence, source_event_ids, "
            "supersedes_memory_id, created_at, updated_at, embedding_model, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.memory_id,
                record.tenant_id,
                record.canonical_key,
                record.version,
                record.content,
                record.status.value,
                record.valid_from.isoformat(),
                record.valid_to.isoformat() if record.valid_to else None,
                record.confidence,
                json.dumps(record.source_event_ids),
                record.supersedes_memory_id,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.embedding_model,
                json.dumps(record.metadata, sort_keys=True),
            ),
        )

    def _close_record(self, record: MemoryRecord, *, valid_to: datetime) -> None:
        """Mark a record superseded and close its validity interval, preserving history."""
        self.connection.execute(
            "UPDATE memories SET status = ?, valid_to = ?, updated_at = ? WHERE memory_id = ?",
            (
                MemoryStatus.superseded.value,
                valid_to.isoformat(),
                utcnow().isoformat(),
                record.memory_id,
            ),
        )
        # A superseded record leaves the active index but stays queryable as history.
        self.index.remove(record.memory_id, record.tenant_id)

    def _mark_conflicted(self, record: MemoryRecord) -> None:
        self.connection.execute(
            "UPDATE memories SET status = ?, confidence = ?, updated_at = ? WHERE memory_id = ?",
            (MemoryStatus.conflicted.value, 0.4, utcnow().isoformat(), record.memory_id),
        )

    # -- deletion (DECISIONS D-008) -----------------------------------------------------
    def delete(self, identifier: str, tenant_id: str) -> DeleteResult:
        """Delete via the API.

        The deletion is first recorded as a ``deletion_request`` event in the log, so that it is
        part of the source of truth and survives an index rebuild. A deletion that lives only in
        derived state would come back the next time the store was rebuilt — which is exactly the
        resurrection the closure contract looks for.
        """
        started = time.perf_counter()
        request_event = Event(
            event_id=stable_id("del", tenant_id, identifier),
            tenant_id=tenant_id,
            session_id="__api__",
            event_type=EventType.deletion_request,
            event_time=utcnow(),
            content=f"Delete: {identifier}",
            metadata={"deletion_target": identifier, "origin": "api"},
        )
        self.log.append(request_event, detect_duplicates=self.idempotency_enabled)
        result = self._delete_target(identifier, tenant_id, reason="explicit_delete")
        self._bump_epoch()
        self.connection.commit()
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    def _apply_deletion(self, event: Event) -> DeleteResult:
        target = str(event.metadata.get("deletion_target", "")) or event.content
        return self._delete_target(target, event.tenant_id, reason="deletion_request")

    def _delete_target(self, identifier: str, tenant_id: str, *, reason: str) -> DeleteResult:
        """Delete across every derived surface in one transaction."""
        records = self._resolve_deletion_targets(identifier, tenant_id)
        deleted_ids: list[str] = []
        purged_rows = 0

        for record in records:
            self.connection.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
                (MemoryStatus.deleted.value, utcnow().isoformat(), record.memory_id),
            )
            deleted_ids.append(record.memory_id)
            if self.tombstones_enabled:
                # (b) purge the lexical index rows
                purged_rows += self.index.remove(record.memory_id, record.tenant_id)

        tombstone_ids: list[str] = []
        if self.tombstones_enabled:
            for record in records:
                tombstone_id = stable_id("tomb", tenant_id, record.canonical_key, record.memory_id)
                self.connection.execute(
                    "INSERT OR REPLACE INTO tombstones (tombstone_id, tenant_id, target, "
                    "canonical_key, content_hash, created_at, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        tombstone_id,
                        tenant_id,
                        record.memory_id,
                        record.canonical_key,
                        content_hash(record.content),
                        utcnow().isoformat(),
                        reason,
                    ),
                )
                tombstone_ids.append(tombstone_id)

        invalidated = 0
        evicted = 0
        if self.tombstones_enabled and deleted_ids:
            # (c) invalidate summaries derived from the deleted evidence
            invalidated = self._invalidate_summaries(tenant_id, deleted_ids)
            # (d) evict retrieval-cache entries that referenced them
            evicted = self._evict_cache(tenant_id, deleted_ids)

        return DeleteResult(
            tenant_id=tenant_id,
            target=identifier,
            deleted_memory_ids=deleted_ids,
            purged_index_rows=purged_rows,
            invalidated_summaries=invalidated,
            evicted_cache_entries=evicted,
            tombstone_ids=tombstone_ids,
            backend_name=self.name,
        )

    def _resolve_deletion_targets(self, identifier: str, tenant_id: str) -> list[MemoryRecord]:
        """Resolve a memory id, an event id, a canonical key, or free-text to records."""
        rows = self.connection.execute(
            "SELECT * FROM memories WHERE tenant_id = ? AND memory_id = ?",
            (tenant_id, identifier),
        ).fetchall()
        if rows:
            return [_row_to_record(row) for row in rows]

        rows = self.connection.execute(
            "SELECT * FROM memories WHERE tenant_id = ? AND canonical_key = ?",
            (tenant_id, identifier),
        ).fetchall()
        if rows:
            return [_row_to_record(row) for row in rows]

        # Event id: delete everything derived from that event.
        rows = self.connection.execute(
            "SELECT * FROM memories WHERE tenant_id = ? AND source_event_ids LIKE ?",
            (tenant_id, f'%"{identifier}"%'),
        ).fetchall()
        if rows:
            return [_row_to_record(row) for row in rows]

        # Free-text deletion request: match on the request's content terms, which is how a user
        # phrases "forget what I told you about X".
        request_terms = content_terms(identifier) - {"delete", "forget", "remove", "erase", "all"}
        if not request_terms:
            return []
        matched: list[MemoryRecord] = []
        for row in self.connection.execute(
            "SELECT * FROM memories WHERE tenant_id = ?", (tenant_id,)
        ):
            record = _row_to_record(row)
            if request_terms & content_terms(record.canonical_key):
                matched.append(record)
        return matched

    # -- cache coherence ----------------------------------------------------------------
    def _epoch(self) -> int:
        row = self.connection.execute("SELECT epoch FROM store_epoch WHERE id = 1").fetchone()
        return int(row["epoch"]) if row else 0

    def _bump_epoch(self) -> None:
        """Advance the store epoch so cached retrievals from before this write cannot be served.

        Without this, a query answered against an empty store stays answered that way for the
        rest of the run: the cache would turn an early abstention into a permanent one.
        """
        self.connection.execute("UPDATE store_epoch SET epoch = epoch + 1 WHERE id = 1")

    def _is_tombstoned(self, tenant_id: str, canonical_key: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM tombstones WHERE tenant_id = ? AND canonical_key = ? LIMIT 1",
            (tenant_id, canonical_key),
        ).fetchone()
        return row is not None

    # -- summaries and cache ------------------------------------------------------------
    def _refresh_summary(self, tenant_id: str, session_id: str) -> None:
        records = [
            _row_to_record(row)
            for row in self.connection.execute(
                "SELECT * FROM memories WHERE tenant_id = ? AND status = ? "
                "AND json_extract(metadata, '$.session_id') = ? ORDER BY memory_id",
                (tenant_id, MemoryStatus.active.value, session_id),
            )
        ]
        summary_id = stable_id("sum", tenant_id, session_id)
        if not records:
            self.connection.execute("DELETE FROM summaries WHERE summary_id = ?", (summary_id,))
            return
        content = " ".join(record.content for record in records)
        self.connection.execute(
            "INSERT OR REPLACE INTO summaries (summary_id, tenant_id, session_id, content, "
            "source_memory_ids, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                summary_id,
                tenant_id,
                session_id,
                content,
                json.dumps([record.memory_id for record in records]),
                utcnow().isoformat(),
            ),
        )

    def _invalidate_summaries(self, tenant_id: str, deleted_ids: list[str]) -> int:
        invalidated = 0
        for row in self.connection.execute(
            "SELECT * FROM summaries WHERE tenant_id = ?", (tenant_id,)
        ).fetchall():
            source_ids = set(json.loads(str(row["source_memory_ids"])))
            if source_ids & set(deleted_ids):
                self.connection.execute(
                    "DELETE FROM summaries WHERE summary_id = ?", (row["summary_id"],)
                )
                invalidated += 1
        # Rebuild the affected sessions from surviving active records.
        for session_id in {
            str(row["session_id"])
            for row in self.connection.execute(
                "SELECT DISTINCT json_extract(metadata, '$.session_id') AS session_id "
                "FROM memories WHERE tenant_id = ?",
                (tenant_id,),
            )
            if row["session_id"]
        }:
            self._refresh_summary(tenant_id, session_id)
        return invalidated

    def _evict_cache(self, tenant_id: str, deleted_ids: list[str]) -> int:
        evicted = 0
        for row in self.connection.execute(
            "SELECT cache_key, memory_ids FROM retrieval_cache WHERE tenant_id = ?", (tenant_id,)
        ).fetchall():
            if set(json.loads(str(row["memory_ids"]))) & set(deleted_ids):
                self.connection.execute(
                    "DELETE FROM retrieval_cache WHERE cache_key = ?", (row["cache_key"],)
                )
                evicted += 1
        return evicted

    def summary_texts(self, tenant_id: str | None = None) -> list[str]:
        """Summary contents, for the deletion-closure contract to probe directly."""
        if tenant_id is None:
            rows = self.connection.execute("SELECT content FROM summaries")
        else:
            rows = self.connection.execute(
                "SELECT content FROM summaries WHERE tenant_id = ?", (tenant_id,)
            )
        return [str(row["content"]) for row in rows]

    def cache_texts(self, tenant_id: str | None = None) -> list[str]:
        if tenant_id is None:
            rows = self.connection.execute("SELECT payload FROM retrieval_cache")
        else:
            rows = self.connection.execute(
                "SELECT payload FROM retrieval_cache WHERE tenant_id = ?", (tenant_id,)
            )
        return [str(row["payload"]) for row in rows]

    # -- Section 11: retrieval lifecycle ------------------------------------------------
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
        if not tenant_id:
            raise BackendError("retrieve() requires an explicit tenant_id")
        effective_tokens = token_budget if token_budget is not None else self.token_budget
        effective_count = evidence_budget if evidence_budget is not None else self.evidence_budget

        cache_key = content_hash(
            [
                self.name,
                self._epoch(),
                tenant_id,
                query,
                as_of_time,
                effective_tokens,
                effective_count,
            ]
        )
        if self.use_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                cached.retrieval_latency_ms = (time.perf_counter() - started) * 1000
                cached.metadata = {**cached.metadata, "cache_hit": True}
                return cached

        # 1/2/3. Tenant, status, and temporal filters, applied in SQL.
        candidates = self._candidate_records(tenant_id, as_of_time)

        # 4/5. Retrieve and rank against the derived index, then map back to records.
        search_tenant = tenant_id if self.tenant_filter_enabled else None
        ranked = self._rank(query, candidates, search_tenant, limit=max(effective_count * 4, 20))

        # 6. Remove duplicate evidence by normalized content.
        deduplicated: list[tuple[MemoryRecord, float]] = []
        seen_content: set[str] = set()
        for record, score in ranked:
            key = normalize_value(record.content)
            if key in seen_content:
                continue
            seen_content.add(key)
            deduplicated.append((record, score))

        # 7. Detect unresolved contradictions among the surviving candidates.
        conflicted = self._unresolved_conflicts(deduplicated)

        evidence: list[Evidence] = []
        injected_tokens = 0
        budget_exhausted = False
        for record, score in deduplicated:
            # 8. Evidence count limit.
            if self.budget_enabled and len(evidence) >= effective_count:
                budget_exhausted = True
                break
            tokens = count_tokens(record.content)
            # 9. Token budget.
            if self.budget_enabled and injected_tokens + tokens > effective_tokens:
                budget_exhausted = True
                break
            injected_tokens += tokens
            evidence.append(
                Evidence(
                    memory_id=record.memory_id,
                    # 10. Source identifiers travel with the evidence.
                    source_event_ids=record.source_event_ids,
                    content=record.content,
                    relevance_score=score,
                    confidence=record.confidence,
                    # 11. Temporal validity travels with the evidence.
                    valid_from=record.valid_from,
                    valid_to=record.valid_to,
                    retrieval_reason=self._reason_for(record, score, conflicted),
                    token_count=tokens,
                )
            )

        # 12. Abstain when evidence is missing, weak, deleted, or conflicted.
        abstained = False
        reason: AbstentionReason | None = None
        if self.abstention_enabled:
            if not evidence:
                abstained, reason = True, AbstentionReason.no_evidence
            elif conflicted and self.abstain_on_conflict:
                abstained, reason = True, AbstentionReason.conflicted_evidence
            elif max(item.confidence for item in evidence) < self.min_confidence:
                abstained, reason = True, AbstentionReason.weak_evidence

        result = RetrievalResult(
            query=query,
            tenant_id=tenant_id,
            evidence=evidence,
            abstained=abstained,
            abstention_reason=reason,
            retrieval_latency_ms=(time.perf_counter() - started) * 1000,
            injected_tokens=injected_tokens,
            backend_name=self.name,
            metadata={
                "candidates_considered": len(candidates),
                "budget_exhausted": budget_exhausted,
                "unresolved_conflicts": sorted(conflicted),
                "query_id": query_id,
                "as_of_time": as_of_time.isoformat() if as_of_time else None,
                "cache_hit": False,
            },
        )
        if self.use_cache:
            self._write_cache(cache_key, tenant_id, query, result)
        return result

    def _candidate_records(
        self, tenant_id: str, as_of_time: datetime | None
    ) -> dict[str, MemoryRecord]:
        if self.tenant_filter_enabled:
            rows = self.connection.execute(
                "SELECT * FROM memories WHERE tenant_id = ? AND status != ?",
                (tenant_id, MemoryStatus.deleted.value),
            )
        else:
            # Ablation only: the tenant predicate is dropped, which is what a store that filters
            # by prompt instead of by storage does.
            rows = self.connection.execute(
                "SELECT * FROM memories WHERE status != ?", (MemoryStatus.deleted.value,)
            )
        candidates: dict[str, MemoryRecord] = {}
        for row in rows:
            record = _row_to_record(row)
            if as_of_time is None:
                # A current query sees only open intervals: active and conflicted records.
                if record.status in (MemoryStatus.active, MemoryStatus.conflicted):
                    candidates[record.memory_id] = record
            elif record.is_valid_at(as_of_time):
                # A historical query sees whatever was valid then, including superseded records.
                candidates[record.memory_id] = record
        return candidates

    def _rank(
        self,
        query: str,
        candidates: dict[str, MemoryRecord],
        tenant_id: str | None,
        *,
        limit: int,
    ) -> list[tuple[MemoryRecord, float]]:
        ranked: list[tuple[MemoryRecord, float]] = []
        tenants = [tenant_id] if tenant_id else self.log.tenants()
        for tenant in tenants:
            for hit in self.index.search(query, tenant, limit=limit):
                record = candidates.get(hit.doc_id)
                if record is not None:
                    ranked.append((record, hit.score))
        # Records that are valid but no longer indexed (superseded history reached through an
        # as_of query) are scored directly so historical queries still work after an index purge.
        indexed = {record.memory_id for record, _ in ranked}
        if any(record.status is MemoryStatus.superseded for record in candidates.values()):
            from memcheck.backends.retrieval import _bm25

            missing = [
                (record.memory_id, record.content)
                for record in candidates.values()
                if record.memory_id not in indexed
            ]
            if missing:
                from memcheck.backends.retrieval import tokenize

                for hit in _bm25(tokenize(query), missing, limit=limit):
                    ranked.append((candidates[hit.doc_id], hit.score))
        ranked.sort(key=lambda item: (-item[1], item[0].memory_id))
        return ranked

    def _unresolved_conflicts(self, ranked: list[tuple[MemoryRecord, float]]) -> set[str]:
        """A key is unresolved when two open records assert different values at the same time."""
        by_key: dict[str, list[MemoryRecord]] = {}
        for record, _ in ranked:
            by_key.setdefault(record.canonical_key, []).append(record)
        conflicted: set[str] = set()
        for key, records in by_key.items():
            open_records = [r for r in records if r.valid_to is None]
            values = {normalize_value(_value_of(r.content)) for r in open_records}
            if len(values) <= 1:
                continue
            times = {r.valid_from for r in open_records}
            # Distinct timestamps let a reader resolve by recency; identical ones do not.
            if len(times) < len(open_records) or any(
                r.status is MemoryStatus.conflicted for r in open_records
            ):
                conflicted.add(key)
        return conflicted

    def _reason_for(self, record: MemoryRecord, score: float, conflicted: set[str]) -> str:
        parts = [f"bm25={score:.3f}", f"status={record.status.value}", f"v{record.version}"]
        if record.valid_to is not None:
            parts.append(f"historical_until={record.valid_to.isoformat()}")
        if record.canonical_key in conflicted:
            parts.append("unresolved_conflict")
        if record.supersedes_memory_id:
            parts.append(f"supersedes={record.supersedes_memory_id}")
        return "; ".join(parts)

    def _read_cache(self, cache_key: str) -> RetrievalResult | None:
        row = self.connection.execute(
            "SELECT payload FROM retrieval_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        return RetrievalResult.model_validate(json.loads(str(row["payload"])))

    def _write_cache(
        self, cache_key: str, tenant_id: str, query: str, result: RetrievalResult
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO retrieval_cache (cache_key, tenant_id, query, payload, "
            "memory_ids, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                cache_key,
                tenant_id,
                query,
                json.dumps(result.model_dump(mode="json"), default=str),
                json.dumps([item.memory_id for item in result.evidence]),
                utcnow().isoformat(),
            ),
        )
        self.connection.commit()

    # -- recovery and rebuild -----------------------------------------------------------
    def _recover_incomplete_writes(self) -> int:
        """Repair any write that was interrupted between the log append and the derived write."""
        pending = self.connection.execute(
            "SELECT journal_id, tenant_id, event_id FROM write_journal WHERE phase = ?",
            ("derived_pending",),
        ).fetchall()
        if not pending:
            return 0
        for row in pending:
            event = self.log.get(str(row["tenant_id"]), str(row["event_id"]))
            if event is not None:
                self._apply_derived(event)
            self.connection.execute(
                "DELETE FROM write_journal WHERE journal_id = ?", (row["journal_id"],)
            )
        self.connection.commit()
        return len(pending)

    def rebuild_indexes(self) -> int:
        """Rebuild every derived structure from the event log alone.

        This is the property the index-loss fault tests: the log must be sufficient.
        """
        self.index.create()
        self.index.clear()
        self.connection.execute("DELETE FROM memories")
        self.connection.execute("DELETE FROM summaries")
        self.connection.execute("DELETE FROM retrieval_cache")
        self.connection.execute("DELETE FROM write_journal")
        # Tombstones are themselves derived from deletion_request events, so they are replayed
        # rather than preserved; a deletion that is in the log stays applied after a rebuild.
        self.connection.execute("DELETE FROM tombstones")
        for event in self.log.replay():
            self._apply_derived(event)
        self.connection.commit()
        return self.index.document_count()

    # -- introspection ------------------------------------------------------------------
    def active_records(self, tenant_id: str | None = None) -> list[MemoryRecord]:
        if tenant_id is None:
            rows = self.connection.execute(
                "SELECT * FROM memories WHERE status = ? ORDER BY memory_id",
                (MemoryStatus.active.value,),
            )
        else:
            rows = self.connection.execute(
                "SELECT * FROM memories WHERE tenant_id = ? AND status = ? ORDER BY memory_id",
                (tenant_id, MemoryStatus.active.value),
            )
        return [_row_to_record(row) for row in rows]

    def all_records(self, tenant_id: str | None = None) -> list[MemoryRecord]:
        if tenant_id is None:
            rows = self.connection.execute("SELECT * FROM memories ORDER BY memory_id")
        else:
            rows = self.connection.execute(
                "SELECT * FROM memories WHERE tenant_id = ? ORDER BY memory_id", (tenant_id,)
            )
        return [_row_to_record(row) for row in rows]

    def get_statistics(self, *, tenant_id: str | None = None) -> StoreStatistics:
        def count(status: str) -> int:
            if tenant_id is None:
                row = self.connection.execute(
                    "SELECT COUNT(*) AS n FROM memories WHERE status = ?", (status,)
                ).fetchone()
            else:
                row = self.connection.execute(
                    "SELECT COUNT(*) AS n FROM memories WHERE tenant_id = ? AND status = ?",
                    (tenant_id, status),
                ).fetchone()
            return int(row["n"])

        total_row = (
            self.connection.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
            if tenant_id is None
            else self.connection.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
        )
        events = self.log.count(tenant_id)
        duplicates = self.log.duplicate_count(tenant_id)
        sessions = self.log.sessions_ingested(tenant_id)
        database_size = self.path.stat().st_size if self.path.exists() else 0
        return StoreStatistics(
            backend_name=self.name,
            tenant_id=tenant_id,
            database_size_bytes=database_size,
            index_size_bytes=self.index.size_bytes(),
            event_count=events,
            memory_record_count=int(total_row["n"]),
            active_memory_count=count(MemoryStatus.active.value),
            superseded_record_count=count(MemoryStatus.superseded.value),
            deleted_record_count=count(MemoryStatus.deleted.value),
            conflicted_record_count=count(MemoryStatus.conflicted.value),
            duplicate_ratio=duplicates / max(1, events + duplicates),
            sessions_ingested=sessions,
            storage_growth_per_session=database_size / max(1, sessions),
            metadata={"ablations": sorted(self.ablations)},
        )

    def export_state(self) -> dict[str, Any]:
        return {
            "events": [event.model_dump(mode="json") for event in self.log.replay()],
            "memories": [record.model_dump(mode="json") for record in self.all_records()],
            "ablations": sorted(self.ablations),
        }

    def import_state(self, state: dict[str, Any]) -> None:
        self.connection.execute("DELETE FROM memories")
        self.index.clear()
        for payload in state.get("events", []):
            self.log.append(Event.model_validate(payload), detect_duplicates=True)
        self.connection.commit()
        self.rebuild_indexes()


class InterruptedWrite(RuntimeError):
    """Raised by the interrupted-write fault after the log append has been committed."""


def _value_of(statement: str) -> str:
    value = extract_value(statement)
    return value if value is not None else statement


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        memory_id=str(row["memory_id"]),
        tenant_id=str(row["tenant_id"]),
        canonical_key=str(row["canonical_key"]),
        version=int(row["version"]),
        content=str(row["content"]),
        status=MemoryStatus(str(row["status"])),
        valid_from=datetime.fromisoformat(str(row["valid_from"])),
        valid_to=datetime.fromisoformat(str(row["valid_to"])) if row["valid_to"] else None,
        confidence=float(row["confidence"]),
        source_event_ids=json.loads(str(row["source_event_ids"])),
        supersedes_memory_id=row["supersedes_memory_id"],
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        embedding_model=row["embedding_model"],
        metadata=json.loads(str(row["metadata"])),
    )
