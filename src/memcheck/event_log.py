"""Immutable, idempotent event log — the source of truth for the guarded backend.

Two properties matter and are enforced here rather than by convention:

1. **Append-only.** There is no update or delete path. A deletion request is itself an event.
2. **Idempotent.** Re-delivering an event is a no-op, detected on the event id *and* on the
   content hash, so a duplicate that arrives under a fresh id is still recognised. This is a
   *guard* in the append path, not a schema constraint, so it can be ablated: with it disabled
   the log stores the duplicate and the resulting amplification becomes measurable rather than
   surfacing as a crash.

The log owns a :class:`sqlite3.Connection` that the guarded backend shares, so a single
transaction can cover the log append and every derived write (DECISIONS D-002).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from memcheck.models import Event, EventType, StrictModel

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    sequence     INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT NOT NULL,
    tenant_id    TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    event_time   TEXT NOT NULL,
    ingest_time  TEXT NOT NULL,
    content      TEXT NOT NULL,
    source_uri   TEXT,
    source_hash  TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
-- Idempotency is a *guard*, not a schema constraint. A UNIQUE (tenant_id, event_id) index here
-- would enforce it unconditionally, which makes the guard impossible to ablate: the store would
-- crash rather than exhibit the duplicate amplification the ablation exists to measure. The
-- append path decides whether a duplicate is ignored; the log only stores what it is given.
CREATE INDEX IF NOT EXISTS idx_events_tenant_event ON events (tenant_id, event_id);
CREATE INDEX IF NOT EXISTS idx_events_tenant_hash ON events (tenant_id, source_hash);
CREATE INDEX IF NOT EXISTS idx_events_tenant_time ON events (tenant_id, event_time);
CREATE INDEX IF NOT EXISTS idx_events_session ON events (tenant_id, session_id);

CREATE TABLE IF NOT EXISTS duplicate_deliveries (
    sequence     INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id    TEXT NOT NULL,
    event_id     TEXT NOT NULL,
    source_hash  TEXT NOT NULL,
    detected_on  TEXT NOT NULL,
    observed_at  TEXT NOT NULL
);
"""


class AppendResult(StrictModel):
    """Outcome of an append. ``appended=False`` means the event was already present."""

    event_id: str
    tenant_id: str
    appended: bool
    sequence: int | None = None
    duplicate_of_event_id: str | None = None
    detected_on: str | None = None  # "event_id" | "source_hash"


class EventLog:
    """Append-only event store over a shared SQLite connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)

    # -- writes -------------------------------------------------------------------------
    def append(self, event: Event, *, detect_duplicates: bool = True) -> AppendResult:
        """Append an event. Duplicate deliveries are recorded and ignored.

        The caller controls the surrounding transaction; this method never commits, so the log
        append and the derived writes it triggers either both land or neither does.
        """
        if detect_duplicates:
            duplicate = self._find_duplicate(event)
            if duplicate is not None:
                existing_id, detected_on = duplicate
                self._connection.execute(
                    "INSERT INTO duplicate_deliveries "
                    "(tenant_id, event_id, source_hash, detected_on, observed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        event.tenant_id,
                        event.event_id,
                        event.source_hash,
                        detected_on,
                        event.ingest_time.isoformat(),
                    ),
                )
                return AppendResult(
                    event_id=event.event_id,
                    tenant_id=event.tenant_id,
                    appended=False,
                    duplicate_of_event_id=existing_id,
                    detected_on=detected_on,
                )

        cursor = self._connection.execute(
            "INSERT INTO events (event_id, tenant_id, session_id, event_type, event_time, "
            "ingest_time, content, source_uri, source_hash, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.tenant_id,
                event.session_id,
                event.event_type.value,
                event.event_time.isoformat(),
                event.ingest_time.isoformat(),
                event.content,
                event.source_uri,
                event.source_hash,
                json.dumps(event.metadata, sort_keys=True),
            ),
        )
        return AppendResult(
            event_id=event.event_id,
            tenant_id=event.tenant_id,
            appended=True,
            sequence=int(cursor.lastrowid or 0),
        )

    def _find_duplicate(self, event: Event) -> tuple[str, str] | None:
        row = self._connection.execute(
            "SELECT event_id FROM events WHERE tenant_id = ? AND event_id = ? LIMIT 1",
            (event.tenant_id, event.event_id),
        ).fetchone()
        if row is not None:
            return (str(row["event_id"]), "event_id")
        row = self._connection.execute(
            "SELECT event_id FROM events WHERE tenant_id = ? AND source_hash = ? LIMIT 1",
            (event.tenant_id, event.source_hash),
        ).fetchone()
        if row is not None:
            return (str(row["event_id"]), "source_hash")
        return None

    # -- reads --------------------------------------------------------------------------
    def replay(
        self,
        *,
        tenant_id: str | None = None,
        until: datetime | None = None,
        session_id: str | None = None,
    ) -> Iterator[Event]:
        """Yield events in append order. This is what index rebuild reads."""
        query = "SELECT * FROM events"
        clauses: list[str] = []
        parameters: list[Any] = []
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            parameters.append(tenant_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            parameters.append(session_id)
        if until is not None:
            clauses.append("event_time <= ?")
            parameters.append(until.isoformat())
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY sequence ASC"
        for row in self._connection.execute(query, parameters):
            yield _row_to_event(row)

    def get(self, tenant_id: str, event_id: str) -> Event | None:
        # Under the no-idempotency ablation the same id can appear more than once; the first
        # delivery is the one the write journal refers to.
        row = self._connection.execute(
            "SELECT * FROM events WHERE tenant_id = ? AND event_id = ? ORDER BY sequence LIMIT 1",
            (tenant_id, event_id),
        ).fetchone()
        return _row_to_event(row) if row is not None else None

    def count(self, tenant_id: str | None = None) -> int:
        if tenant_id is None:
            row = self._connection.execute("SELECT COUNT(*) AS n FROM events").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM events WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
        return int(row["n"])

    def duplicate_count(self, tenant_id: str | None = None) -> int:
        if tenant_id is None:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM duplicate_deliveries"
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM duplicate_deliveries WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
        return int(row["n"])

    def tenants(self) -> list[str]:
        return [
            str(row["tenant_id"])
            for row in self._connection.execute(
                "SELECT DISTINCT tenant_id FROM events ORDER BY tenant_id"
            )
        ]

    def sessions_ingested(self, tenant_id: str | None = None) -> int:
        if tenant_id is None:
            row = self._connection.execute(
                "SELECT COUNT(DISTINCT session_id) AS n FROM events"
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(DISTINCT session_id) AS n FROM events WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        return int(row["n"])


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        event_id=str(row["event_id"]),
        tenant_id=str(row["tenant_id"]),
        session_id=str(row["session_id"]),
        event_type=EventType(str(row["event_type"])),
        event_time=datetime.fromisoformat(str(row["event_time"])),
        ingest_time=datetime.fromisoformat(str(row["ingest_time"])),
        content=str(row["content"]),
        source_uri=row["source_uri"],
        source_hash=str(row["source_hash"]),
        metadata=json.loads(str(row["metadata"])),
    )
