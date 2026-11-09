"""Derived retrieval indexes.

The index is *derived* state: it lives in its own tables, it can be dropped or corrupted
(that is the index-loss fault), and it can be rebuilt from the event log alone. Nothing in the
retrieval path may read from the index without a tenant predicate (DECISIONS D-009).

Ranking is a pure-Python BM25 so that scores are identical on every machine regardless of how
the local SQLite was compiled. FTS5, when present, is used only as a fast candidate filter, and
the ranker identity is recorded in the run manifest so scores are never compared across rankers.
"""

from __future__ import annotations

import math
import random
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

from memcheck.utilities.environment import sqlite_has_fts5
from memcheck.utilities.tokens import STOPWORDS

RANKER_IDENTITY = "bm25_py_v1"

# Same tokenizer shape as the reader's (memcheck.providers.mock): a dot joins only when followed
# by an alphanumeric, so sentence-final punctuation never fuses onto a term.
_TOKEN = re.compile(r"[a-z0-9](?:[a-z0-9_\-/]|\.(?=[a-z0-9]))*")
_BM25_K1 = 1.5
_BM25_B = 0.75

SCHEMA = """
CREATE TABLE IF NOT EXISTS index_docs (
    doc_id      TEXT NOT NULL,
    tenant_id   TEXT NOT NULL,
    content     TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'memory',
    PRIMARY KEY (tenant_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_index_docs_tenant ON index_docs (tenant_id);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS index_fts USING fts5(doc_id, tenant_id, content);
"""


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens excluding function words — what a query is actually asking about."""
    return [token for token in tokenize(text) if token not in STOPWORDS]


@dataclass(frozen=True)
class ScoredDocument:
    doc_id: str
    score: float
    content: str


class LexicalIndex:
    """BM25 index over derived documents, scoped by tenant."""

    def __init__(self, connection: sqlite3.Connection, *, use_fts5: bool | None = None) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self.use_fts5 = sqlite_has_fts5() if use_fts5 is None else use_fts5
        self.create()

    # -- lifecycle ----------------------------------------------------------------------
    def create(self) -> None:
        self._connection.executescript(SCHEMA)
        if self.use_fts5:
            self._connection.executescript(FTS_SCHEMA)

    def drop(self) -> None:
        """Destroy the derived index entirely. Used by the index-loss fault."""
        self._connection.execute("DROP TABLE IF EXISTS index_docs")
        if self.use_fts5:
            self._connection.execute("DROP TABLE IF EXISTS index_fts")

    def clear(self) -> None:
        self._connection.execute("DELETE FROM index_docs")
        if self.use_fts5:
            self._connection.execute("DELETE FROM index_fts")

    def corrupt(self, rng: random.Random, fraction: float = 0.5) -> int:
        """Delete a deterministic random subset of index rows without touching the log."""
        doc_ids = [
            (str(row["tenant_id"]), str(row["doc_id"]))
            for row in self._connection.execute(
                "SELECT tenant_id, doc_id FROM index_docs ORDER BY tenant_id, doc_id"
            )
        ]
        if not doc_ids:
            return 0
        count = max(1, int(len(doc_ids) * fraction))
        victims = rng.sample(doc_ids, count)
        for tenant_id, doc_id in victims:
            self._connection.execute(
                "DELETE FROM index_docs WHERE tenant_id = ? AND doc_id = ?", (tenant_id, doc_id)
            )
            if self.use_fts5:
                self._connection.execute(
                    "DELETE FROM index_fts WHERE tenant_id = ? AND doc_id = ?", (tenant_id, doc_id)
                )
        return len(victims)

    # -- writes -------------------------------------------------------------------------
    def add(self, doc_id: str, tenant_id: str, content: str, *, kind: str = "memory") -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO index_docs (doc_id, tenant_id, content, kind) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, tenant_id, content, kind),
        )
        if self.use_fts5:
            self._connection.execute(
                "DELETE FROM index_fts WHERE tenant_id = ? AND doc_id = ?", (tenant_id, doc_id)
            )
            self._connection.execute(
                "INSERT INTO index_fts (doc_id, tenant_id, content) VALUES (?, ?, ?)",
                (doc_id, tenant_id, content),
            )

    def remove(self, doc_id: str, tenant_id: str) -> int:
        cursor = self._connection.execute(
            "DELETE FROM index_docs WHERE tenant_id = ? AND doc_id = ?", (tenant_id, doc_id)
        )
        removed = cursor.rowcount if cursor.rowcount > 0 else 0
        if self.use_fts5:
            self._connection.execute(
                "DELETE FROM index_fts WHERE tenant_id = ? AND doc_id = ?", (tenant_id, doc_id)
            )
        return removed

    # -- reads --------------------------------------------------------------------------
    def search(self, query: str, tenant_id: str, *, limit: int = 20) -> list[ScoredDocument]:
        """BM25 search restricted to one tenant. A missing tenant is a programming error."""
        if not tenant_id:
            raise ValueError("retrieval requires an explicit tenant_id; refusing to search all")
        # Score on content words only. Without this a document sharing nothing but "the" and
        # "is" with the query earns a non-zero BM25 score and is returned as evidence, which
        # then makes an empty-handed store look as though it had found something.
        query_terms = content_tokens(query)
        if not query_terms:
            return []
        documents = [
            (str(row["doc_id"]), str(row["content"]))
            for row in self._connection.execute(
                "SELECT doc_id, content FROM index_docs WHERE tenant_id = ? ORDER BY doc_id",
                (tenant_id,),
            )
        ]
        if not documents:
            return []
        return _bm25(query_terms, documents, limit=limit)

    def contains_text(self, needle: str, tenant_id: str | None = None) -> list[str]:
        """Substring probe used by the deletion-closure contract to inspect the raw index."""
        if tenant_id is None:
            rows = self._connection.execute("SELECT doc_id, content FROM index_docs")
        else:
            rows = self._connection.execute(
                "SELECT doc_id, content FROM index_docs WHERE tenant_id = ?", (tenant_id,)
            )
        lowered = needle.lower()
        return [str(row["doc_id"]) for row in rows if lowered in str(row["content"]).lower()]

    def document_count(self, tenant_id: str | None = None) -> int:
        if tenant_id is None:
            row = self._connection.execute("SELECT COUNT(*) AS n FROM index_docs").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM index_docs WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
        return int(row["n"])

    def size_bytes(self) -> int:
        """Approximate on-disk size of the derived index."""
        row = self._connection.execute(
            "SELECT COALESCE(SUM(LENGTH(content) + LENGTH(doc_id) + LENGTH(tenant_id)), 0) AS n "
            "FROM index_docs"
        ).fetchone()
        base = int(row["n"])
        # An FTS5 index roughly doubles the stored bytes; count it when it is in use so the
        # storage-growth metric reflects what the backend actually maintains.
        return base * 2 if self.use_fts5 else base


def _bm25(
    query_terms: list[str], documents: list[tuple[str, str]], *, limit: int
) -> list[ScoredDocument]:
    tokenized = [(doc_id, content, tokenize(content)) for doc_id, content in documents]
    total_documents = len(tokenized)
    average_length = sum(len(tokens) for _, _, tokens in tokenized) / total_documents or 1.0

    document_frequency: Counter[str] = Counter()
    for _, _, tokens in tokenized:
        for term in set(tokens):
            document_frequency[term] += 1

    scored: list[ScoredDocument] = []
    for doc_id, content, tokens in tokenized:
        counts = Counter(tokens)
        length = len(tokens) or 1
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            n_q = document_frequency[term]
            idf = math.log(1 + (total_documents - n_q + 0.5) / (n_q + 0.5))
            numerator = frequency * (_BM25_K1 + 1)
            denominator = frequency + _BM25_K1 * (1 - _BM25_B + _BM25_B * length / average_length)
            score += idf * numerator / denominator
        if score > 0:
            scored.append(ScoredDocument(doc_id=doc_id, score=score, content=content))
    # Ties break on doc_id so the ordering is total and reproducible.
    scored.sort(key=lambda item: (-item.score, item.doc_id))
    return scored[:limit]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return numerator / (left_norm * right_norm)
