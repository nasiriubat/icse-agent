"""Exact-match response cache.

A cached response is reused only when the *entire* request is identical: model, system prompt,
messages, temperature, max tokens, tools, response schema, and structured input (Section 7).
Nothing derived from a secret ever enters the key or the stored payload.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memcheck.models import ChatResponse, content_hash
from memcheck.providers.base import ChatRequest

_SCHEMA = """
CREATE TABLE IF NOT EXISTS response_cache (
    request_key TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def request_key(provider: str, request: ChatRequest) -> str:
    payload = request.model_dump(mode="json")
    if payload.get("attempt", 0) == 0:
        payload.pop("attempt", None)
    return content_hash({"provider": provider, "request": payload})


def stamp_attempt(request: ChatRequest, attempt: int) -> ChatRequest:
    if attempt == 0 or request.attempt == attempt:
        return request
    return request.model_copy(update={"attempt": attempt})


class ResponseCache:
    """SQLite-backed cache keyed on the full request hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def get(self, key: str) -> ChatResponse | None:
        row = self._connection.execute(
            "SELECT payload FROM response_cache WHERE request_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        response = ChatResponse.model_validate(json.loads(row[0]))
        response.call.cached = True
        return response

    def put(self, key: str, provider: str, model: str, response: ChatResponse) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO response_cache (request_key, provider, model, payload) "
            "VALUES (?, ?, ?, ?)",
            (key, provider, model, json.dumps(response.model_dump(mode="json"), default=str)),
        )
        self._connection.commit()

    def size(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM response_cache").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._connection.close()
