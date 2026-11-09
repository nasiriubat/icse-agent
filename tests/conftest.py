"""Shared fixtures. Everything here is local and mock-only: no test may touch a network."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memcheck.backends import build_backend
from memcheck.backends.base import MemoryBackend
from memcheck.config import CORE_BACKENDS, RetrievalConfig
from memcheck.models import Event, EventType

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _block_paid_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may authorize a paid run or see a credential, whatever the developer's shell has."""
    for variable in (
        "ALLOW_PAID_RUNS",
        "MAX_EXPERIMENT_COST_USD",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)
    assert not os.environ.get("ALLOW_PAID_RUNS")


@pytest.fixture
def retrieval() -> RetrievalConfig:
    return RetrievalConfig(evidence_budget=5, token_budget=256)


@pytest.fixture
def make_event() -> object:
    def factory(
        index: int,
        content: str,
        *,
        tenant_id: str = "tenant_a",
        session_id: str = "session_01",
        event_type: EventType = EventType.user_message,
        days: int | None = None,
        metadata: dict[str, object] | None = None,
        event_id: str | None = None,
    ) -> Event:
        offset = index if days is None else days
        return Event(
            event_id=event_id or f"evt_{index:04d}",
            tenant_id=tenant_id,
            session_id=session_id,
            event_type=event_type,
            event_time=EPOCH + timedelta(days=offset),
            ingest_time=EPOCH + timedelta(days=offset),
            content=content,
            metadata=dict(metadata or {}),
        )

    return factory


@pytest.fixture
def guarded(tmp_path: Path, retrieval: RetrievalConfig) -> Iterator[MemoryBackend]:
    backend = build_backend("guarded_memory", tmp_path / "guarded.db", retrieval)
    yield backend
    backend.close()


@pytest.fixture(params=list(CORE_BACKENDS))
def any_backend(
    request: pytest.FixtureRequest, tmp_path: Path, retrieval: RetrievalConfig
) -> Iterator[MemoryBackend]:
    """Every core backend, so interface-level tests cover all four."""
    backend = build_backend(request.param, tmp_path / f"{request.param}.db", retrieval)
    yield backend
    backend.close()


@pytest.fixture
def backend_factory(tmp_path: Path, retrieval: RetrievalConfig) -> object:
    def factory(name: str) -> object:
        def build(suffix: str) -> MemoryBackend:
            return build_backend(name, tmp_path / f"{name}_{suffix}.db", retrieval)

        return build

    return factory
