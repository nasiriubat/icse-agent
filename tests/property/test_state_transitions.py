"""Property-based tests over sequences of memory state transitions.

Hypothesis generates the operation sequences; the invariants below must hold for *every* such
sequence, not merely for the ones we thought to write down. Any counterexample Hypothesis finds
should be pinned as a deterministic regression fixture in ``tests/fixtures``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from memcheck.backends import build_backend
from memcheck.config import RetrievalConfig
from memcheck.models import Event, EventType, MemoryStatus
from memcheck.providers.mock import normalize_value

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
ATTRIBUTES = ["deployment target", "primary database", "cache layer"]
VALUES = ["postgres", "kubernetes", "redis", "nomad", "mysql"]

pytestmark = pytest.mark.property

SLOW = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def operation_sequences(draw: st.DrawFn) -> list[tuple[str, str, str, int]]:
    """Sequences of (kind, attribute, value, day) operations."""
    length = draw(st.integers(min_value=1, max_value=12))
    operations: list[tuple[str, str, str, int]] = []
    day = 0
    for _ in range(length):
        kind = draw(st.sampled_from(["write", "write", "write", "duplicate", "delete"]))
        attribute = draw(st.sampled_from(ATTRIBUTES))
        value = draw(st.sampled_from(VALUES))
        day += draw(st.integers(min_value=1, max_value=3))
        operations.append((kind, attribute, value, day))
    return operations


def _apply(backend, operations: list[tuple[str, str, str, int]]) -> None:
    for index, (kind, attribute, value, day) in enumerate(operations):
        if kind == "delete":
            backend.ingest_event(
                Event(
                    event_id=f"op_{index:03d}",
                    tenant_id="t",
                    session_id="s",
                    event_type=EventType.deletion_request,
                    event_time=EPOCH + timedelta(days=day),
                    content=f"Delete everything about the {attribute}.",
                    metadata={"deletion_target": attribute},
                )
            )
            continue
        event = Event(
            event_id=f"op_{index:03d}",
            tenant_id="t",
            session_id="s",
            event_type=EventType.user_message,
            event_time=EPOCH + timedelta(days=day),
            content=f"The {attribute} is {value}.",
        )
        backend.ingest_event(event)
        if kind == "duplicate":
            backend.ingest_event(event)


@given(operations=operation_sequences())
@SLOW
def test_at_most_one_open_active_record_per_key(tmp_path_factory, operations) -> None:
    """However facts are written, updated, duplicated, or deleted, a key never ends up with
    two open active records: that state is what makes a store answer from a stale value."""
    directory = tmp_path_factory.mktemp("prop")
    backend = build_backend("guarded_memory", directory / "g.db", RetrievalConfig())
    try:
        _apply(backend, operations)
        open_active = [
            record
            for record in backend.all_records()
            if record.status is MemoryStatus.active and record.valid_to is None
        ]
        by_key: dict[str, int] = {}
        for record in open_active:
            by_key[record.canonical_key] = by_key.get(record.canonical_key, 0) + 1
        assert all(count == 1 for count in by_key.values()), by_key
    finally:
        backend.close()


@given(operations=operation_sequences())
@SLOW
def test_deleted_values_never_appear_in_current_evidence(tmp_path_factory, operations) -> None:
    directory = tmp_path_factory.mktemp("prop")
    backend = build_backend("guarded_memory", directory / "g.db", RetrievalConfig())
    try:
        _apply(backend, operations)
        deleted_keys = {attribute for kind, attribute, _, _ in operations if kind == "delete"}
        for attribute in deleted_keys:
            # Only assert for keys that were not re-asserted after their deletion.
            deletion_day = max(
                day for kind, attr, _, day in operations if kind == "delete" and attr == attribute
            )
            rewritten = any(
                kind != "delete" and attr == attribute and day > deletion_day
                for kind, attr, _, day in operations
            )
            if rewritten:
                continue
            result = backend.retrieve(f"What is the {attribute}?", "t")
            assert all(attribute not in item.content for item in result.evidence)
    finally:
        backend.close()


@given(operations=operation_sequences())
@SLOW
def test_restart_preserves_the_canonical_projection(tmp_path_factory, operations) -> None:
    from memcheck.contracts.restart_equivalence import canonical_state, divergence

    directory = tmp_path_factory.mktemp("prop")
    backend = build_backend("guarded_memory", directory / "g.db", RetrievalConfig())
    try:
        _apply(backend, operations)
        backend.checkpoint()
        before = canonical_state(backend, evidence_budget=5, token_budget=256)
        backend.restart()
        after = canonical_state(backend, evidence_budget=5, token_budget=256)
        assert divergence(before, after) == []
    finally:
        backend.close()


@given(operations=operation_sequences())
@SLOW
def test_rebuild_from_the_log_reproduces_retrieval(tmp_path_factory, operations) -> None:
    """The event log must be sufficient to reconstruct every derived structure."""
    from memcheck.contracts.restart_equivalence import canonical_state, divergence

    directory = tmp_path_factory.mktemp("prop")
    backend = build_backend("guarded_memory", directory / "g.db", RetrievalConfig())
    try:
        _apply(backend, operations)
        backend.checkpoint()
        before = canonical_state(backend, evidence_budget=5, token_budget=256)
        backend.rebuild_indexes()
        after = canonical_state(backend, evidence_budget=5, token_budget=256)
        assert divergence(before, after) == []
    finally:
        backend.close()


@given(
    operations=operation_sequences(),
    evidence_budget=st.integers(min_value=1, max_value=6),
    token_budget=st.integers(min_value=8, max_value=256),
)
@SLOW
def test_budgets_are_never_exceeded(
    tmp_path_factory, operations, evidence_budget, token_budget
) -> None:
    directory = tmp_path_factory.mktemp("prop")
    backend = build_backend("guarded_memory", directory / "g.db", RetrievalConfig())
    try:
        _apply(backend, operations)
        for attribute in ATTRIBUTES:
            result = backend.retrieve(
                f"What is the {attribute}?",
                "t",
                evidence_budget=evidence_budget,
                token_budget=token_budget,
            )
            assert len(result.evidence) <= evidence_budget
            assert result.injected_tokens <= token_budget
    finally:
        backend.close()


@given(operations=operation_sequences())
@SLOW
def test_duplicate_delivery_never_amplifies(tmp_path_factory, operations) -> None:
    directory = tmp_path_factory.mktemp("prop")
    once = build_backend("guarded_memory", directory / "once.db", RetrievalConfig())
    twice = build_backend("guarded_memory", directory / "twice.db", RetrievalConfig())
    try:
        _apply(once, operations)
        _apply(twice, operations)
        _apply(twice, operations)  # re-deliver the identical stream
        identity = lambda backend: sorted(  # noqa: E731
            (record.canonical_key, normalize_value(record.content))
            for record in backend.active_records()
        )
        assert len(identity(twice)) <= len(identity(once))
    finally:
        once.close()
        twice.close()


def test_regression_fixtures_are_replayable() -> None:
    """Every discovered counterexample is pinned here as a deterministic fixture."""
    from tests.fixtures.regressions import REGRESSIONS

    for name, operations in REGRESSIONS.items():
        assert isinstance(operations, list), name
        assert operations, f"{name} has no operations"


@pytest.mark.parametrize("name", ["contradiction_then_later_update", "delete_then_rewrite"])
def test_pinned_regressions(tmp_path, name) -> None:
    from tests.fixtures.regressions import REGRESSIONS

    backend = build_backend("guarded_memory", tmp_path / f"{name}.db", RetrievalConfig())
    try:
        _apply(backend, REGRESSIONS[name])
        open_active = [
            record
            for record in backend.all_records()
            if record.status is MemoryStatus.active and record.valid_to is None
        ]
        keys = [record.canonical_key for record in open_active]
        assert len(keys) == len(set(keys))
    finally:
        backend.close()


def _unused(_: Path) -> None:  # pragma: no cover
    return None
