"""Deterministic regression fixtures.

Every failure discovered during development — whether by a property test, a contract, or a run —
is pinned here as an exact operation sequence so it is replayed on every future test run. The
sequences use the ``(kind, attribute, value, day)`` shape that
``tests/property/test_state_transitions.py`` applies.
"""

from __future__ import annotations

REGRESSIONS: dict[str, list[tuple[str, str, str, int]]] = {
    # Found during bring-up: an early contradiction left two open records for one key, and
    # supersession only closed the newest, so the key stayed conflicted forever and the store
    # abstained on it for the rest of the run. A later dated assertion must resolve it.
    "contradiction_then_later_update": [
        ("write", "deployment target", "postgres", 1),
        ("write", "deployment target", "kubernetes", 1),
        ("write", "deployment target", "nomad", 5),
    ],
    # Found during bring-up: an API-level delete lived only in derived state, so an index
    # rebuild resurrected the deleted fact. Deletions are now recorded in the event log.
    "delete_then_rewrite": [
        ("write", "primary database", "postgres", 1),
        ("delete", "primary database", "", 2),
        ("write", "primary database", "mysql", 3),
    ],
    # Found during bring-up: a duplicate delivered under a fresh id was stored twice because the
    # tokenizer fused the sentence-final period onto the value, making the two contents differ.
    "duplicate_under_fresh_id": [
        ("write", "cache layer", "redis", 1),
        ("duplicate", "cache layer", "redis", 2),
    ],
}
