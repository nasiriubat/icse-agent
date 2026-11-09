"""The v4 synthetic workload: no marker values in the stream, temporal gold from the last change.

v2 and v3 are pinned byte for byte in test_synthetic_v3.py; these tests pin what v4 fixes
(DEVIATIONS V-024).
"""

from __future__ import annotations

import pytest

from memcheck.models import EventType
from memcheck.workloads.synthetic import CONTRADICTION_MARKER, SyntheticLongitudinalWorkload

SWEEP_PARAMS = {
    "facts_per_session": 4,
    "distractors_per_session": 3,
    "update_fraction": 0.35,
    "deletion_fraction": 0.12,
    "duplicate_fraction": 0.10,
    "contradiction_fraction": 0.08,
    "tenants": 3,
}


def _workload(version: str, seed: int, sessions: int) -> SyntheticLongitudinalWorkload:
    return SyntheticLongitudinalWorkload(
        version=version, seed=seed, sessions=sessions, **SWEEP_PARAMS
    )


@pytest.mark.parametrize(("seed", "sessions"), [(1, 24), (3, 40), (7, 64)])
def test_no_event_asserts_an_internal_marker(seed: int, sessions: int) -> None:
    for event in _workload("v4", seed, sessions).events():
        assert event.metadata.get("value") != CONTRADICTION_MARKER
        assert CONTRADICTION_MARKER not in event.content


def test_v2_still_emits_the_marker_so_the_fix_is_v4_only() -> None:
    events = _workload("v2", 7, 64).events()
    assert any(event.metadata.get("value") == CONTRADICTION_MARKER for event in events)


@pytest.mark.parametrize(("seed", "sessions"), [(1, 24), (3, 40), (7, 64)])
def test_temporal_gold_is_the_value_current_at_its_instant_and_not_the_current_one(
    seed: int, sessions: int
) -> None:
    workload = _workload("v4", seed, sessions)
    facts = [
        event
        for event in workload.events()
        if event.event_type == EventType.user_message and "value" in event.metadata
    ]
    queries = workload.queries()
    current = {
        (query.tenant_id, query.metadata["attribute"]): query.gold_answer
        for query in queries
        if query.query_id.endswith("_cur")
    }
    temporal = [query for query in queries if query.category == "temporal_reasoning"]
    assert temporal
    for query in temporal:
        key = (query.tenant_id, query.metadata["attribute"])
        assert query.as_of_time is not None
        assert query.gold_answer != current[key]
        asserted = [
            event for event in facts if (event.tenant_id, event.metadata["attribute"]) == key
        ]
        assert query.gold_answer in {event.metadata["value"] for event in asserted}
        before = [event for event in asserted if event.event_time <= query.as_of_time]
        latest = max(event.event_time for event in before)
        values_then = {event.metadata["value"] for event in before if event.event_time == latest}
        assert values_then == {query.gold_answer}
        assert any(event.event_time > query.as_of_time for event in asserted)
