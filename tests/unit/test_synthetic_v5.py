"""The v5 synthetic workload: v3's checkpoint-relative queries and padding, with v4's labels.

v5 exists so the longitudinal run and the padding control use the same corrected labels as the
v4 sweep (DEVIATIONS V-025). These tests pin that its fact stream is v4's, that its final
checkpoint asks exactly what v4 asks, that no event carries the internal marker, that every
temporal gold is a value that was current before a change, and that padding changes nothing but
the events appended after the fact stream.
"""

from __future__ import annotations

import pytest

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
FRACTIONS = [0.125, 0.25, 0.375, 0.625, 1.0]


def _signature(query: object) -> tuple[object, ...]:
    q = query
    return (
        q.tenant_id,  # type: ignore[attr-defined]
        q.metadata["attribute"],  # type: ignore[attr-defined]
        q.category,  # type: ignore[attr-defined]
        q.gold_answer,  # type: ignore[attr-defined]
        q.expects_abstention,  # type: ignore[attr-defined]
        tuple(q.stale_values),  # type: ignore[attr-defined]
        q.as_of_time,  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize(("seed", "sessions"), [(1, 16), (4, 24), (7, 64)])
def test_the_final_checkpoint_asks_what_v4_asks(seed: int, sessions: int) -> None:
    v4 = SyntheticLongitudinalWorkload(version="v4", seed=seed, sessions=sessions, **SWEEP_PARAMS)
    v5 = SyntheticLongitudinalWorkload(
        version="v5", seed=seed, sessions=sessions, checkpoint_fractions=FRACTIONS, **SWEEP_PARAMS
    )
    assert [e.model_dump() for e in v5.events()] == [e.model_dump() for e in v4.events()]
    final = [q for q in v5.queries() if q.asked_at_fraction == 1.0]
    assert sorted(map(_signature, final), key=repr) == sorted(
        map(_signature, v4.queries()), key=repr
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_event_asserts_the_marker_and_temporal_gold_differs_from_current(seed: int) -> None:
    workload = SyntheticLongitudinalWorkload(
        version="v5", seed=seed, sessions=64, checkpoint_fractions=FRACTIONS, **SWEEP_PARAMS
    )
    assert all(e.metadata.get("value") != CONTRADICTION_MARKER for e in workload.events())
    current = {
        (q.tenant_id, q.metadata["attribute"], q.asked_at_fraction): q.gold_answer
        for q in workload.queries()
        if q.category in ("knowledge_update", "single_session_extraction")
    }
    temporal = [q for q in workload.queries() if q.category == "temporal_reasoning"]
    assert temporal
    for query in temporal:
        key = (query.tenant_id, query.metadata["attribute"], query.asked_at_fraction)
        assert query.gold_answer != current[key]


def test_v3_still_uses_its_own_temporal_definition() -> None:
    """v3's labels are pinned by the reported longitudinal run; v5 must not change them."""
    v3 = SyntheticLongitudinalWorkload(
        version="v3", seed=1, sessions=64, checkpoint_fractions=FRACTIONS, **SWEEP_PARAMS
    )
    degenerate = [
        q
        for q in v3.queries()
        if q.category == "temporal_reasoning"
        and any(
            other.category in ("knowledge_update", "single_session_extraction")
            and other.tenant_id == q.tenant_id
            and other.metadata["attribute"] == q.metadata["attribute"]
            and other.asked_at_fraction == q.asked_at_fraction
            and other.gold_answer == q.gold_answer
            for other in v3.queries()
        )
    ]
    assert degenerate, "v3 keeps the v2 temporal definition, degenerate queries included"


@pytest.mark.parametrize("style", ["topical", "unrelated"])
def test_padding_leaves_the_fact_stream_and_the_query_set_untouched(style: str) -> None:
    base = SyntheticLongitudinalWorkload(version="v5", seed=2, sessions=8, **SWEEP_PARAMS)
    padded = SyntheticLongitudinalWorkload(
        version="v5", seed=2, sessions=8, padding_sessions=16, padding_style=style, **SWEEP_PARAMS
    )
    base_events, padded_events = base.events(), padded.events()
    assert padded_events[: len(base_events)] == base_events
    assert all(e.metadata.get("role") == "padding" for e in padded_events[len(base_events) :])
    strip = {"metadata": {"sessions"}}
    assert [q.model_dump(exclude=strip) for q in padded.queries()] == [
        q.model_dump(exclude=strip) for q in base.queries()
    ]


def test_padding_is_refused_for_v4() -> None:
    with pytest.raises(ValueError, match="v5"):
        SyntheticLongitudinalWorkload(version="v4", seed=1, sessions=4, padding_sessions=2)
