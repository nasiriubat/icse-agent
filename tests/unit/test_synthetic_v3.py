"""The v3 synthetic workload: checkpoint-relative gold and padding sessions.

v3 exists so one stream can be asked, at each checkpoint, what was true then (DEVIATIONS V-023).
These tests pin three things: v2 is byte-identical to what it was, every v3 gold agrees with an
independent replay of the events up to its checkpoint, and padding changes nothing but the
events appended after the fact stream.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import pytest

from memcheck.backends.retrieval import content_tokens
from memcheck.models import Event, EventType
from memcheck.workloads.base import Checkpoint, query_is_asked
from memcheck.workloads.synthetic import (
    ATTRIBUTES,
    UNRELATED_PADDING_TEMPLATES,
    VALUES,
    SyntheticLongitudinalWorkload,
)

SWEEP_PARAMS = {
    "facts_per_session": 4,
    "distractors_per_session": 3,
    "update_fraction": 0.35,
    "deletion_fraction": 0.12,
    "duplicate_fraction": 0.10,
    "contradiction_fraction": 0.08,
    "tenants": 3,
}


def _digest(items: list[object], exclude: set[str] | None = None) -> str:
    text = "\n".join(
        item.model_dump_json(exclude=exclude)  # type: ignore[attr-defined]
        for item in items
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class TestV2IsUnchanged:
    @pytest.mark.parametrize(
        ("sessions", "events_digest", "queries_digest"),
        [(8, "c90c8e19fed89cce", "71f06620c0275a82"), (24, "3e2642e100cbd884", "414fea58dadfc3ae")],
    )
    def test_seed_three_streams_match_the_pinned_digests(
        self, sessions: int, events_digest: str, queries_digest: str
    ) -> None:
        """Pinned on 2026-11-04 before v3 was added; the reported runs depend on these."""
        workload = SyntheticLongitudinalWorkload(
            version="v2", seed=3, sessions=sessions, **SWEEP_PARAMS
        )
        assert _digest(list(workload.events())) == events_digest
        # The optional v3 field did not exist when the digest was pinned; it is None in v2.
        assert all(query.asked_at_fraction is None for query in workload.queries())
        assert _digest(list(workload.queries()), {"asked_at_fraction"}) == queries_digest

    def test_padding_is_refused_outside_v3(self) -> None:
        with pytest.raises(ValueError, match="v3"):
            SyntheticLongitudinalWorkload(version="v2", seed=1, sessions=4, padding_sessions=2)


def _replay(events: list[Event], upto: int) -> dict[tuple[str, str], dict[str, object]]:
    """Independent reconstruction of the world after ``upto`` events."""
    state: dict[tuple[str, str], dict[str, object]] = {}
    for event in events[:upto]:
        if event.event_type == EventType.deletion_request:
            key = (event.tenant_id, str(event.metadata["deletion_target"]))
            state.setdefault(key, {"values": []})["deleted"] = True
            continue
        if "attribute" not in event.metadata:
            continue
        key = (event.tenant_id, str(event.metadata["attribute"]))
        entry = state.setdefault(key, {"values": []})
        values: list[tuple[datetime, str]] = entry["values"]  # type: ignore[assignment]
        assertion = (event.event_time, str(event.metadata["value"]))
        # A duplicate delivery repeats the previous assertion verbatim and is not a new one.
        if values and values[-1] == assertion:
            continue
        values.append(assertion)
    return state


class TestCheckpointGold:
    @pytest.mark.parametrize("seed", [1, 2, 5])
    def test_every_gold_agrees_with_a_replay_of_the_events_up_to_its_checkpoint(
        self, seed: int
    ) -> None:
        fractions = [0.25, 0.5, 0.75, 1.0]
        workload = SyntheticLongitudinalWorkload(
            version="v3", seed=seed, sessions=16, checkpoint_fractions=fractions, **SWEEP_PARAMS
        )
        events = workload.events()
        checkpoints = {cp.fraction: cp for cp in workload.checkpoints(fractions)}
        asked = 0
        for query in workload.queries():
            assert query.asked_at_fraction in checkpoints
            checkpoint = checkpoints[query.asked_at_fraction]
            # A query is asked at exactly one checkpoint.
            assert [cp for cp in checkpoints.values() if query_is_asked(query, cp)] == [checkpoint]
            key = (query.tenant_id, str(query.metadata["attribute"]))
            world = _replay(events, checkpoint.event_index)[key]
            values: list[tuple[datetime, str]] = world["values"]  # type: ignore[assignment]
            assert values, "a query must have supporting events delivered by its checkpoint"
            latest_time = max(when for when, _ in values)
            latest_values = {value for when, value in values if when == latest_time}
            # Event order decides which value is current; two updates of one key in one
            # session share a timestamp and the later one wins, as in the generator.
            current = values[-1][1]
            asked += 1
            if query.metadata.get("reason") == "deleted":
                assert world.get("deleted") is True
                assert set(query.forbidden_values) == {value for _, value in values}
                continue
            assert world.get("deleted") is not True
            # A contradiction needs two values at the latest time, or a latest assertion whose
            # value is the marker string: the generator re-asserts "the same fact" from its own
            # contradiction marker (a v2 quirk v3 reproduces on purpose, so the streams match).
            # The converse is not checked: a same-session pair of updates also leaves two values
            # at one timestamp, and only the generator knows which pairs it meant as conflicts.
            if query.metadata.get("reason") == "unresolved_contradiction":
                assert len(latest_values) > 1 or current == "__contradiction__"
                continue
            assert current != "__contradiction__"
            if query.category == "temporal_reasoning":
                # Gold is an earlier assertion of the key, delivered by the checkpoint, asked as
                # of a time between that assertion and the latest one. (Exactly which earlier
                # entry is v2's definition, pinned by the final-checkpoint equivalence test; a
                # same-session re-assertion and a duplicate delivery look alike in the events.)
                assert (query.as_of_time is not None) and query.gold_answer is not None
                assert query.as_of_time <= latest_time
                asserted_by_then = {value for when, value in values if when <= query.as_of_time}
                assert query.gold_answer in asserted_by_then, (
                    "temporal gold must have been asserted by the as-of time"
                )
            else:
                assert query.gold_answer == current
                assert current not in query.stale_values
        assert asked > 0

    def test_the_final_checkpoint_asks_what_v2_asks(self) -> None:
        """At the full store the v3 set must reduce to v2's: same keys, same gold, same kind."""
        v2 = SyntheticLongitudinalWorkload(version="v2", seed=4, sessions=16, **SWEEP_PARAMS)
        v3 = SyntheticLongitudinalWorkload(
            version="v3", seed=4, sessions=16, checkpoint_fractions=[0.5, 1.0], **SWEEP_PARAMS
        )
        assert [e.event_id for e in v2.events()] == [e.event_id for e in v3.events()]

        def signature(query: object) -> tuple[object, ...]:
            q = query  # type: ignore[assignment]
            return (
                q.tenant_id,  # type: ignore[attr-defined]
                q.metadata["attribute"],  # type: ignore[attr-defined]
                q.category,  # type: ignore[attr-defined]
                q.gold_answer,  # type: ignore[attr-defined]
                q.expects_abstention,  # type: ignore[attr-defined]
                tuple(q.stale_values),  # type: ignore[attr-defined]
            )

        final = [q for q in v3.queries() if q.asked_at_fraction == 1.0]
        assert sorted(map(signature, final)) == sorted(map(signature, v2.queries()))

    def test_checkpoints_snap_to_session_ends(self) -> None:
        workload = SyntheticLongitudinalWorkload(
            version="v3",
            seed=1,
            sessions=64,
            checkpoint_fractions=[0.125, 0.25, 0.375, 0.625, 1.0],
            **SWEEP_PARAMS,
        )
        events = workload.events()
        for checkpoint, sessions in zip(
            workload.checkpoints([0.125, 0.25, 0.375, 0.625, 1.0]), (8, 16, 24, 40, 64), strict=True
        ):
            assert isinstance(checkpoint, Checkpoint)
            assert events[checkpoint.event_index - 1].session_id == f"s{sessions - 1:03d}"
            if checkpoint.event_index < len(events):
                assert events[checkpoint.event_index].session_id == f"s{sessions:03d}"


class TestPadding:
    def test_padding_leaves_the_fact_stream_and_the_query_set_untouched(self) -> None:
        base = SyntheticLongitudinalWorkload(version="v3", seed=2, sessions=8, **SWEEP_PARAMS)
        padded = SyntheticLongitudinalWorkload(
            version="v3", seed=2, sessions=8, padding_sessions=16, **SWEEP_PARAMS
        )
        base_events, padded_events = base.events(), padded.events()
        assert padded_events[: len(base_events)] == base_events
        assert len(padded_events) > len(base_events)
        assert all(e.metadata.get("role") == "padding" for e in padded_events[len(base_events) :])
        # Only the session count recorded in the metadata may differ (8 against 24).
        strip = {"metadata": {"sessions"}}
        assert [q.model_dump(exclude=strip) for q in padded.queries()] == [
            q.model_dump(exclude=strip) for q in base.queries()
        ]
        assert padded.checkpoints([1.0])[0].event_index == len(padded_events)

    def test_unrelated_padding_shares_no_content_token_with_facts(self) -> None:
        fact_tokens = {
            token for phrase in (*ATTRIBUTES, *VALUES) for token in content_tokens(phrase)
        }
        for template in UNRELATED_PADDING_TEMPLATES:
            assert not (set(content_tokens(template)) & fact_tokens), template
        workload = SyntheticLongitudinalWorkload(
            version="v3", seed=2, sessions=8, padding_sessions=4, padding_style="unrelated"
        )
        padding = [e for e in workload.events() if e.metadata.get("role") == "padding"]
        assert padding
        assert all(not (set(content_tokens(e.content)) & fact_tokens) for e in padding)

    def test_topical_padding_names_attributes_but_asserts_no_value(self) -> None:
        workload = SyntheticLongitudinalWorkload(
            version="v3", seed=2, sessions=8, padding_sessions=4, padding_style="topical"
        )
        padding = [e for e in workload.events() if e.metadata.get("role") == "padding"]
        assert padding
        assert all("attribute" not in e.metadata for e in padding)
        assert any(any(a in e.content for a in ATTRIBUTES) for e in padding)
