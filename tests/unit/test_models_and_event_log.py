"""Event validation, provenance enforcement, and idempotent append."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from memcheck.event_log import EventLog
from memcheck.models import Event, EventType, MemoryRecord, MemoryStatus, RetrievalResult

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class TestEventValidation:
    def test_source_hash_is_derived_from_content(self, make_event) -> None:
        event = make_event(1, "The primary database is postgres.")
        assert event.source_hash
        assert event.source_hash == event.compute_source_hash()

    def test_same_content_under_different_ids_hashes_identically(self, make_event) -> None:
        first = make_event(1, "The cache layer is redis.", event_id="a")
        second = make_event(1, "The cache layer is redis.", event_id="b")
        assert first.source_hash == second.source_hash

    def test_different_content_hashes_differently(self, make_event) -> None:
        assert make_event(1, "A is x.").source_hash != make_event(1, "A is y.").source_hash

    def test_empty_identifier_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Event(
                event_id="  ",
                tenant_id="t",
                session_id="s",
                event_type=EventType.user_message,
                event_time=EPOCH,
                content="x",
            )

    def test_unknown_field_is_rejected(self, make_event) -> None:
        payload = make_event(1, "A is x.").model_dump()
        payload["not_a_field"] = 1
        with pytest.raises(ValidationError):
            Event.model_validate(payload)


class TestMemoryRecord:
    def test_provenance_is_required(self) -> None:
        with pytest.raises(ValidationError, match="provenance"):
            MemoryRecord(
                memory_id="m1",
                tenant_id="t",
                canonical_key="k",
                version=1,
                content="c",
                valid_from=EPOCH,
                source_event_ids=[],
            )

    def test_inverted_validity_interval_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="valid_to"):
            MemoryRecord(
                memory_id="m1",
                tenant_id="t",
                canonical_key="k",
                version=1,
                content="c",
                valid_from=EPOCH + timedelta(days=2),
                valid_to=EPOCH,
                source_event_ids=["e1"],
            )

    @pytest.mark.parametrize(
        ("valid_to", "when", "expected"),
        [
            (None, None, True),
            (EPOCH + timedelta(days=5), None, False),
            (EPOCH + timedelta(days=5), EPOCH + timedelta(days=3), True),
            (EPOCH + timedelta(days=5), EPOCH + timedelta(days=7), False),
            (None, EPOCH - timedelta(days=1), False),
        ],
    )
    def test_temporal_validity(self, valid_to, when, expected) -> None:
        record = MemoryRecord(
            memory_id="m1",
            tenant_id="t",
            canonical_key="k",
            version=1,
            content="c",
            valid_from=EPOCH,
            valid_to=valid_to,
            source_event_ids=["e1"],
        )
        assert record.is_valid_at(when) is expected

    def test_status_values_cover_the_specified_set(self) -> None:
        assert {status.value for status in MemoryStatus} == {
            "active",
            "superseded",
            "deleted",
            "conflicted",
            "invalid",
        }


class TestRetrievalResult:
    def test_abstention_requires_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="abstention_reason"):
            RetrievalResult(query="q", tenant_id="t", abstained=True)


class TestEventLog:
    @pytest.fixture
    def log(self) -> EventLog:
        connection = sqlite3.connect(":memory:")
        return EventLog(connection)

    def test_append_then_replay_preserves_order(self, log, make_event) -> None:
        events = [make_event(index, f"The item {index} is value{index}.") for index in range(1, 6)]
        for event in events:
            log.append(event)
        assert [event.event_id for event in log.replay()] == [e.event_id for e in events]

    def test_duplicate_event_id_is_ignored(self, log, make_event) -> None:
        event = make_event(1, "The primary database is postgres.")
        assert log.append(event).appended is True
        second = log.append(event)
        assert second.appended is False
        assert second.detected_on == "event_id"
        assert log.count() == 1
        assert log.duplicate_count() == 1

    def test_duplicate_content_under_a_fresh_id_is_ignored(self, log, make_event) -> None:
        log.append(make_event(1, "The cache layer is redis.", event_id="a"))
        result = log.append(make_event(1, "The cache layer is redis.", event_id="b"))
        assert result.appended is False
        assert result.detected_on == "source_hash"

    def test_duplicate_detection_can_be_disabled(self, log, make_event) -> None:
        """With the guard off the duplicate is *stored*, not rejected.

        Idempotency must live in the append path rather than in a schema constraint: a UNIQUE
        index would make the guard impossible to ablate, because the store would crash instead
        of exhibiting the duplicate amplification the ablation exists to measure.
        """
        event = make_event(1, "The cache layer is redis.")
        log.append(event)
        result = log.append(event, detect_duplicates=False)
        assert result.appended is True
        assert log.count() == 2

    def test_get_returns_the_first_delivery_when_duplicates_were_stored(
        self, log, make_event
    ) -> None:
        event = make_event(1, "The cache layer is redis.")
        first = log.append(event)
        log.append(event, detect_duplicates=False)
        fetched = log.get(event.tenant_id, event.event_id)
        assert fetched is not None
        assert first.sequence is not None

    def test_tenants_are_isolated_in_counts(self, log, make_event) -> None:
        log.append(make_event(1, "A is x.", tenant_id="t1"))
        log.append(make_event(2, "B is y.", tenant_id="t2"))
        assert log.count("t1") == 1
        assert log.count("t2") == 1
        assert log.count() == 2
        assert log.tenants() == ["t1", "t2"]

    def test_replay_respects_an_until_bound(self, log, make_event) -> None:
        for index in range(1, 6):
            log.append(make_event(index, f"The item {index} is value{index}."))
        until = EPOCH + timedelta(days=3)
        assert len(list(log.replay(until=until))) == 3

    def test_there_is_no_update_or_delete_path(self, log) -> None:
        assert not hasattr(log, "update")
        assert not hasattr(log, "delete")


class TestGradingCannotRewardAnEmptyAnswer:
    """A real-model run scored 78 empty completions as correct before this was fixed.

    `gpt-5-mini` is a reasoning model: reasoning tokens are charged against the output budget, so
    a budget sized for a short answer can be consumed entirely, returning no content. The
    deterministic grader then matched the empty string against every gold answer, because "" is a
    substring of everything.
    """

    def _answer(self, text: str):
        from memcheck.models import AnswerRecord

        return AnswerRecord(
            query_id="q1",
            query="What is the retention period?",
            tenant_id="t",
            checkpoint="cp100",
            backend_name="guarded_memory",
            answer=text,
        )

    def _query(self, gold: str):
        from memcheck.workloads.base import Query

        return Query(
            query_id="q1", tenant_id="t", text="What is the retention period?", gold_answer=gold
        )

    def _retrieval(self):
        from memcheck.models import Evidence, RetrievalResult

        return RetrievalResult(
            query="What is the retention period?",
            tenant_id="t",
            evidence=[
                Evidence(
                    memory_id="m1",
                    content="The retention period is thirty-days.",
                    source_event_ids=["e1"],
                )
            ],
        )

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_empty_is_never_correct(self, text) -> None:
        from memcheck.workloads.base import grade_answer

        grade = grade_answer(self._query("thirty-days"), self._answer(text), self._retrieval())
        assert grade.correct is False
        assert grade.label.value == "unsupported"

    def test_exact_value_is_correct(self) -> None:
        from memcheck.workloads.base import grade_answer

        grade = grade_answer(
            self._query("thirty-days"), self._answer("thirty-days"), self._retrieval()
        )
        assert grade.correct is True

    def test_a_short_substring_does_not_match_a_long_gold(self) -> None:
        from memcheck.workloads.base import grade_answer

        gold = "The user prefers Sony-compatible accessories and high-quality photography gear"
        grade = grade_answer(self._query(gold), self._answer("a"), self._retrieval())
        assert grade.correct is False

    def test_llm_judge_is_used_when_the_query_asks_for_it(self) -> None:
        from memcheck.workloads.base import Query, grade_answer

        query = Query(
            query_id="q1",
            tenant_id="t",
            text="How many appointments?",
            gold_answer="two appointments",
            grader="llm_judge",
        )
        calls = []

        def judge(q, gold, model):
            calls.append((gold, model))
            return "correct", "semantically equivalent"

        grade = grade_answer(query, self._answer("2"), self._retrieval(), judge=judge)
        assert grade.correct is True
        assert grade.grader == "llm_judge"
        assert grade.judge_rationale == "semantically equivalent"
        assert calls == [("two appointments", "2")]

    def test_judge_is_not_consulted_for_an_empty_answer(self) -> None:
        from memcheck.workloads.base import Query, grade_answer

        query = Query(
            query_id="q1", tenant_id="t", text="How many?", gold_answer="two", grader="llm_judge"
        )
        judge_called = []

        def judge(q, gold, model):
            judge_called.append(model)
            return "correct", "should not be reached"

        grade = grade_answer(query, self._answer(""), self._retrieval(), judge=judge)
        assert grade.correct is False
        assert judge_called == []
