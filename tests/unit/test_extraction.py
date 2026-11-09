"""Tests for the write-side extraction policies.

The factorial rests on one property: the two extractors must differ in *what* they select and in
nothing else. If they computed identity differently, a difference between them would be an
artefact of the interface rather than a result about write policy, so several of these tests pin
that down rather than testing either extractor's output in isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memcheck.backends.extraction import (
    EXTRACTION_MAX_TOKENS,
    MAX_CANDIDATES_PER_EVENT,
    Candidate,
    LLMExtractor,
    RuleBasedExtractor,
    build_extractor,
)
from memcheck.models import Event, EventType
from memcheck.providers.base import ProviderLengthError
from memcheck.providers.mock import MockProvider


def _event(content: str, event_id: str = "e1") -> Event:
    return Event(
        event_id=event_id,
        tenant_id="acme",
        session_id="s1",
        event_type=EventType.user_message,
        event_time=datetime(2026, 1, 1, tzinfo=UTC),
        content=content,
    )


class TestRuleBasedExtractor:
    def test_a_stated_fact_becomes_one_candidate(self) -> None:
        candidates = RuleBasedExtractor().extract(_event("The deployment target is kubernetes."))
        assert len(candidates) == 1
        assert candidates[0].normalized_value == "kubernetes"
        # The key is what a later question about this fact would carry, minus the answer.
        assert "kubernetes" not in candidates[0].canonical_key
        assert "deployment" in candidates[0].canonical_key

    def test_two_statements_of_the_same_fact_share_a_key(self) -> None:
        """Supersession depends entirely on this."""
        first = RuleBasedExtractor().extract(_event("The deployment target is docker-swarm."))
        second = RuleBasedExtractor().extract(_event("The deployment target is kubernetes."))
        assert first[0].canonical_key == second[0].canonical_key
        assert first[0].normalized_value != second[0].normalized_value

    def test_a_sentence_asserting_nothing_yields_nothing(self) -> None:
        assert RuleBasedExtractor().extract(_event("Could you help me with the deploy?")) == []

    def test_it_fires_on_conversational_filler_which_is_the_known_failure(self) -> None:
        """Documented, not fixed: this behaviour is a result the paper reports."""
        candidates = RuleBasedExtractor().extract(
            _event("Lucidchart is a good option. Assigning blocks is usually easier.")
        )
        assert candidates, "the copula rule is expected to fire here; that is the point"

    def test_one_event_cannot_flood_the_store(self) -> None:
        content = " ".join(f"Item {i} is value{i}." for i in range(40))
        assert len(RuleBasedExtractor().extract(_event(content))) == MAX_CANDIDATES_PER_EVENT

    def test_a_key_asserted_twice_in_one_event_keeps_the_first(self) -> None:
        """Storing both would manufacture a conflict the source never contained."""
        candidates = RuleBasedExtractor().extract(
            _event("The deployment target is docker-swarm. The deployment target is kubernetes.")
        )
        assert len(candidates) == 1
        assert candidates[0].normalized_value == "docker-swarm"


class TestLLMExtractor:
    def test_it_reads_facts_from_a_structured_payload(self) -> None:
        extractor = LLMExtractor(MockProvider(), "oracle-reader-v1")
        candidates = extractor.extract(_event("The deployment target is kubernetes."))
        assert [c.normalized_value for c in candidates] == ["kubernetes"]
        assert extractor.malformed_responses == 0

    def test_identity_names_the_model_and_the_prompt(self) -> None:
        """Two runs differing here are not two samples of one experiment."""
        extractor = LLMExtractor(MockProvider(), "gpt-5-mini")
        assert extractor.identity == "llm:gpt-5-mini:extract_v1"
        assert extractor.identity != RuleBasedExtractor().identity

    def test_the_output_budget_is_the_callers_and_is_generous_by_default(self) -> None:
        """Regression: a reasoning model spends this budget before emitting any content.

        The answering path learned this first, when a 256-token budget produced 47% empty
        completions. The same mistake on the write path later cost three conditions of a paid
        run, so the default is generous and the caller can raise it further.
        """
        captured: list[int] = []

        class Recording(MockProvider):
            def chat(self, request):  # type: ignore[no-untyped-def]
                captured.append(request.max_output_tokens)
                return super().chat(request)

        LLMExtractor(Recording(), "m", max_output_tokens=9000).extract(_event("The a is b."))
        LLMExtractor(Recording(), "m").extract(_event("The a is b."))
        assert captured == [9000, EXTRACTION_MAX_TOKENS]
        assert EXTRACTION_MAX_TOKENS >= 1000, "a budget sized for the visible output returns none"

    def test_provider_calls_are_retained_so_write_cost_can_be_attributed(self) -> None:
        extractor = LLMExtractor(MockProvider(), "oracle-reader-v1")
        extractor.extract(_event("The retention period is thirty days.", "e1"))
        extractor.extract(_event("The primary database is postgres.", "e2"))
        assert len(extractor.calls) == 2

    def test_identity_is_computed_the_same_way_as_the_rule_based_extractor(self) -> None:
        """Otherwise a difference between the two would be an interface artefact."""
        rule = RuleBasedExtractor().extract(_event("The deployment target is kubernetes."))
        llm = LLMExtractor(MockProvider(), "oracle-reader-v1").extract(
            _event("The deployment target is kubernetes.")
        )
        assert rule[0].canonical_key == llm[0].canonical_key

    def test_an_empty_fact_list_is_a_decision_not_a_failure(self) -> None:
        extractor = LLMExtractor(_ScriptedProvider('{"facts": []}'), "scripted")
        assert extractor.extract(_event("hello")) == []
        assert extractor.malformed_responses == 0

    def test_unparseable_output_is_counted_rather_than_guessed_at(self) -> None:
        """Silently extracting nothing looks identical to correctly remembering nothing."""
        extractor = LLMExtractor(_ScriptedProvider("I could not comply."), "scripted")
        assert extractor.extract(_event("The target is kubernetes.")) == []
        assert extractor.malformed_responses == 1

    def test_json_wrapped_in_a_code_fence_is_still_read(self) -> None:
        payload = (
            '```json\n{"facts": [{"subject": "deployment target", "value": "kubernetes"}]}\n```'
        )
        extractor = LLMExtractor(_ScriptedProvider(payload), "scripted")
        candidates = extractor.extract(_event("anything"))
        assert [c.normalized_value for c in candidates] == ["kubernetes"]

    def test_a_fact_without_a_value_is_dropped(self) -> None:
        payload = '{"facts": [{"subject": "deployment target", "value": ""}]}'
        extractor = LLMExtractor(_ScriptedProvider(payload), "scripted")
        assert extractor.extract(_event("anything")) == []

    def test_a_statement_is_synthesised_when_the_model_omits_one(self) -> None:
        payload = '{"facts": [{"subject": "deployment target", "value": "kubernetes"}]}'
        extractor = LLMExtractor(_ScriptedProvider(payload), "scripted")
        assert extractor.extract(_event("anything"))[0].statement == (
            "deployment target is kubernetes"
        )


class TestBuildExtractor:
    def test_the_rule_based_policy_needs_no_provider(self) -> None:
        assert isinstance(build_extractor("rule_based"), RuleBasedExtractor)

    def test_the_llm_policy_refuses_to_be_built_without_one(self) -> None:
        with pytest.raises(ValueError, match="needs a provider"):
            build_extractor("llm")

    def test_an_unknown_policy_is_an_error_not_a_silent_default(self) -> None:
        with pytest.raises(ValueError, match="unknown extractor"):
            build_extractor("whatever")


class TestGuardedBackendUsesTheExtractor:
    def test_the_lifecycle_stores_whatever_the_policy_hands_it(self, tmp_path) -> None:
        """The lifecycle must not second-guess the write policy, or the factorial is confounded."""
        from memcheck.backends import build_backend
        from memcheck.config import RetrievalConfig

        class OneFact:
            identity = "fixed_v1"

            def extract(self, event: Event) -> list[Candidate]:
                return [Candidate("colour", "The colour is blue.", "blue")]

        backend = build_backend(
            "guarded_memory", tmp_path / "s.db", RetrievalConfig(), extractor=OneFact()
        )
        try:
            result = backend.ingest_event(_event("nothing here resembles a fact"))
            assert result.created_memory_ids
            evidence = backend.retrieve("colour", "acme").evidence
            assert any("blue" in item.content for item in evidence)
        finally:
            backend.close()

    def test_a_baseline_cannot_be_given_a_write_policy_it_does_not_have(self, tmp_path) -> None:
        from memcheck.backends import BackendError, build_backend
        from memcheck.config import RetrievalConfig

        with pytest.raises(BackendError, match="takes no extractor"):
            build_backend(
                "transcript_rag",
                tmp_path / "t.db",
                RetrievalConfig(),
                extractor=RuleBasedExtractor(),
            )


class _ScriptedProvider(MockProvider):
    """Returns one fixed body, so parsing can be tested apart from any model's behaviour."""

    def __init__(self, body: str) -> None:
        super().__init__("scripted")
        self.body = body

    def chat(self, request):  # type: ignore[no-untyped-def]
        response = super().chat(
            request.model_copy(update={"purpose": "extract_memories", "structured_input": {}})
        )
        return response.model_copy(update={"text": self.body, "structured": {}})


class _LengthStopProvider(_ScriptedProvider):
    """Stops on length for the first ``failures`` calls, then answers; records each budget."""

    def __init__(self, body: str, failures: int) -> None:
        super().__init__(body)
        self.failures = failures
        self.budgets: list[int] = []

    def chat(self, request):  # type: ignore[no-untyped-def]
        self.budgets.append(request.max_output_tokens)
        if len(self.budgets) <= self.failures:
            raise ProviderLengthError("stopped on 'length'")
        return super().chat(request)


class TestLengthStopRetry:
    """D-021: one over-long event must not void a condition."""

    BODY = '{"facts": [{"subject": "favourite colour", "value": "green"}]}'

    def test_retries_once_with_a_larger_budget(self) -> None:
        provider = _LengthStopProvider(self.BODY, failures=1)
        extractor = LLMExtractor(provider, "m", max_output_tokens=100)
        candidates = extractor.extract(_event("My favourite colour is green."))
        assert [c.canonical_key for c in candidates]
        assert provider.budgets == [100, 400]
        assert extractor.truncated_responses == 0

    def test_second_failure_is_counted_not_raised(self) -> None:
        provider = _LengthStopProvider(self.BODY, failures=2)
        extractor = LLMExtractor(provider, "m", max_output_tokens=100)
        assert extractor.extract(_event("My favourite colour is green.")) == []
        assert provider.budgets == [100, 400]
        assert extractor.truncated_responses == 1
        assert extractor.malformed_responses == 0


def test_temperature_is_sent_to_models_that_accept_it_including_zero() -> None:
    from memcheck.providers.openai_provider import accepts_temperature

    assert accepts_temperature("gpt-4o-2024-08-06")
    assert not accepts_temperature("gpt-5-mini-2025-08-07")
    assert not accepts_temperature("o3-mini")
    assert accepts_temperature("gpt-5-chat-latest")
