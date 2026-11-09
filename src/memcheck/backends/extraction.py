"""What to remember: the write-side policy, separated from the lifecycle that manages it.

The guarded backend's lifecycle -- versioning, supersession, deletion closure, isolation --
operates on whatever candidates it is handed. Which candidates those are is a *different*
decision, and until now the two were fused in one method, which made it impossible to ask how
much of the system's behaviour each is responsible for. Pulling extraction behind this interface
lets the lifecycle be held fixed while the write policy varies, and vice versa.

Two implementations are provided.

:class:`RuleBasedExtractor` is the original: any sentence containing a copula yields a fact. It
is sound on templated text and catastrophic on dialogue, where it fires on generic assistant
advice and fills the store with fragments keyed on strings no question will ever match.

:class:`LLMExtractor` asks a model what is worth remembering, and --- crucially --- allows it to
answer "nothing". The canonical key comes from a short subject phrase the model is asked to name
consistently, because supersession can only work if the same fact gets the same key on a later
occasion. Whether a model actually does that is an empirical question this interface exists to
let us measure, not an assumption.

Every extractor carries an ``identity`` string that is recorded in the run manifest. Two runs
using different extractors are not two samples of one experiment, and the fairness validator
treats a difference here exactly as it treats a change of model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from memcheck.models import Event
from memcheck.providers.base import ChatMessage, ChatRequest, Provider, ProviderLengthError
from memcheck.providers.mock import content_terms, extract_value, normalize_value, split_sentences
from memcheck.providers.prompts import get_prompt

#: Default output budget for extraction. A fact list is short, but on a reasoning model the
#: reasoning tokens are charged against this budget *before* any content is emitted, so a budget
#: sized for the visible output returns nothing at all. This bit the answering path first; the
#: same mistake on the write path cost three conditions of a paid run. Callers should pass the
#: run's configured budget rather than relying on this.
EXTRACTION_MAX_TOKENS = 4000
#: Budget multiplier for the single retry after a length stop (D-021).
LENGTH_RETRY_FACTOR = 4

#: A single event that yields more candidates than this is a symptom of the failure mode this
#: interface exists to study, so the surplus is dropped and counted rather than silently stored.
MAX_CANDIDATES_PER_EVENT = 12


@dataclass(frozen=True)
class Candidate:
    """One fact proposed for storage.

    ``canonical_key`` is the identity under which supersession happens: a later candidate with
    the same key on the same tenant revises this one. ``normalized_value`` is what two versions
    of the fact are compared on when deciding whether an update actually changed anything.
    """

    canonical_key: str
    statement: str
    normalized_value: str


@runtime_checkable
class Extractor(Protocol):
    """Turns an event into the facts worth storing from it."""

    @property
    def identity(self) -> str:
        """Recorded in the manifest; two runs differing here are not comparable."""

    def extract(self, event: Event) -> list[Candidate]: ...


def _deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the first candidate per key, and cap the total.

    One event asserting the same key twice is a contradiction within a single utterance, which
    the lifecycle has no way to order; taking the first is arbitrary but deterministic, and the
    alternative -- storing both -- would manufacture a conflict the source never contained.
    """
    seen: set[str] = set()
    kept: list[Candidate] = []
    for candidate in candidates:
        if candidate.canonical_key in seen:
            continue
        seen.add(candidate.canonical_key)
        kept.append(candidate)
        if len(kept) >= MAX_CANDIDATES_PER_EVENT:
            break
    return kept


class RuleBasedExtractor:
    """Copula-matching extraction: sound on templated text, indiscriminate on dialogue.

    The canonical key is the set of content terms that are *not* part of the asserted value,
    which is exactly the term set a question about that fact carries. Identity therefore means
    the same thing to the writer and to the reader, and supersession lines up with what a later
    query will match --- as long as the input is phrased like a statement of fact. On free-form
    dialogue it is not, and this extractor's behaviour there is a result reported in the paper
    rather than a defect to be quietly patched.
    """

    identity = "rule_copula_v1"

    def extract(self, event: Event) -> list[Candidate]:
        candidates: list[Candidate] = []
        for sentence in split_sentences(event.content):
            value = extract_value(sentence)
            if value is None:
                continue
            value_terms = content_terms(value)
            key_terms = content_terms(sentence) - value_terms
            if not key_terms or not value_terms:
                continue
            candidates.append(
                Candidate(
                    canonical_key=" ".join(sorted(key_terms)),
                    statement=sentence.strip(),
                    normalized_value=normalize_value(value),
                )
            )
        return _deduplicate(candidates)


class LLMExtractor:
    """Model-decided extraction, with an explicit option to remember nothing.

    The model names a short ``subject`` for each fact; the canonical key is that subject's
    content terms. This is the load-bearing assumption: supersession requires that the same fact
    receives the same subject on a later occasion, and a model that names it differently each
    time will accumulate versions instead of replacing them. The extractor does not try to
    prevent that --- measuring it is the point --- but it does normalise the subject the same way
    the rule-based extractor normalises its keys, so the two are compared on equal terms.
    """

    def __init__(
        self,
        provider: Provider,
        model_identifier: str,
        *,
        temperature: float = 0.0,
        prompt_id: str = "extract_v1",
        max_output_tokens: int = EXTRACTION_MAX_TOKENS,
    ) -> None:
        self.provider = provider
        self.model_identifier = model_identifier
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.prompt_id = prompt_id
        self.system_prompt = get_prompt(prompt_id)
        #: Every call is retained so ingestion cost can be attributed and reported separately
        #: from answering cost.
        self.calls: list[object] = []
        self.malformed_responses = 0
        #: Events whose extraction stopped on the output budget even after one retry with a
        #: larger budget. Counted rather than raised: one such event must not void a whole
        #: condition, and counting keeps it distinguishable from "the model chose to remember
        #: nothing".
        self.truncated_responses = 0

    @property
    def identity(self) -> str:
        return f"llm:{self.model_identifier}:{self.prompt_id}"

    def _request(self, event: Event, max_output_tokens: int) -> ChatRequest:
        return ChatRequest(
            model_identifier=self.model_identifier,
            system_prompt=self.system_prompt,
            messages=[ChatMessage(role="user", content=event.content)],
            temperature=self.temperature,
            max_output_tokens=max_output_tokens,
            purpose="extract_memories",
            structured_input={"content": event.content, "event_time": event.event_time.isoformat()},
        )

    def extract(self, event: Event) -> list[Candidate]:
        # A reasoning model can spend the whole budget before emitting content. That is a
        # property of one event's length, not of the policy, so the call is retried once with
        # a budget large enough for any event in the workloads here; a second failure is
        # counted and the event contributes no candidates.
        try:
            response = self.provider.chat(self._request(event, self.max_output_tokens))
        except ProviderLengthError:
            try:
                response = self.provider.chat(
                    self._request(event, self.max_output_tokens * LENGTH_RETRY_FACTOR)
                )
            except ProviderLengthError:
                self.truncated_responses += 1
                return []
        self.calls.append(response.call)
        facts = self._facts_from(response.structured, response.text)
        return _deduplicate([c for c in (self._candidate(fact) for fact in facts) if c is not None])

    def _facts_from(self, structured: dict[str, object] | None, text: str) -> list[dict[str, str]]:
        """Read the fact list from a structured payload, falling back to parsing JSON text.

        A real provider returns JSON in the message body; the mock returns a structured payload
        directly. A response we cannot parse is counted rather than guessed at, because silently
        extracting nothing looks identical to a model that correctly decided to remember nothing.
        """
        payload = structured or {}
        facts = payload.get("facts")
        if facts is None:
            stripped = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            try:
                parsed = json.loads(stripped or "{}")
            except json.JSONDecodeError:
                self.malformed_responses += 1
                return []
            facts = parsed.get("facts") if isinstance(parsed, dict) else parsed
        if not isinstance(facts, list):
            self.malformed_responses += 1
            return []
        return [fact for fact in facts if isinstance(fact, dict)]

    def _candidate(self, fact: dict[str, str]) -> Candidate | None:
        subject = str(fact.get("subject") or "").strip()
        value = str(fact.get("value") or "").strip()
        statement = str(fact.get("statement") or "").strip()
        if not subject or not value:
            return None
        # The subject is normalised exactly as the rule-based extractor normalises its keys, so
        # a difference between the two extractors is a difference in *what* they select, not in
        # how identity is computed.
        key_terms = content_terms(subject) - content_terms(value)
        if not key_terms:
            key_terms = content_terms(subject)
        if not key_terms:
            return None
        return Candidate(
            canonical_key=" ".join(sorted(key_terms)),
            statement=statement or f"{subject} is {value}",
            normalized_value=normalize_value(value),
        )


def build_extractor(
    name: str,
    *,
    provider: Provider | None = None,
    model_identifier: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = EXTRACTION_MAX_TOKENS,
) -> Extractor:
    """Construct an extractor by name, so a configuration can select one."""
    if name in {"rule_based", "rule_copula_v1"}:
        return RuleBasedExtractor()
    if name in {"llm", "llm_v1"}:
        if provider is None or model_identifier is None:
            raise ValueError("the LLM extractor needs a provider and a model identifier")
        return LLMExtractor(
            provider,
            model_identifier,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(f"unknown extractor {name!r}; known: rule_based, llm")
