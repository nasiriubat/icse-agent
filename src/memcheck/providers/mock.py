"""Deterministic mock provider — the *oracle reader* (DECISIONS D-006).

Why this exists
---------------
The research questions ask about the **memory layer**, not about language-model ability. A
frontier model would add non-determinism, cost, and a confound: a wrong answer could come from
bad memory *or* from bad reading. This provider is a reader that is deliberately perfect at
reading and perfectly obedient about scope:

* it answers **only** from the bounded evidence it is given;
* it answers **correctly** when the evidence contains the currently-valid fact;
* it answers **incorrectly** when the evidence contains only a stale or superseded value;
* it **abstains** when evidence is absent, or when two values conflict and nothing in the
  evidence establishes which one is current.

Consequently every error observed in a mock-provider run is attributable to what the memory
system stored, kept, deleted, or retrieved. Results are an upper bound on end-to-end agent
accuracy, never a substitute for it; the README and the threats-to-validity section say so.

The resolution rule is identical for every backend. Backends differ only in what evidence and
what validity metadata they are able to supply.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any

from memcheck.models import ChatResponse, EmbeddingResponse, ProviderCall, sha256_text
from memcheck.providers.base import ChatRequest, Provider, ProviderError
from memcheck.utilities.tokens import STOPWORDS

ABSTAIN_NO_EVIDENCE = "ABSTAIN: no supporting evidence"
ABSTAIN_CONFLICT = "ABSTAIN: conflicting evidence"

_STOPWORDS = STOPWORDS

# A dot is part of a token only when something follows it, so "config.yaml" and "v1.2.3" stay
# whole while a sentence-final period does not become part of the last word. Getting this wrong
# silently breaks fact identity: "kubernetes" and "kubernetes." would be different facts.
_WORD = re.compile(r"[a-z0-9](?:[a-z0-9_\-/]|\.(?=[a-z0-9]))*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_VALUE_PATTERNS = (
    re.compile(r"\b(?:is|are|was|were)\s+(?:now\s+|currently\s+)?(?P<value>.+?)\s*[.;!?]?$", re.I),
    re.compile(r"[:=]\s*(?P<value>.+?)\s*[.;!?]?$"),
)


def content_terms(text: str) -> set[str]:
    """Lowercased content words, stopwords removed. The reader's only notion of matching."""
    return {word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS}


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def extract_value(sentence: str) -> str | None:
    """Pull the asserted value out of a fact sentence."""
    for pattern in _VALUE_PATTERNS:
        match = pattern.search(sentence)
        if match:
            value = match.group("value").strip().strip('"').strip("'")
            if value:
                return value
    return None


def normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower()).rstrip(".")


class _Candidate:
    __slots__ = ("confidence", "memory_id", "raw_value", "score", "valid_from", "valid_to", "value")

    def __init__(
        self,
        memory_id: str,
        raw_value: str,
        valid_from: datetime | None,
        valid_to: datetime | None,
        confidence: float,
        score: float,
    ) -> None:
        self.memory_id = memory_id
        self.raw_value = raw_value
        self.value = normalize_value(raw_value)
        self.valid_from = valid_from
        self.valid_to = valid_to
        self.confidence = confidence
        self.score = score


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


class MockProvider(Provider):
    """Deterministic oracle reader. Never performs network I/O and always costs zero."""

    name = "mock"
    is_paid = False

    def __init__(self, identifier: str = "oracle-reader-v1") -> None:
        self.identifier = identifier
        self._request_counter = 0

    # -- chat ---------------------------------------------------------------------------
    def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        self._request_counter += 1
        payload = request.structured_input or {}
        purpose = request.purpose

        if purpose == "answer":
            text, structured = self._answer(payload)
        elif purpose == "judge":
            text, structured = self._judge(payload)
        elif purpose == "code_patch":
            text, structured = self._code_patch(payload)
        elif purpose == "extract_memories":
            text, structured = self._extract_memories(payload)
        else:
            raise ProviderError(f"mock provider has no behavior for purpose {purpose!r}")

        input_tokens = self.count_tokens(request.system_prompt) + sum(
            self.count_tokens(message.content) for message in request.messages
        )
        latency_ms = (time.perf_counter() - started) * 1000
        call = ProviderCall(
            provider=self.name,
            model_identifier=self.identifier,
            request_identifier=f"mock-{sha256_text(text + str(self._request_counter))[:16]}",
            input_tokens=input_tokens,
            output_tokens=self.count_tokens(text),
            latency_ms=latency_ms,
            retry_count=0,
            estimated_cost_usd=0.0,
            purpose=purpose,
        )
        return ChatResponse(text=text, call=call, structured=structured)

    # -- behaviors ----------------------------------------------------------------------
    def _answer(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Answer strictly from bounded evidence, using one rule for every backend."""
        query: str = payload.get("query", "")
        evidence: list[dict[str, Any]] = payload.get("evidence", [])
        abstain_on_conflict: bool = payload.get("abstain_on_conflict", True)
        as_of = _parse_time(payload.get("as_of_time"))

        query_terms = content_terms(query)
        if not query_terms:
            return ABSTAIN_NO_EVIDENCE, {"reason": "empty_query", "used_memory_ids": []}

        candidates: list[_Candidate] = []
        for item in evidence:
            valid_from = _parse_time(item.get("valid_from"))
            valid_to = _parse_time(item.get("valid_to"))
            # A closed validity interval that has already ended is explicitly historical:
            # the reader treats it as background, not as a current assertion.
            if valid_to is not None and (as_of is None or valid_to <= as_of):
                continue
            for sentence in split_sentences(str(item.get("content", ""))):
                sentence_terms = content_terms(sentence)
                if not query_terms.issubset(sentence_terms):
                    continue
                value = extract_value(sentence)
                if value is None:
                    continue
                # The value must add information beyond restating the question.
                if not (content_terms(value) - query_terms):
                    continue
                candidates.append(
                    _Candidate(
                        memory_id=str(item.get("memory_id", "")),
                        raw_value=value,
                        valid_from=valid_from,
                        valid_to=valid_to,
                        confidence=float(item.get("confidence", 1.0)),
                        score=len(query_terms) / max(1, len(sentence_terms)),
                    )
                )

        if not candidates:
            return ABSTAIN_NO_EVIDENCE, {"reason": "no_matching_evidence", "used_memory_ids": []}

        # Keep only the most *specific* matches. A chatty sentence that happens to contain the
        # question's terms ("The queue partition count review meeting was productive.") carries
        # more unrelated terms than the fact itself ("The queue partition count is sixteen."),
        # so it scores lower and is discarded. Without this the reader answers from distractors
        # and the measurement stops being about the memory system.
        best_score = max(candidate.score for candidate in candidates)
        candidates = [candidate for candidate in candidates if candidate.score >= best_score - 1e-9]

        distinct = {candidate.value for candidate in candidates}
        if len(distinct) == 1:
            winner = candidates[0]
            return winner.raw_value, {
                "reason": "single_supported_value",
                "used_memory_ids": sorted({c.memory_id for c in candidates}),
            }

        # Several distinct values: recency decides, but only when the evidence actually
        # establishes an order. Missing or tied timestamps mean the reader cannot tell.
        dated = [c for c in candidates if c.valid_from is not None]
        if len(dated) == len(candidates):
            newest = max(c.valid_from for c in dated)  # type: ignore[type-var]
            leaders = [c for c in dated if c.valid_from == newest]
            if len({c.value for c in leaders}) == 1:
                return leaders[0].raw_value, {
                    "reason": "most_recent_valid_value",
                    "used_memory_ids": sorted({c.memory_id for c in leaders}),
                    "competing_values": sorted(distinct),
                }
        if abstain_on_conflict:
            return ABSTAIN_CONFLICT, {
                "reason": "unresolved_conflict",
                "used_memory_ids": sorted({c.memory_id for c in candidates}),
                "competing_values": sorted(distinct),
            }
        # Conflict-tolerant mode (used by the abstention ablation): take the highest-scoring
        # candidate, breaking ties on evidence order, which is what a reader without an
        # abstention policy would do.
        winner = max(candidates, key=lambda c: (c.score, c.confidence))
        return winner.raw_value, {
            "reason": "conflict_ignored",
            "used_memory_ids": [winner.memory_id],
            "competing_values": sorted(distinct),
        }

    def _judge(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Deterministic stand-in for an LLM judge: normalized containment comparison."""
        model_answer = normalize_value(str(payload.get("model_answer", "")))
        gold_answer = normalize_value(str(payload.get("gold_answer", "")))
        if model_answer.startswith("abstain"):
            label = "abstained"
        elif gold_answer and (gold_answer in model_answer or model_answer in gold_answer):
            label = "correct"
        else:
            label = "incorrect"
        rationale = f"normalized comparison of {model_answer!r} against {gold_answer!r}"
        return (
            f'{{"label": "{label}", "rationale": "{rationale}"}}',
            {"label": label, "rationale": rationale},
        )

    def _code_patch(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Emit a structured file edit derived from the evidence, never from hidden knowledge.

        The coding task states a placeholder token that must be replaced by the project's
        *current* convention. The mock resolves that convention with the same evidence rule the
        answer path uses, so a stale memory produces a stale patch and the repository's unit
        tests fail — which is exactly the downstream effect RQ2 measures.
        """
        answer_text, answer_meta = self._answer(payload)
        template: str = payload.get("patch_template", "")
        if answer_text.startswith("ABSTAIN"):
            return "", {
                "abstained": True,
                "reason": answer_meta.get("reason", "abstained"),
                "edits": [],
            }
        resolved = answer_text.strip().rstrip(".")
        edits = [
            {
                "path": edit["path"],
                "content": edit["content"].replace("{{VALUE}}", resolved),
            }
            for edit in payload.get("edits", [])
        ]
        rendered = template.replace("{{VALUE}}", resolved)
        return rendered, {
            "abstained": False,
            "resolved_value": resolved,
            "edits": edits,
            "used_memory_ids": answer_meta.get("used_memory_ids", []),
        }

    def _extract_memories(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Candidate-memory extraction from an event, in the shape a real model is asked for.

        This is a stand-in, not a simulation: it applies the same copula rule as the built-in
        rule-based extractor and dresses the result in the subject/value/statement shape the
        extraction prompt specifies. That keeps the LLM extraction path exercisable offline and
        in CI, while making no pretence of reproducing a model's judgement about what is worth
        remembering --- which is precisely the thing under study, and so must come from a real
        model whenever it is being measured.
        """
        content = str(payload.get("content", ""))
        facts: list[dict[str, str]] = []
        for sentence in split_sentences(content):
            value = extract_value(sentence)
            if value is None:
                continue
            subject_terms = content_terms(sentence) - content_terms(value)
            if not subject_terms:
                continue
            facts.append(
                {
                    "subject": " ".join(sorted(subject_terms)),
                    "value": value,
                    "statement": sentence.strip(),
                }
            )
        return (json.dumps({"facts": facts}), {"facts": facts})

    # -- embeddings ---------------------------------------------------------------------
    def embed(self, texts: list[str], *, model_identifier: str | None = None) -> EmbeddingResponse:
        """Deterministic hash embedding. Reproducible, offline, and clearly not semantic."""
        dimensions = 128
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * dimensions
            for term in content_terms(text):
                index = int(sha256_text(term)[:8], 16) % dimensions
                vector[index] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        call = ProviderCall(
            provider=self.name,
            model_identifier=model_identifier or "hash-embed-v1",
            request_identifier=f"mock-embed-{len(texts)}",
            input_tokens=sum(self.count_tokens(text) for text in texts),
            purpose="embed",
        )
        return EmbeddingResponse(vectors=vectors, call=call)

    def estimate_cost(self, input_tokens: int, output_tokens: int, model_identifier: str) -> float:
        return 0.0
