"""The answering agent.

Every backend's evidence is rendered into the *same* prompt with the *same* system instructions
and the *same* budgets, so the only thing that varies across conditions is what the memory system
supplied. The rendered prompt and its hash are recorded in the run manifest.
"""

from __future__ import annotations

import time

from memcheck.config import ModelConfig
from memcheck.models import AbstentionReason, AnswerRecord, ProviderCall, RetrievalResult
from memcheck.providers.base import ChatMessage, ChatRequest, Provider
from memcheck.providers.prompts import get_prompt
from memcheck.workloads.base import Query

ABSTAIN_PREFIX = "ABSTAIN"


def render_evidence(retrieval: RetrievalResult) -> str:
    """Render bounded evidence as the prompt text every provider sees."""
    if not retrieval.evidence:
        return "(no evidence retrieved)"
    lines: list[str] = []
    for position, item in enumerate(retrieval.evidence, start=1):
        validity = "current"
        if item.valid_to is not None:
            validity = f"valid until {item.valid_to.isoformat()}"
        elif item.valid_from is not None:
            validity = f"asserted {item.valid_from.isoformat()}"
        lines.append(
            f"[{position}] {item.content}\n"
            f"    source_events={item.source_event_ids} validity={validity} "
            f"confidence={item.confidence:.2f} reason={item.retrieval_reason}"
        )
    return "\n".join(lines)


def render_prompt(query: Query, retrieval: RetrievalResult) -> str:
    as_of = (
        f"\nAnswer as of {query.as_of_time.isoformat()}." if query.as_of_time is not None else ""
    )
    return f"Memory evidence:\n{render_evidence(retrieval)}\n\nQuestion: {query.text}{as_of}"


class JudgeAgent:
    """Semantic grading for queries whose gold answer is free text (DECISIONS D-014).

    The judge prompt is version-controlled and its hash is recorded in the run manifest, and
    every judged decision is exported for human audit by ``memcheck audit`` so the agreement
    number is measured rather than assumed.
    """

    def __init__(self, provider: Provider, model: ModelConfig) -> None:
        self.provider = provider
        self.model = model
        self.system_prompt = get_prompt("judge_v1")
        self.calls: list[ProviderCall] = []

    def __call__(self, query: object, gold_answer: str, model_answer: str) -> tuple[str, str]:
        text = str(getattr(query, "text", ""))
        # An empty reply needs no judging: there is nothing to compare.
        if not model_answer.strip():
            return "incorrect", "the model returned no content"
        request = ChatRequest(
            model_identifier=self.model.identifier,
            system_prompt=self.system_prompt,
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        f"Question: {text}\n"
                        f"Gold answer: {gold_answer}\n"
                        f"Assistant answer: {model_answer}"
                    ),
                )
            ],
            temperature=self.model.temperature,
            max_output_tokens=self.model.max_output_tokens,
            purpose="judge",
            structured_input={"gold_answer": gold_answer, "model_answer": model_answer},
        )
        response = self.provider.chat(request)
        self.calls.append(response.call)
        structured = response.structured or {}
        label = str(structured.get("label") or "").strip().lower()
        rationale = str(structured.get("rationale") or "")
        if not label:
            # A real provider returns JSON text rather than a structured payload.
            lowered = response.text.lower()
            if '"correct"' in lowered or "'correct'" in lowered:
                label = "correct"
            elif "abstain" in lowered:
                label = "abstained"
            else:
                label = "incorrect"
            rationale = response.text[:300]
        return label, rationale


class AnswerAgent:
    """Turns a retrieval result into a graded answer record."""

    def __init__(
        self,
        provider: Provider,
        model: ModelConfig,
        *,
        abstain_on_conflict: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.abstain_on_conflict = abstain_on_conflict
        self.system_prompt = get_prompt(model.system_prompt_id)

    def answer(
        self,
        query: Query,
        retrieval: RetrievalResult,
        *,
        checkpoint: str,
        backend_name: str,
    ) -> AnswerRecord:
        started = time.perf_counter()

        # A backend that already abstained is not asked to answer: paying for a call whose
        # evidence is empty would inflate cost without changing the outcome.
        if retrieval.abstained and not retrieval.evidence:
            return AnswerRecord(
                query_id=query.query_id,
                query=query.text,
                tenant_id=query.tenant_id,
                checkpoint=checkpoint,
                backend_name=backend_name,
                answer=f"{ABSTAIN_PREFIX}: {retrieval.abstention_reason.value}"
                if retrieval.abstention_reason
                else f"{ABSTAIN_PREFIX}: no supporting evidence",
                abstained=True,
                abstention_reason=retrieval.abstention_reason or AbstentionReason.no_evidence,
                injected_tokens=retrieval.injected_tokens,
                retrieval_latency_ms=retrieval.retrieval_latency_ms,
                answer_latency_ms=(time.perf_counter() - started) * 1000,
                metadata={"skipped_provider_call": True},
            )

        prompt = render_prompt(query, retrieval)
        request = ChatRequest(
            model_identifier=self.model.identifier,
            system_prompt=self.system_prompt,
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=self.model.temperature,
            max_output_tokens=self.model.max_output_tokens,
            tools=list(self.model.tools_enabled),
            purpose="answer",
            structured_input={
                "query": query.text,
                "as_of_time": query.as_of_time.isoformat() if query.as_of_time else None,
                "abstain_on_conflict": self.abstain_on_conflict,
                "evidence": [
                    {
                        "memory_id": item.memory_id,
                        "content": item.content,
                        "confidence": item.confidence,
                        "valid_from": item.valid_from.isoformat() if item.valid_from else None,
                        "valid_to": item.valid_to.isoformat() if item.valid_to else None,
                    }
                    for item in retrieval.evidence
                ],
            },
        )
        response = self.provider.chat(request)
        abstained = response.text.strip().upper().startswith(ABSTAIN_PREFIX)
        reason: AbstentionReason | None = None
        if abstained:
            reason = (
                AbstentionReason.conflicted_evidence
                if "conflict" in response.text.lower()
                else AbstentionReason.no_evidence
            )

        return AnswerRecord(
            query_id=query.query_id,
            query=query.text,
            tenant_id=query.tenant_id,
            checkpoint=checkpoint,
            backend_name=backend_name,
            answer=response.text,
            abstained=abstained,
            abstention_reason=reason,
            evidence_memory_ids=[item.memory_id for item in retrieval.evidence],
            evidence_event_ids=sorted(
                {eid for item in retrieval.evidence for eid in item.source_event_ids}
            ),
            injected_tokens=retrieval.injected_tokens,
            retrieval_latency_ms=retrieval.retrieval_latency_ms,
            answer_latency_ms=(time.perf_counter() - started) * 1000,
            provider_call=response.call,
            metadata={
                "reader_reason": (response.structured or {}).get("reason"),
                "finish_reason": response.finish_reason,
                "empty_completion": not response.text.strip(),
            },
        )
