"""Common workload interface and the deterministic grader.

A workload supplies a chronological event stream, a query set with gold labels, optional coding
tasks, and a checkpoint schedule. Grading is deterministic wherever possible (Section 18): the
grader compares normalized values and classifies *why* an answer is wrong — stale, leaked,
unsupported, or wrongly abstained — because "incorrect" alone cannot answer RQ2.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from memcheck.models import (
    AnswerRecord,
    Event,
    GradeLabel,
    GradeRecord,
    RetrievalResult,
    StrictModel,
)
from memcheck.providers.mock import normalize_value

QUERY_CATEGORIES = (
    "single_session_extraction",
    "cross_session_reasoning",
    "temporal_reasoning",
    "knowledge_update",
    "abstention",
)


class Query(StrictModel):
    """One graded question, with everything the deterministic grader needs."""

    query_id: str
    tenant_id: str
    text: str
    category: str = "single_session_extraction"
    gold_answer: str | None = None
    expects_abstention: bool = False
    #: Values that were once true but are not any more. Answering one is a *stale* error.
    stale_values: list[str] = Field(default_factory=list)
    #: Values that must never be answered: deleted content or another tenant's content.
    forbidden_values: list[str] = Field(default_factory=list)
    #: Events whose content supports the gold answer, used for evidence recall/precision.
    expected_event_ids: list[str] = Field(default_factory=list)
    #: Which grader is sound for this query. Deterministic containment is exact when the gold
    #: answer is a single value, as in the synthetic and repository workloads. It is *not* sound
    #: for sentence-length free-text golds, where semantic equivalence is required.
    grader: Literal["deterministic", "llm_judge"] = "deterministic"
    as_of_time: datetime | None = None
    #: The query is only asked at checkpoints at or after this fraction of the stream.
    available_from_fraction: float = 0.0
    #: When set, the query is asked at exactly this checkpoint and no other: its gold answer is
    #: the state of the world as of that checkpoint, so asking it later would grade a stale gold.
    asked_at_fraction: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def query_is_asked(query: Query, checkpoint: Checkpoint) -> bool:
    """Whether ``query`` is graded at ``checkpoint``."""
    if query.asked_at_fraction is not None:
        return abs(query.asked_at_fraction - checkpoint.fraction) < 1e-9
    return query.available_from_fraction <= checkpoint.fraction + 1e-9


@dataclass
class Checkpoint:
    """A longitudinal measurement point."""

    label: str
    fraction: float
    event_index: int  # number of events ingested when the checkpoint fires


class Workload(abc.ABC):
    """Chronological interactions plus a graded query set."""

    name: str = "abstract"
    version: str = "v1"
    #: Whether the seed changes the stream. When it does, each seed is its own sample and the
    #: runner builds one stream per seed; when it does not, extra seeds are replicas.
    seed_sensitive: bool = False

    @abc.abstractmethod
    def load(self) -> None:
        """Load or generate the interactions. Must be deterministic given the same seed."""

    @abc.abstractmethod
    def events(self) -> list[Event]:
        """Interactions in chronological order."""

    @abc.abstractmethod
    def queries(self) -> list[Query]:
        """The graded query set."""

    def coding_tasks(self) -> list[Any]:
        """Coding tasks, if the workload has any (RepoEvolutionBench does)."""
        return []

    def tenants(self) -> list[str]:
        return sorted({event.tenant_id for event in self.events()})

    def checkpoints(self, fractions: list[float]) -> list[Checkpoint]:
        total = len(self.events())
        result: list[Checkpoint] = []
        for fraction in fractions:
            index = max(1, round(total * fraction))
            result.append(
                Checkpoint(
                    label=f"cp{round(fraction * 100):03d}",
                    fraction=fraction,
                    event_index=min(index, total),
                )
            )
        return result

    def queries_at(self, checkpoint: Checkpoint) -> list[Query]:
        return [query for query in self.queries() if query_is_asked(query, checkpoint)]

    def describe(self) -> dict[str, Any]:
        """Metadata recorded in the run manifest."""
        return {
            "name": self.name,
            "version": self.version,
            "event_count": len(self.events()),
            "query_count": len(self.queries()),
            "tenants": self.tenants(),
        }


# --------------------------------------------------------------------------------------
# Deterministic grading
# --------------------------------------------------------------------------------------


#: A model answer contained *inside* the gold answer must still carry most of its content, or a
#: one-character reply would match a sentence-long gold answer.
MIN_CONTAINED_FRACTION = 0.5


def _values_match(model_answer: str, gold: str) -> bool:
    """Whether a normalized model answer matches a normalized gold answer.

    Containment in either direction counts, because a gold value may be quoted inside a longer
    reply ("the retention period is thirty-days") or a reply may be the bare value where gold is
    a phrase. The second direction is guarded by length: without that, any short string is a
    substring of a long gold answer.
    """
    if not model_answer or not gold:
        return False
    if gold in model_answer:
        return True
    if model_answer in gold:
        return len(model_answer) >= MIN_CONTAINED_FRACTION * len(gold)
    return False


#: A judge maps (query, gold, model answer) to (label, rationale).
JudgeFn = Callable[[Query, str, str], tuple[str, str]]


def grade_answer(
    query: Query,
    answer: AnswerRecord,
    retrieval: RetrievalResult,
    judge: JudgeFn | None = None,
) -> GradeRecord:
    """Classify an answer. The label says *how* it failed, not merely that it did.

    Leakage, staleness, abstention, and emptiness are decided deterministically, because those
    are exact tests. Only the correct/incorrect decision may be delegated to a judge, and only
    for queries whose gold answer is free text (``query.grader == "llm_judge"``).
    """
    model_answer = normalize_value(answer.answer)
    abstained = answer.abstained or model_answer.startswith("abstain")

    retrieved_event_ids = {
        event_id for item in retrieval.evidence for event_id in item.source_event_ids
    }
    expected_event_ids = set(query.expected_event_ids)
    evidence_recall: float | None = None
    evidence_precision: float | None = None
    if expected_event_ids:
        evidence_recall = len(retrieved_event_ids & expected_event_ids) / len(expected_event_ids)
    if retrieved_event_ids:
        evidence_precision = len(retrieved_event_ids & expected_event_ids) / len(
            retrieved_event_ids
        )

    used_judge = False
    rationale: str | None = None

    def record(label: GradeLabel, correct: bool) -> GradeRecord:
        return GradeRecord(
            query_id=query.query_id,
            checkpoint=answer.checkpoint,
            backend_name=answer.backend_name,
            label=label,
            correct=correct,
            gold_answer=query.gold_answer,
            model_answer=answer.answer,
            grader="llm_judge" if used_judge else "deterministic",
            judge_rationale=rationale,
            evidence_recall=evidence_recall,
            evidence_precision=evidence_precision,
            expected_memory_ids=query.expected_event_ids,
            retrieved_memory_ids=sorted(retrieved_event_ids),
            metadata={
                "category": query.category,
                "abstention_reason": (
                    answer.abstention_reason.value if answer.abstention_reason else None
                ),
            },
        )

    # A forbidden value is the most serious outcome and is checked first: answering deleted or
    # cross-tenant content is a leak regardless of what else the answer contains.
    for forbidden in query.forbidden_values:
        if forbidden and normalize_value(forbidden) in model_answer:
            return record(GradeLabel.leaked, correct=False)

    if query.expects_abstention:
        return record(
            GradeLabel.abstained_correctly if abstained else GradeLabel.unsupported,
            correct=abstained,
        )

    if abstained:
        return record(GradeLabel.abstained_wrongly, correct=False)

    for stale in query.stale_values:
        if stale and normalize_value(stale) in model_answer:
            return record(GradeLabel.stale, correct=False)

    # An empty response is never correct. Reasoning models can consume their whole output budget
    # on reasoning and return no content at all; with a naive containment test the empty string
    # is a substring of every gold answer, so 78 of 177 such answers were being scored correct.
    if not model_answer:
        return record(GradeLabel.unsupported, correct=False)

    gold = normalize_value(query.gold_answer or "")

    if query.grader == "llm_judge" and judge is not None and query.gold_answer:
        used_judge = True
        verdict, rationale = judge(query, query.gold_answer, answer.answer)
        if verdict == "correct":
            return record(GradeLabel.correct, correct=True)
        if not retrieval.evidence:
            return record(GradeLabel.unsupported, correct=False)
        return record(GradeLabel.incorrect, correct=False)

    if gold and _values_match(model_answer, gold):
        return record(GradeLabel.correct, correct=True)

    # An answer with no supporting evidence at all is unsupported rather than merely incorrect.
    if not retrieval.evidence:
        return record(GradeLabel.unsupported, correct=False)
    return record(GradeLabel.incorrect, correct=False)
