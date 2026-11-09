"""Memory backends and the factory that builds them from a configuration."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from memcheck.backends.base import BackendError, MemoryBackend
from memcheck.backends.baselines import (
    FullHistoryBackend,
    NoMemoryBackend,
    TranscriptRAGBackend,
)
from memcheck.backends.extraction import (
    Candidate,
    Extractor,
    LLMExtractor,
    RuleBasedExtractor,
    build_extractor,
)
from memcheck.backends.guarded import GuardedMemoryBackend, InterruptedWrite
from memcheck.backends.retrieval import RANKER_IDENTITY, LexicalIndex
from memcheck.config import RetrievalConfig

__all__ = [
    "RANKER_IDENTITY",
    "BackendError",
    "Candidate",
    "Extractor",
    "FullHistoryBackend",
    "GuardedMemoryBackend",
    "InterruptedWrite",
    "LLMExtractor",
    "LexicalIndex",
    "MemoryBackend",
    "NoMemoryBackend",
    "RuleBasedExtractor",
    "TranscriptRAGBackend",
    "build_backend",
    "build_extractor",
]


def build_backend(
    name: str,
    path: str | Path,
    retrieval: RetrievalConfig,
    *,
    ablations: Iterable[str] = (),
    initialize: bool = True,
    extractor: Extractor | None = None,
) -> MemoryBackend:
    """Construct a backend. Every condition goes through this one function so budgets,
    thresholds, and paths cannot drift between conditions in a comparison."""
    ablation_list = list(ablations)
    backend: MemoryBackend
    if name == "no_memory":
        backend = NoMemoryBackend(
            path, evidence_budget=retrieval.evidence_budget, token_budget=retrieval.token_budget
        )
    elif name == "full_history":
        backend = FullHistoryBackend(
            path, evidence_budget=retrieval.evidence_budget, token_budget=retrieval.token_budget
        )
    elif name == "transcript_rag":
        backend = TranscriptRAGBackend(
            path,
            evidence_budget=retrieval.evidence_budget,
            token_budget=retrieval.token_budget,
            chunk_tokens=retrieval.chunk_tokens,
        )
    elif name == "guarded_memory":
        backend = GuardedMemoryBackend(
            path,
            evidence_budget=retrieval.evidence_budget,
            token_budget=retrieval.token_budget,
            min_confidence=retrieval.min_confidence,
            abstain_on_conflict=retrieval.abstain_on_conflict,
            ablations=ablation_list,
            extractor=extractor,
        )
    else:
        raise BackendError(f"unknown backend {name!r}")

    if ablation_list and name != "guarded_memory":
        raise BackendError(f"ablations are only defined for guarded_memory, not {name!r}")
    if extractor is not None and name != "guarded_memory":
        # The baselines store transcripts or nothing at all; handing them an extractor would
        # imply a write policy they do not have, and quietly ignoring it would make a
        # configuration mean something different depending on the backend.
        raise BackendError(f"only guarded_memory extracts memories, so {name!r} takes no extractor")
    if initialize:
        backend.initialize()
    return backend
