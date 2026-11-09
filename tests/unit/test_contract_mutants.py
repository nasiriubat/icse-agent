"""Seeded-defect tests: every contract must fail on a backend that has its defect.

A contract that passes on every backend it meets may be checking nothing. Each test here pairs
one contract with a backend that breaks exactly the property the contract names, and requires
the contract to report a violation, while the guarded backend passes the same contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memcheck.backends import build_backend
from memcheck.backends.base import MemoryBackend
from memcheck.backends.baselines import TranscriptRAGBackend
from memcheck.backends.extraction import build_extractor
from memcheck.config import RetrievalConfig
from memcheck.contracts import ContractContext
from memcheck.contracts.deletion_closure import DeletionClosure
from memcheck.contracts.duplicate_replay import DuplicateReplayInvariance
from memcheck.contracts.evidence_budget import EvidenceBudgetCompliance
from memcheck.contracts.restart_equivalence import RestartEquivalence
from memcheck.contracts.supersession import SupersessionCorrectness
from memcheck.contracts.tenant_isolation import TenantIsolation
from memcheck.models import Event, WriteOutcome, WriteResult


class _NonIdempotentTranscripts(TranscriptRAGBackend):
    """Indexes every delivery, re-deliveries included: at-least-once without deduplication."""

    def ingest_event(self, event: Event) -> WriteResult:
        self.log.append(event, detect_duplicates=False)
        delivery = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        redelivered = event.model_copy(update={"event_id": f"{event.event_id}_d{delivery}"})
        chunk_ids = self._chunk_and_index(redelivered)
        self.connection.commit()
        return WriteResult(
            event_id=event.event_id,
            tenant_id=event.tenant_id,
            outcome=WriteOutcome.created,
            created_memory_ids=chunk_ids,
            backend_name=self.name,
        )


class _IndexLostOnRestart(TranscriptRAGBackend):
    """Keeps its lexical index in a table it wipes on restart and never rebuilds."""

    def restart(self) -> None:
        super().restart()
        self.index.clear()
        self.connection.commit()


Factory = object


def _context(name: str, build: Factory) -> ContractContext:
    return ContractContext(
        run_id="mutant",
        backend_name=name,
        backend_factory=build,  # type: ignore[arg-type]
        seed=1,
    )


@pytest.fixture
def guarded_with(tmp_path: Path, retrieval: RetrievalConfig) -> object:
    def factory(ablations: list[str]) -> Factory:
        def build(suffix: str) -> MemoryBackend:
            return build_backend(
                "guarded_memory",
                tmp_path / f"guarded_{'_'.join(ablations)}_{suffix}.db",
                retrieval,
                ablations=ablations,
                extractor=build_extractor("rule_based"),
            )

        return build

    return factory


@pytest.fixture
def custom(tmp_path: Path, retrieval: RetrievalConfig) -> object:
    def factory(cls: type[TranscriptRAGBackend]) -> Factory:
        def build(suffix: str) -> MemoryBackend:
            backend = cls(
                tmp_path / f"{cls.__name__}_{suffix}.db",
                evidence_budget=retrieval.evidence_budget,
                token_budget=retrieval.token_budget,
                chunk_tokens=retrieval.chunk_tokens,
            )
            backend.initialize()
            return backend

        return build

    return factory


@pytest.mark.parametrize(
    ("contract", "ablation"),
    [
        (SupersessionCorrectness, "abl_no_supersession"),
        (DeletionClosure, "abl_no_tombstones"),
        (TenantIsolation, "abl_no_tenant_filter"),
        (EvidenceBudgetCompliance, "abl_no_budget"),
    ],
)
def test_contract_fails_on_the_ablation_that_removes_its_guard(
    contract: type, ablation: str, guarded_with: object
) -> None:
    broken = contract().run(_context("guarded_memory", guarded_with([ablation])))  # type: ignore[operator]
    assert not broken.vacuous, broken.vacuous_reason
    assert not broken.passed
    assert broken.violations
    intact = contract().run(_context("guarded_memory", guarded_with([])))  # type: ignore[operator]
    assert intact.passed
    assert not intact.vacuous


def test_duplicate_replay_fails_on_a_store_that_indexes_every_delivery(custom: object) -> None:
    build = custom(_NonIdempotentTranscripts)  # type: ignore[operator]
    result = DuplicateReplayInvariance().run(_context("transcript_rag", build))
    assert not result.passed
    assert result.violations


def test_restart_equivalence_fails_on_a_store_that_loses_its_index(custom: object) -> None:
    build = custom(_IndexLostOnRestart)  # type: ignore[operator]
    result = RestartEquivalence().run(_context("transcript_rag", build))
    assert not result.passed
    assert result.violations


def test_budget_contract_reports_vacuity_rather_than_a_pass_on_an_empty_store(
    tmp_path: Path, retrieval: RetrievalConfig
) -> None:
    def build(suffix: str) -> MemoryBackend:
        return build_backend("no_memory", tmp_path / f"none_{suffix}.db", retrieval)

    result = EvidenceBudgetCompliance().run(_context("no_memory", build))
    assert result.vacuous
    assert not result.passed
    assert not result.violations
