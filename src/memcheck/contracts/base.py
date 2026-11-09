"""Executable memory contracts.

A contract is code, not prose. Each one carries the fields Section 12 requires — name,
description, preconditions, test generator, expected invariant, violation detector, severity,
reproduction trace — and each returns a :class:`~memcheck.models.ContractResult` whose
violations are reproducible from the recorded steps alone.

Contracts build their own state on a backend obtained from the context's factory, so the same
contract runs unchanged against every backend. The factory normally returns an empty store; a
run configured with ``contract_store: aged`` returns a snapshot of the condition's grown store
instead, and the result records which one it was.
"""

from __future__ import annotations

import abc
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

from memcheck.backends.base import MemoryBackend
from memcheck.models import (
    ContractResult,
    ContractViolation,
    Event,
    EventType,
    Severity,
)

#: Fixed epoch so generated scenarios are byte-identical between runs.
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass
class ContractContext:
    """Everything a contract needs to build and probe a backend."""

    run_id: str
    backend_name: str
    backend_factory: Callable[[str], MemoryBackend]
    seed: int = 0
    evidence_budget: int = 5
    token_budget: int = 512
    fault_name: str | None = None
    condition_id: str | None = None
    store_state: Literal["fresh", "aged"] = "fresh"
    steps: list[str] = field(default_factory=list)

    def rng(self) -> random.Random:
        return random.Random(self.seed)

    def fresh_backend(self, suffix: str = "") -> MemoryBackend:
        return self.backend_factory(suffix)


def make_event(
    index: int,
    content: str,
    *,
    tenant_id: str = "tenant_a",
    session_id: str = "session_01",
    event_type: EventType = EventType.user_message,
    days: int | None = None,
    metadata: dict[str, object] | None = None,
    event_id: str | None = None,
) -> Event:
    """Deterministic event factory shared by every contract and fault."""
    return Event(
        event_id=event_id or f"evt_{index:04d}",
        tenant_id=tenant_id,
        session_id=session_id,
        event_type=event_type,
        event_time=EPOCH + timedelta(days=index if days is None else days),
        ingest_time=EPOCH + timedelta(days=index if days is None else days),
        content=content,
        metadata=dict(metadata or {}),
    )


class PreconditionUnmet(Exception):
    """Raised by a contract whose scenario failed to establish its own precondition.

    Without this, a scenario that never exercised the invariant reports a pass: an evidence
    budget cannot be exceeded by a store that holds nothing.
    """


class Contract(abc.ABC):
    """One executable invariant."""

    name: str = "abstract"
    description: str = ""
    severity: Severity = Severity.high

    @property
    @abc.abstractmethod
    def preconditions(self) -> list[str]:
        """What must hold for the contract to be meaningful against a backend."""

    @property
    @abc.abstractmethod
    def expected_invariant(self) -> str:
        """The property in one sentence, as it appears in the paper."""

    def applies_to(self, backend_name: str) -> tuple[bool, str]:
        """Whether the contract is meaningful for this backend. Default: always."""
        return True, ""

    @abc.abstractmethod
    def execute(self, context: ContractContext) -> list[ContractViolation]:
        """Generate the scenario, check the invariant, and return any violations."""

    def run(self, context: ContractContext) -> ContractResult:
        started = time.perf_counter()
        applies, reason = self.applies_to(context.backend_name)
        if not applies:
            return ContractResult(
                contract_name=self.name,
                backend_name=context.backend_name,
                passed=True,
                skipped=True,
                skip_reason=reason,
                duration_ms=(time.perf_counter() - started) * 1000,
                store_state=context.store_state,
            )
        context.steps = []
        try:
            violations = self.execute(context)
        except PreconditionUnmet as unmet:
            return ContractResult(
                contract_name=self.name,
                backend_name=context.backend_name,
                passed=False,
                vacuous=True,
                vacuous_reason=str(unmet),
                checks_run=len(context.steps),
                duration_ms=(time.perf_counter() - started) * 1000,
                store_state=context.store_state,
            )
        return ContractResult(
            contract_name=self.name,
            backend_name=context.backend_name,
            passed=not violations,
            violations=violations,
            checks_run=max(1, len(context.steps)),
            duration_ms=(time.perf_counter() - started) * 1000,
            store_state=context.store_state,
        )

    def violation(
        self,
        context: ContractContext,
        *,
        expected: str,
        observed: str,
        event_ids: list[str] | None = None,
        memory_ids: list[str] | None = None,
        query_id: str | None = None,
        severity: Severity | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ContractViolation:
        return ContractViolation(
            contract_name=self.name,
            severity=severity or self.severity,
            backend_name=context.backend_name,
            run_id=context.run_id,
            event_ids=event_ids or [],
            memory_ids=memory_ids or [],
            query_id=query_id,
            expected_behavior=expected,
            observed_behavior=observed,
            reproduction_steps=list(context.steps),
            fault_name=context.fault_name,
            condition_id=context.condition_id,
            metadata=dict(metadata or {}),
        )
