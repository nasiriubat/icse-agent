"""Composable fault-injection framework.

A fault has two independent hooks, so faults compose without knowing about each other:

* :meth:`Fault.wrap_events` — rewrites the delivery stream (duplication, reordering, injected
  contradictions, injected deletions).
* :meth:`Fault.perturb` — damages live state at a checkpoint (an interrupted write, a lost index).

Determinism comes from an explicit ``random.Random(seed)`` handed to every hook. No module-level
RNG is used anywhere in this package (DECISIONS D-010).
"""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field

from memcheck.backends.base import MemoryBackend
from memcheck.models import Event, FaultReport


@dataclass
class FaultContext:
    """State shared between a fault's hooks during one run."""

    seed: int
    backend_name: str
    notes: list[str] = field(default_factory=list)
    events_affected: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def rng(self) -> random.Random:
        return random.Random(self.seed)


class Fault(abc.ABC):
    """One deterministic, replayable fault."""

    name: str = "abstract"
    injection_point: str = ""
    activation_condition: str = ""
    affected_components: tuple[str, ...] = ()
    expected_recovery_behavior: str = ""

    def wrap_events(self, events: list[Event], context: FaultContext) -> list[Event]:
        """Rewrite the delivery stream. Default: deliver unchanged."""
        return list(events)

    def perturb(self, backend: MemoryBackend, context: FaultContext) -> None:
        """Damage live state. Default: no-op."""
        return None

    def recover(self, backend: MemoryBackend, context: FaultContext) -> None:
        """Take the recovery action a system operator would take. Default: no-op."""
        return None

    @abc.abstractmethod
    def verify(self, backend: MemoryBackend, context: FaultContext) -> tuple[bool, str]:
        """Return ``(recovered, observed_recovery_behavior)`` after recovery."""

    def report(self, context: FaultContext, recovered: bool, observed: str) -> FaultReport:
        return FaultReport(
            fault_name=self.name,
            injection_point=self.injection_point,
            activation_condition=self.activation_condition,
            affected_components=list(self.affected_components),
            expected_recovery_behavior=self.expected_recovery_behavior,
            observed_recovery_behavior=observed,
            recovered=recovered,
            seed=context.seed,
            backend_name=context.backend_name,
            events_affected=list(context.events_affected),
            metadata={**context.metadata, "notes": list(context.notes)},
        )


class NoFault(Fault):
    """The control condition. Present so 'no fault' is an explicit, recorded condition."""

    name = "none"
    injection_point = "none"
    activation_condition = "never"
    affected_components = ()
    expected_recovery_behavior = "nothing to recover from"

    def verify(self, backend: MemoryBackend, context: FaultContext) -> tuple[bool, str]:
        return True, "no fault was injected"
