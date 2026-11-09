"""Deterministic fault injection."""

from __future__ import annotations

from memcheck.backends.base import MemoryBackend
from memcheck.faults.base import Fault, FaultContext, NoFault
from memcheck.faults.injectors import (
    ContradictoryUpdate,
    DeletionDuringGrowth,
    DuplicateEventDelivery,
    IndexLoss,
    InterruptedWriteFault,
    OutOfOrderDelivery,
)
from memcheck.models import Event, FaultReport

ALL_FAULTS: tuple[type[Fault], ...] = (
    DuplicateEventDelivery,
    OutOfOrderDelivery,
    InterruptedWriteFault,
    IndexLoss,
    ContradictoryUpdate,
    DeletionDuringGrowth,
)

_REGISTERED: tuple[type[Fault], ...] = (NoFault, *ALL_FAULTS)
FAULT_REGISTRY: dict[str, type[Fault]] = {fault.name: fault for fault in _REGISTERED}

__all__ = [
    "ALL_FAULTS",
    "FAULT_REGISTRY",
    "ContradictoryUpdate",
    "DeletionDuringGrowth",
    "DuplicateEventDelivery",
    "Fault",
    "FaultContext",
    "IndexLoss",
    "InterruptedWriteFault",
    "NoFault",
    "OutOfOrderDelivery",
    "resolve_fault",
    "run_fault",
]


def resolve_fault(name: str) -> Fault:
    if name not in FAULT_REGISTRY:
        raise KeyError(f"unknown fault {name!r}; known: {sorted(FAULT_REGISTRY)}")
    return FAULT_REGISTRY[name]()


def run_fault(
    fault: Fault,
    backend: MemoryBackend,
    events: list[Event],
    *,
    seed: int,
    backend_name: str,
) -> FaultReport:
    """Inject a fault end to end against a live backend and report the recovery observed.

    The sequence is fixed so every fault is measured the same way: rewrite the delivery stream,
    ingest it, perturb live state, attempt recovery, then verify.
    """
    context = FaultContext(seed=seed, backend_name=backend_name)
    delivered = fault.wrap_events(events, context)
    for event in delivered:
        backend.ingest_event(event)
    backend.checkpoint()
    fault.perturb(backend, context)
    fault.recover(backend, context)
    recovered, observed = fault.verify(backend, context)
    return fault.report(context, recovered, observed)
