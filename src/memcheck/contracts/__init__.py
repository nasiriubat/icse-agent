"""The six executable memory contracts and the registry that runs them."""

from __future__ import annotations

from memcheck.contracts.base import EPOCH, Contract, ContractContext, make_event
from memcheck.contracts.deletion_closure import DeletionClosure
from memcheck.contracts.duplicate_replay import DuplicateReplayInvariance
from memcheck.contracts.evidence_budget import EvidenceBudgetCompliance
from memcheck.contracts.restart_equivalence import (
    PROBE_QUERIES,
    RestartEquivalence,
    canonical_state,
    divergence,
)
from memcheck.contracts.supersession import SupersessionCorrectness
from memcheck.contracts.tenant_isolation import TenantIsolation
from memcheck.models import ContractResult

ALL_CONTRACTS: tuple[type[Contract], ...] = (
    DuplicateReplayInvariance,
    RestartEquivalence,
    SupersessionCorrectness,
    DeletionClosure,
    TenantIsolation,
    EvidenceBudgetCompliance,
)

CONTRACT_REGISTRY: dict[str, type[Contract]] = {
    contract.name: contract for contract in ALL_CONTRACTS
}

__all__ = [
    "ALL_CONTRACTS",
    "CONTRACT_REGISTRY",
    "EPOCH",
    "PROBE_QUERIES",
    "Contract",
    "ContractContext",
    "DeletionClosure",
    "DuplicateReplayInvariance",
    "EvidenceBudgetCompliance",
    "RestartEquivalence",
    "SupersessionCorrectness",
    "TenantIsolation",
    "canonical_state",
    "divergence",
    "make_event",
    "resolve_contracts",
    "run_contracts",
]


def resolve_contracts(names: list[str]) -> list[Contract]:
    """Resolve config names to contract instances. ``["all"]`` selects every contract."""
    if not names or names == ["all"]:
        return [contract() for contract in ALL_CONTRACTS]
    resolved: list[Contract] = []
    for name in names:
        if name not in CONTRACT_REGISTRY:
            raise KeyError(f"unknown contract {name!r}; known: {sorted(CONTRACT_REGISTRY)}")
        resolved.append(CONTRACT_REGISTRY[name]())
    return resolved


def run_contracts(contracts: list[Contract], context: ContractContext) -> list[ContractResult]:
    return [contract.run(context) for contract in contracts]
