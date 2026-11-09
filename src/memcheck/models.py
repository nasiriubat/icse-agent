"""Typed, validated domain models for MemCheck.

Every object that crosses a module boundary is one of these. Backends return models, never
printed text (DECISIONS D-005), so the CLI and the experiment runner can consume the same values.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utcnow() -> datetime:
    """Timezone-aware current time. Used everywhere instead of naive ``datetime.now``."""
    return datetime.now(UTC)


def canonical_json(payload: Any) -> str:
    """Stable JSON encoding used for every hash MemCheck computes."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


class StrictModel(BaseModel):
    """Base model: unknown fields are an error, values are validated on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)


# --------------------------------------------------------------------------------------
# 6.1 Event
# --------------------------------------------------------------------------------------


class EventType(StrEnum):
    user_message = "user_message"
    assistant_message = "assistant_message"
    tool_call = "tool_call"
    tool_result = "tool_result"
    document = "document"
    code_change = "code_change"
    feedback = "feedback"
    deletion_request = "deletion_request"
    system_event = "system_event"


class Event(StrictModel):
    """An item entering the memory system. Immutable once appended to the event log."""

    event_id: str
    tenant_id: str
    session_id: str
    event_type: EventType
    event_time: datetime
    ingest_time: datetime = Field(default_factory=utcnow)
    content: str
    source_uri: str | None = None
    source_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "tenant_id", "session_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("identifier fields must be non-empty")
        return value

    @model_validator(mode="after")
    def _fill_source_hash(self) -> Event:
        if not self.source_hash:
            # Direct __dict__ write: normal assignment would re-enter validation
            # because the model is configured with validate_assignment=True.
            self.__dict__["source_hash"] = self.compute_source_hash()
        return self

    def compute_source_hash(self) -> str:
        """Content identity of the event, independent of delivery time or event id.

        Duplicate *delivery* is detected on ``event_id``; duplicate *content* re-delivered under a
        fresh id is detected on this hash. The guarded backend checks both.
        """
        return content_hash(
            {
                "tenant_id": self.tenant_id,
                "session_id": self.session_id,
                "event_type": self.event_type.value,
                "event_time": self.event_time.isoformat(),
                "content": self.content,
                "source_uri": self.source_uri,
                "metadata": self.metadata,
            }
        )


# --------------------------------------------------------------------------------------
# 6.2 MemoryRecord
# --------------------------------------------------------------------------------------


class MemoryStatus(StrEnum):
    active = "active"
    superseded = "superseded"
    deleted = "deleted"
    conflicted = "conflicted"
    invalid = "invalid"


class MemoryRecord(StrictModel):
    """A derived, versioned memory. Never mutated in place except for status transitions."""

    memory_id: str
    tenant_id: str
    canonical_key: str
    version: int = Field(ge=1)
    content: str
    status: MemoryStatus = MemoryStatus.active
    valid_from: datetime
    valid_to: datetime | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    source_event_ids: list[str] = Field(default_factory=list)
    supersedes_memory_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    embedding_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_provenance_and_interval(self) -> MemoryRecord:
        if not self.source_event_ids:
            raise ValueError(
                f"memory {self.memory_id} has no source provenance; "
                "every derived memory must cite at least one event"
            )
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to must not precede valid_from")
        return self

    def is_valid_at(self, when: datetime | None) -> bool:
        """Temporal validity test. ``when=None`` means 'now' (only open intervals qualify)."""
        if when is None:
            return self.valid_to is None
        if when < self.valid_from:
            return False
        return self.valid_to is None or when < self.valid_to


# --------------------------------------------------------------------------------------
# 6.3 Evidence / 6.4 RetrievalResult
# --------------------------------------------------------------------------------------


class Evidence(StrictModel):
    """One retrieved memory, carrying why it was selected and what it costs in tokens."""

    memory_id: str
    source_event_ids: list[str] = Field(default_factory=list)
    content: str
    relevance_score: float = 0.0
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    retrieval_reason: str = ""
    token_count: int = Field(ge=0, default=0)


class AbstentionReason(StrEnum):
    no_evidence = "no_evidence"
    weak_evidence = "weak_evidence"
    deleted_evidence = "deleted_evidence"
    conflicted_evidence = "conflicted_evidence"
    budget_exhausted = "budget_exhausted"
    backend_has_no_memory = "backend_has_no_memory"


class RetrievalResult(StrictModel):
    query: str
    tenant_id: str
    evidence: list[Evidence] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: AbstentionReason | None = None
    retrieval_latency_ms: float = 0.0
    injected_tokens: int = 0
    backend_name: str = ""
    checkpoint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _abstention_consistency(self) -> RetrievalResult:
        if self.abstained and self.abstention_reason is None:
            raise ValueError("an abstaining retrieval must record an abstention_reason")
        return self


# --------------------------------------------------------------------------------------
# 6.5 ContractViolation
# --------------------------------------------------------------------------------------


class Severity(StrEnum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class ContractViolation(StrictModel):
    contract_name: str
    severity: Severity
    backend_name: str
    run_id: str
    event_ids: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    query_id: str | None = None
    expected_behavior: str
    observed_behavior: str
    reproduction_steps: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    fault_name: str | None = None
    condition_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContractResult(StrictModel):
    """Outcome of running one contract against one backend state."""

    contract_name: str
    backend_name: str
    passed: bool
    violations: list[ContractViolation] = Field(default_factory=list)
    checks_run: int = 0
    duration_ms: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None
    #: The scenario did not establish the contract's precondition, so the invariant was not
    #: exercised. Such a result is neither a pass nor a failure and is reported separately.
    vacuous: bool = False
    vacuous_reason: str | None = None
    #: ``fresh``: the scenario ran on an empty store. ``aged``: it ran on a snapshot of the
    #: condition's store taken after the final checkpoint.
    store_state: Literal["fresh", "aged"] = "fresh"


# --------------------------------------------------------------------------------------
# Backend operation results
# --------------------------------------------------------------------------------------


class WriteOutcome(StrEnum):
    created = "created"
    duplicate_ignored = "duplicate_ignored"
    version_created = "version_created"
    conflict_recorded = "conflict_recorded"
    deleted = "deleted"
    rejected = "rejected"
    no_op = "no_op"


class WriteResult(StrictModel):
    """Structured result of ingesting one event."""

    event_id: str
    tenant_id: str
    outcome: WriteOutcome
    created_memory_ids: list[str] = Field(default_factory=list)
    superseded_memory_ids: list[str] = Field(default_factory=list)
    deleted_memory_ids: list[str] = Field(default_factory=list)
    conflicted_memory_ids: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    backend_name: str = ""
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeleteResult(StrictModel):
    tenant_id: str
    target: str
    deleted_memory_ids: list[str] = Field(default_factory=list)
    purged_index_rows: int = 0
    invalidated_summaries: int = 0
    evicted_cache_entries: int = 0
    tombstone_ids: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    backend_name: str = ""


class StoreStatistics(StrictModel):
    """Section 16.5 storage and growth metrics, sampled at every checkpoint."""

    backend_name: str
    tenant_id: str | None = None
    checkpoint: str | None = None
    database_size_bytes: int = 0
    index_size_bytes: int = 0
    event_count: int = 0
    memory_record_count: int = 0
    active_memory_count: int = 0
    superseded_record_count: int = 0
    deleted_record_count: int = 0
    conflicted_record_count: int = 0
    duplicate_ratio: float = 0.0
    sessions_ingested: int = 0
    storage_growth_per_session: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BackendState(StrictModel):
    """Canonical projection of a backend's state, used for restart equivalence (D-007)."""

    records: list[tuple[str, str, int, str, str, str, str | None]] = Field(default_factory=list)
    probe_results: dict[str, list[tuple[str, int]]] = Field(default_factory=dict)
    probe_abstentions: dict[str, bool] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        return content_hash(
            {
                "records": sorted(str(r) for r in self.records),
                "probe_results": {
                    key: [list(pair) for pair in value]
                    for key, value in sorted(self.probe_results.items())
                },
                "probe_abstentions": dict(sorted(self.probe_abstentions.items())),
            }
        )


# --------------------------------------------------------------------------------------
# Provider telemetry
# --------------------------------------------------------------------------------------


class ProviderCall(StrictModel):
    """Section 7: every real provider response records this. Secrets never appear here."""

    provider: str
    model_identifier: str
    request_identifier: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    retry_count: int = 0
    estimated_cost_usd: float = 0.0
    timestamp: datetime = Field(default_factory=utcnow)
    cached: bool = False
    purpose: str = ""


class ChatResponse(StrictModel):
    text: str
    call: ProviderCall
    finish_reason: str = "stop"
    structured: dict[str, Any] | None = None


class EmbeddingResponse(StrictModel):
    vectors: list[list[float]]
    call: ProviderCall


# --------------------------------------------------------------------------------------
# Answers, grading, faults
# --------------------------------------------------------------------------------------


class AnswerRecord(StrictModel):
    query_id: str
    query: str
    tenant_id: str
    checkpoint: str
    backend_name: str
    answer: str
    abstained: bool = False
    abstention_reason: AbstentionReason | None = None
    evidence_memory_ids: list[str] = Field(default_factory=list)
    evidence_event_ids: list[str] = Field(default_factory=list)
    injected_tokens: int = 0
    answer_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    provider_call: ProviderCall | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GradeLabel(StrEnum):
    correct = "correct"
    incorrect = "incorrect"
    stale = "stale"  # answered with a superseded value
    leaked = "leaked"  # answered with deleted or cross-tenant content
    abstained_correctly = "abstained_correctly"
    abstained_wrongly = "abstained_wrongly"
    unsupported = "unsupported"  # answered without supporting evidence


class GradeRecord(StrictModel):
    query_id: str
    checkpoint: str
    backend_name: str
    label: GradeLabel
    correct: bool
    gold_answer: str | None = None
    model_answer: str = ""
    grader: Literal["deterministic", "unit_test", "llm_judge"] = "deterministic"
    judge_rationale: str | None = None
    evidence_recall: float | None = None
    evidence_precision: float | None = None
    expected_memory_ids: list[str] = Field(default_factory=list)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FaultReport(StrictModel):
    """Section 13: what a fault did and whether the backend recovered as expected."""

    fault_name: str
    injection_point: str
    activation_condition: str
    affected_components: list[str] = Field(default_factory=list)
    expected_recovery_behavior: str
    observed_recovery_behavior: str
    recovered: bool = False
    seed: int = 0
    backend_name: str = ""
    events_affected: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# 6.6 ExperimentRunManifest
# --------------------------------------------------------------------------------------


class ExperimentRunManifest(StrictModel):
    run_id: str
    timestamp: datetime = Field(default_factory=utcnow)
    git_commit: str | None = None
    working_tree_status: str = "unknown"
    operating_system: str = ""
    python_version: str = ""
    dependency_lock_hash: str = ""
    cpu_information: str = ""
    gpu_information: str = "none"
    backend: str = ""
    model_provider: str = ""
    model_identifier: str = ""
    embedding_identifier: str = "none"
    temperature: float = 0.0
    random_seeds: list[int] = Field(default_factory=list)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    benchmark_name: str = ""
    benchmark_version: str = ""
    #: Hash of the workload's subset and generation parameters. Two runs of the same benchmark
    #: at different sizes or failure densities are *not* comparable, and only this field
    #: distinguishes them.
    benchmark_parameters_hash: str = ""
    configuration_hash: str = ""
    token_budget: int = 0
    evidence_budget: int = 0
    cost_limit_usd: float = 0.0
    checkpoint_schedule: list[str] = Field(default_factory=list)
    repetition_number: int = 1
    attempt: int = 0
    #: Whether the contracts ran on fresh scenario stores or on snapshots of the grown stores.
    contract_store: Literal["fresh", "aged"] = "fresh"
    condition_id: str = ""
    retrieval_mode: str = "lexical"
    #: Which write-side policy decided what to remember. A run using a different extractor is
    #: not another sample of the same experiment, so this is a comparability field.
    extractor_identity: str = "rule_copula_v1"
    ablations: list[str] = Field(default_factory=list)
    faults: list[str] = Field(default_factory=list)
    memcheck_version: str = __import__("memcheck").__version__
    notes: str = ""


class RunStatus(StrictModel):
    run_id: str
    state: Literal["running", "completed", "failed", "partial"] = "running"
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    conditions_total: int = 0
    conditions_completed: int = 0
    conditions_failed: int = 0
    error: str | None = None
    completed_condition_ids: list[str] = Field(default_factory=list)
    failed_condition_ids: list[str] = Field(default_factory=list)
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
