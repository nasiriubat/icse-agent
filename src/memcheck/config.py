"""YAML experiment configuration, validated by Pydantic.

Unknown keys are rejected (``extra="forbid"``) so a typo in a config file fails loudly instead of
silently changing an experiment's meaning.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from memcheck.models import StrictModel, content_hash

CHECKPOINT_SCHEDULE: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)

CORE_BACKENDS: tuple[str, ...] = (
    "no_memory",
    "full_history",
    "transcript_rag",
    "guarded_memory",
)

KNOWN_ABLATIONS: tuple[str, ...] = (
    "abl_no_idempotency",
    "abl_no_supersession",
    "abl_no_tombstones",
    "abl_no_abstention",
    "abl_no_budget",
    "abl_no_tenant_filter",
)


class ModelConfig(StrictModel):
    provider: Literal["mock", "anthropic", "openai"] = "mock"
    identifier: str = "oracle-reader-v1"
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    max_output_tokens: int = Field(gt=0, default=512)
    context_limit_tokens: int = Field(gt=0, default=8192)
    system_prompt_id: str = "answer_v1"
    tools_enabled: list[str] = Field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.provider}:{self.identifier}"

    @property
    def is_paid(self) -> bool:
        return self.provider != "mock"


class EmbeddingConfig(StrictModel):
    provider: Literal["mock", "openai", "local"] = "mock"
    identifier: str = "hash-embed-v1"
    dimensions: int = Field(gt=0, default=128)


class ExtractionConfig(StrictModel):
    """Which write-side policy decides what is worth remembering.

    Separated from the model config because extraction and answering are different decisions
    that a study may want to vary independently: the point of the factorial is to hold one fixed
    while the other moves.
    """

    policy: Literal["rule_based", "llm"] = "rule_based"
    #: Defaults to the answering model when unset, so a config need not name it twice.
    model_identifier: str | None = None


class RetrievalConfig(StrictModel):
    mode: Literal["lexical", "semantic", "hybrid"] = "lexical"
    evidence_budget: int = Field(gt=0, default=5)
    token_budget: int = Field(gt=0, default=512)
    min_confidence: float = Field(ge=0.0, le=1.0, default=0.35)
    chunk_tokens: int = Field(gt=0, default=120)
    abstain_on_conflict: bool = True


class WorkloadConfig(StrictModel):
    name: Literal["synthetic_longitudinal", "repo_evolution", "longmemeval"]
    version: str = "v1"
    subset: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    data_path: str | None = None


class CostConfig(StrictModel):
    limit_usd: float = Field(ge=0.0, default=0.0)
    pricing_file: str = "configs/pricing.example.yaml"
    max_requests: int = Field(gt=0, default=100_000)


class OutputConfig(StrictModel):
    results_dir: str = "results/raw"
    keep_stdout: bool = True


class ExperimentConfig(StrictModel):
    """One experiment matrix. Expanded into conditions by ``evaluation.matrix``."""

    name: str
    description: str = ""
    backends: list[str] = Field(default_factory=lambda: list(CORE_BACKENDS))
    ablations: list[str] = Field(default_factory=list)
    workload: WorkloadConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    faults: list[str] = Field(default_factory=lambda: ["none"])
    contracts: list[str] = Field(default_factory=lambda: ["all"])
    #: ``fresh``: each contract builds its scenario on an empty store. ``aged``: each contract
    #: builds its scenario on a snapshot of the condition's store taken after the final checkpoint.
    contract_store: Literal["fresh", "aged"] = "fresh"
    checkpoints: list[float] = Field(default_factory=lambda: list(CHECKPOINT_SCHEDULE))
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3])
    repetitions: int = Field(gt=0, default=1)
    # Independent repetition of a whole run: enters every provider cache key when non-zero.
    attempt: int = Field(ge=0, default=0)
    cost: CostConfig = Field(default_factory=CostConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    notes: str = ""
    source_path: str | None = None

    @field_validator("backends")
    @classmethod
    def _known_backends(cls, value: list[str]) -> list[str]:
        unknown = [b for b in value if b not in CORE_BACKENDS]
        if unknown:
            raise ValueError(f"unknown backend(s): {unknown}; known: {list(CORE_BACKENDS)}")
        if not value:
            raise ValueError("at least one backend is required")
        return value

    @field_validator("ablations")
    @classmethod
    def _known_ablations(cls, value: list[str]) -> list[str]:
        unknown = [a for a in value if a not in KNOWN_ABLATIONS]
        if unknown:
            raise ValueError(f"unknown ablation(s): {unknown}; known: {list(KNOWN_ABLATIONS)}")
        return value

    @field_validator("checkpoints")
    @classmethod
    def _valid_checkpoints(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("at least one checkpoint is required")
        for fraction in value:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"checkpoint {fraction} must be in (0, 1]")
        return sorted(set(value))

    @model_validator(mode="after")
    def _ablations_need_guarded(self) -> ExperimentConfig:
        if self.ablations and "guarded_memory" not in self.backends:
            raise ValueError("ablations apply to guarded_memory, which is not in `backends`")
        return self

    @property
    def checkpoint_labels(self) -> list[str]:
        return [f"cp{round(fraction * 100):03d}" for fraction in self.checkpoints]

    def configuration_hash(self) -> str:
        """Hash of everything that changes an experiment's meaning (excludes path/notes)."""
        payload = self.model_dump(mode="json", exclude={"notes", "source_path", "description"})
        return content_hash(payload)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    raw["source_path"] = str(config_path)
    return ExperimentConfig.model_validate(raw)


# --------------------------------------------------------------------------------------
# Pricing (DECISIONS D-015)
# --------------------------------------------------------------------------------------


class ModelPricing(StrictModel):
    input_usd_per_million: float = Field(ge=0.0)
    output_usd_per_million: float = Field(ge=0.0)
    source: str = ""


class PricingTable(StrictModel):
    as_of: str
    currency: Literal["USD"] = "USD"
    note: str = ""
    models: dict[str, ModelPricing] = Field(default_factory=dict)

    def cost_for(self, model_key: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self.models.get(model_key)
        if pricing is None:
            raise KeyError(
                f"no pricing entry for '{model_key}' in the pricing file dated {self.as_of}; "
                "add one before running a paid experiment"
            )
        return (
            input_tokens * pricing.input_usd_per_million / 1_000_000
            + output_tokens * pricing.output_usd_per_million / 1_000_000
        )


def load_pricing(path: str | Path) -> PricingTable:
    pricing_path = Path(path)
    if not pricing_path.exists():
        raise FileNotFoundError(
            f"pricing file not found: {pricing_path}. Copy configs/pricing.example.yaml "
            "and update it with dated, sourced prices before a paid run."
        )
    with pricing_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return PricingTable.model_validate(raw)


# --------------------------------------------------------------------------------------
# Paid-run authorization (Section 26)
# --------------------------------------------------------------------------------------


class PaidRunAuthorization(StrictModel):
    allowed: bool
    max_cost_usd: float
    reason: str


def paid_run_authorization() -> PaidRunAuthorization:
    """Read the two required environment controls. Both must be present and valid."""
    allow_raw = os.environ.get("ALLOW_PAID_RUNS", "").strip().lower()
    limit_raw = os.environ.get("MAX_EXPERIMENT_COST_USD", "").strip()
    if allow_raw != "true":
        return PaidRunAuthorization(
            allowed=False,
            max_cost_usd=0.0,
            reason="ALLOW_PAID_RUNS is not 'true'; paid providers are blocked",
        )
    if not limit_raw:
        return PaidRunAuthorization(
            allowed=False,
            max_cost_usd=0.0,
            reason="MAX_EXPERIMENT_COST_USD is unset; a numeric ceiling is required",
        )
    try:
        limit = float(limit_raw)
    except ValueError:
        return PaidRunAuthorization(
            allowed=False,
            max_cost_usd=0.0,
            reason=f"MAX_EXPERIMENT_COST_USD is not numeric: {limit_raw!r}",
        )
    if limit <= 0:
        return PaidRunAuthorization(
            allowed=False, max_cost_usd=limit, reason="MAX_EXPERIMENT_COST_USD must be positive"
        )
    return PaidRunAuthorization(allowed=True, max_cost_usd=limit, reason="authorized")
