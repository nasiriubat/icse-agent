"""Pre-run cost estimation and enforcement (Section 26)."""

from __future__ import annotations

from dataclasses import dataclass

from memcheck.config import ExperimentConfig, load_pricing, paid_run_authorization
from memcheck.evaluation.matrix import Condition
from memcheck.providers.prompts import get_prompt
from memcheck.utilities.tokens import count_tokens


class CostLimitExceeded(RuntimeError):
    """Raised when an estimate exceeds the configured ceiling."""


@dataclass
class CostEstimate:
    requests: int
    input_tokens: int
    output_tokens: int
    usd: float
    limit_usd: float
    pricing_as_of: str
    is_paid: bool
    within_limit: bool
    notes: list[str]

    def render(self) -> str:
        lines = [
            "Cost estimate",
            f"  provider          : {'paid' if self.is_paid else 'mock (free)'}",
            f"  requests          : {self.requests:,}",
            f"  input tokens      : {self.input_tokens:,}",
            f"  output tokens     : {self.output_tokens:,}",
            f"  estimated cost    : ${self.usd:,.4f}",
            f"  configured limit  : ${self.limit_usd:,.4f}",
            f"  pricing as of     : {self.pricing_as_of}",
        ]
        lines.extend(f"  note              : {note}" for note in self.notes)
        return "\n".join(lines)


def estimate_for_workload(
    config: ExperimentConfig, conditions: list[Condition], workload: object
) -> CostEstimate:
    """Estimate from a loaded workload, so the runner and ``--dry-run`` cannot disagree.

    Both callers previously assembled the arguments themselves, which meant a new cost term had
    to be remembered in two places --- and the one that was forgotten was the dry run, the only
    path whose entire purpose is to be right before money is spent.
    """
    queries = workload.queries()  # type: ignore[attr-defined]
    events = workload.events()  # type: ignore[attr-defined]
    return estimate(
        config,
        conditions,
        query_count=len(queries),
        event_count=len(events),
        average_event_tokens=(
            sum(count_tokens(event.content) for event in events) // len(events) if events else 0
        ),
    )


def estimate(
    config: ExperimentConfig,
    conditions: list[Condition],
    *,
    query_count: int,
    event_count: int = 0,
    average_event_tokens: int = 0,
    average_evidence_tokens: int | None = None,
) -> CostEstimate:
    """Estimate requests, tokens, and money before anything is spent.

    An LLM write policy is charged once per *event*, and a longitudinal workload carries far
    more events than questions, so extraction can dominate a run's cost. Omitting it would make
    the estimate an underestimate of the thing the cost limit exists to bound.
    """
    notes: list[str] = []
    checkpoints = len(config.checkpoints)
    evidence_tokens = (
        average_evidence_tokens
        if average_evidence_tokens is not None
        else config.retrieval.token_budget
    )
    system_tokens = count_tokens(get_prompt(config.model.system_prompt_id))
    # One answer request per query per checkpoint per condition, plus scaffolding tokens.
    requests = len(conditions) * checkpoints * query_count
    per_request_input = system_tokens + evidence_tokens + 48
    input_tokens = requests * per_request_input
    output_tokens = requests * min(32, config.model.max_output_tokens)

    if config.extraction.policy == "llm":
        # Only the guarded backend extracts, and it does so once per event per condition. The
        # event stream is replayed in full for every condition, so this scales with conditions.
        extracting = sum(1 for condition in conditions if condition.backend == "guarded_memory")
        extraction_requests = extracting * event_count
        extraction_system = count_tokens(get_prompt("extract_v1"))
        requests += extraction_requests
        input_tokens += extraction_requests * (extraction_system + average_event_tokens)
        # A fact list is short, but the model may emit several; budget generously rather than
        # discovering the shortfall halfway through a paid run.
        output_tokens += extraction_requests * 96
        if extraction_requests:
            notes.append(
                f"{extraction_requests:,} of these requests are write-side extraction, one per "
                f"event across {extracting} guarded condition(s)"
            )

    if not config.model.is_paid:
        return CostEstimate(
            requests=requests,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=0.0,
            limit_usd=config.cost.limit_usd,
            pricing_as_of="n/a (mock provider)",
            is_paid=False,
            within_limit=True,
            notes=["mock provider: no network calls and no monetary cost"],
        )

    pricing = load_pricing(config.cost.pricing_file)
    usd = pricing.cost_for(config.model.qualified_name, input_tokens, output_tokens)
    authorization = paid_run_authorization()
    limit = min(config.cost.limit_usd, authorization.max_cost_usd) if authorization.allowed else 0.0
    if not authorization.allowed:
        notes.append(f"paid runs are not authorized: {authorization.reason}")
    if requests > config.cost.max_requests:
        notes.append(
            f"request count {requests:,} exceeds cost.max_requests={config.cost.max_requests:,}"
        )
    notes.append(
        "estimate is list-price and excludes batch discounts, cache reads, and retries; "
        "treat it as an upper bound"
    )
    within = authorization.allowed and usd <= limit and requests <= config.cost.max_requests
    return CostEstimate(
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=usd,
        limit_usd=limit,
        pricing_as_of=pricing.as_of,
        is_paid=True,
        within_limit=within,
        notes=notes,
    )


def enforce(estimate_result: CostEstimate) -> None:
    """Refuse to continue when the estimate is over the limit or the run is unauthorized."""
    if estimate_result.within_limit:
        return
    raise CostLimitExceeded(
        f"refusing to start: estimated ${estimate_result.usd:,.4f} against a limit of "
        f"${estimate_result.limit_usd:,.4f}. " + " ".join(estimate_result.notes)
    )
