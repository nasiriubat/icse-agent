"""RepoEvolutionBench — a compact multi-session software-engineering workload.

Each scenario is a small Python repository plus a chronological series of sessions in which a
project decision is made and later *changed*. At the checkpoint, the agent is asked to apply a
change that depends on knowing the project's **current** decision, and the result is graded by
running the repository's own unit tests.

That makes the downstream effect in RQ2 directly observable: if the memory system hands the
agent a superseded decision, the patch encodes the old value and the test fails. Nothing about
the grade depends on a judge model.

Scope note, stated plainly: these tasks are *controlled probes* of whether current project state
reached the patch, not a general software-engineering benchmark. Each edit is small by design so
that a failure is attributable to memory rather than to coding ability. This is repeated in the
threats to validity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from memcheck.models import Event, EventType, StrictModel
from memcheck.workloads.base import Query, Workload

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class CodingTask(StrictModel):
    """One graded coding task, resolved against retrieved memory and graded by unit tests."""

    task_id: str
    tenant_id: str
    scenario: str
    instruction: str
    #: The question whose answer the patch must encode; asked against memory.
    memory_query: str
    gold_value: str
    stale_values: list[str] = []
    forbidden_values: list[str] = []
    expected_event_ids: list[str] = []
    #: Files the agent may edit, with ``{{VALUE}}`` standing for the value it must resolve.
    edit_templates: dict[str, str] = {}
    #: Repository contents before the patch.
    repository: dict[str, str] = {}
    test_command: tuple[str, ...] = ("-m", "pytest", "-q", "-p", "no:cacheprovider", "tests")
    available_from_fraction: float = 1.0


@dataclass(frozen=True)
class Scenario:
    """A scenario definition: what changed, where it lives, and how it is verified."""

    key: str
    title: str
    attribute: str
    old_value: str
    new_value: str
    module: str
    symbol: str
    narrative: tuple[str, ...]
    deleted: bool = False


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="api_rename",
        title="An API method is renamed",
        attribute="public fetch method name",
        old_value="fetch_data",
        new_value="load_records",
        module="app/api_names.py",
        symbol="PUBLIC_FETCH_METHOD",
        narrative=(
            "We are designing the public client surface for the records service.",
            "The public fetch method name is fetch_data.",
            "Review feedback: fetch_data is ambiguous next to fetch_page.",
            "The public fetch method name is load_records.",
        ),
    ),
    Scenario(
        key="config_format",
        title="A configuration format changes",
        attribute="configuration file format",
        old_value="ini",
        new_value="yaml",
        module="app/config_format.py",
        symbol="CONFIG_FORMAT",
        narrative=(
            "We need to settle how service configuration is stored on disk.",
            "The configuration file format is ini.",
            "Operations asked for nested structures that ini cannot express.",
            "The configuration file format is yaml.",
        ),
    ),
    Scenario(
        key="dependency_version",
        title="A dependency version changes",
        attribute="pinned client library version",
        old_value="2.4.1",
        new_value="3.1.0",
        module="app/dependency_pin.py",
        symbol="PINNED_CLIENT_VERSION",
        narrative=(
            "We must pin the client library so builds are reproducible.",
            "The pinned client library version is 2.4.1.",
            "A security advisory affects the 2.x line.",
            "The pinned client library version is 3.1.0.",
        ),
    ),
    Scenario(
        key="rejected_decision",
        title="A design decision is rejected",
        attribute="chosen transport protocol",
        old_value="grpc",
        new_value="http-json",
        module="app/transport.py",
        symbol="CHOSEN_TRANSPORT",
        narrative=(
            "We are choosing a transport for the integration boundary.",
            "The chosen transport protocol is grpc.",
            "The architecture board rejected grpc because partners cannot terminate http2.",
            "The chosen transport protocol is http-json.",
        ),
    ),
    Scenario(
        key="convention_introduced",
        title="A coding convention is introduced",
        attribute="module naming convention",
        old_value="camelcase",
        new_value="snake_case",
        module="app/conventions.py",
        symbol="MODULE_NAMING_CONVENTION",
        narrative=(
            "The codebase has drifted on module naming.",
            "The module naming convention is camelcase.",
            "The team agreed to align with the standard library style.",
            "The module naming convention is snake_case.",
        ),
    ),
    Scenario(
        key="obsolete_fix",
        title="A prior bug fix becomes obsolete",
        attribute="retry backoff strategy",
        old_value="fixed-delay",
        new_value="exponential",
        module="app/retry_policy.py",
        symbol="RETRY_BACKOFF_STRATEGY",
        narrative=(
            "The gateway used to rate-limit us during bursts.",
            "The retry backoff strategy is fixed-delay.",
            "The gateway now publishes retry-after headers, so the old workaround is obsolete.",
            "The retry backoff strategy is exponential.",
        ),
    ),
    Scenario(
        key="secret_deleted",
        title="A secret or sensitive value is deleted",
        attribute="signing key material",
        old_value="leaked-signing-secret-9f2a41",
        new_value="managed-by-kms",
        module="app/signing.py",
        symbol="SIGNING_KEY_SOURCE",
        narrative=(
            "We need to record where request signing material comes from.",
            "The signing key material is leaked-signing-secret-9f2a41.",
            "That key was pasted into chat by mistake and must be forgotten.",
            "The signing key material is managed-by-kms.",
        ),
        deleted=True,
    ),
    Scenario(
        key="requirement_superseded",
        title="A requirement is superseded by a newer requirement",
        attribute="data retention requirement",
        old_value="ninety-days",
        new_value="thirty-days",
        module="app/retention.py",
        symbol="DATA_RETENTION",
        narrative=(
            "Legal is specifying how long request logs are kept.",
            "The data retention requirement is ninety-days.",
            "A newer privacy requirement supersedes the previous retention rule.",
            "The data retention requirement is thirty-days.",
        ),
    ),
)


#: Unrelated project chatter used to grow each scenario's store after the decision changes.
FILLER_TEMPLATES: tuple[str, ...] = (
    "The sprint {index} burndown is on track.",
    "The nightly build {index} completed without warnings.",
    "The oncall rotation {index} handover is documented.",
    "The dependency audit {index} found no advisories.",
    "The staging soak {index} ran for six hours.",
    "The changelog entry {index} was reviewed.",
)


class RepoEvolutionWorkload(Workload):
    """Eight multi-session scenarios with deterministic unit-test grading."""

    name = "repo_evolution"

    def __init__(
        self,
        *,
        version: str = "v1",
        seed: int = 1,
        subset: str | None = None,
        filler_sessions: int = 12,
    ) -> None:
        self.version = version
        self.seed = seed
        self.subset = subset
        # Unrelated sessions delivered *after* the decision changes. Without them each
        # scenario's store is a handful of events and every backend trivially finds the current
        # decision, so the task would measure nothing. Growth is what makes retrieval,
        # truncation, and supersession matter.
        self.filler_sessions = filler_sessions
        self._events: list[Event] = []
        self._queries: list[Query] = []
        self._tasks: list[CodingTask] = []
        self._loaded = False

    def _selected(self) -> tuple[Scenario, ...]:
        if self.subset == "smoke":
            return SCENARIOS[:3]
        return SCENARIOS

    def load(self) -> None:
        if self._loaded:
            return
        events: list[Event] = []
        queries: list[Query] = []
        tasks: list[CodingTask] = []
        day = 0

        for index, scenario in enumerate(self._selected()):
            tenant_id = f"repo_{scenario.key}"
            fact_event_ids: list[str] = []
            for line_index, line in enumerate(scenario.narrative):
                day += 1
                event_id = f"{scenario.key}_e{line_index:02d}"
                events.append(
                    Event(
                        event_id=event_id,
                        tenant_id=tenant_id,
                        session_id=f"{scenario.key}_s{line_index // 2:02d}",
                        event_type=(
                            EventType.user_message if line_index % 2 == 0 else EventType.feedback
                        ),
                        event_time=EPOCH + timedelta(days=day),
                        ingest_time=EPOCH + timedelta(days=day),
                        content=line,
                        metadata={"scenario": scenario.key},
                    )
                )
                if scenario.attribute in line:
                    fact_event_ids.append(event_id)

            for filler_index in range(self.filler_sessions):
                day += 1
                events.append(
                    Event(
                        event_id=f"{scenario.key}_f{filler_index:02d}",
                        tenant_id=tenant_id,
                        session_id=f"{scenario.key}_fill{filler_index:02d}",
                        event_type=EventType.assistant_message,
                        event_time=EPOCH + timedelta(days=day),
                        ingest_time=EPOCH + timedelta(days=day),
                        content=FILLER_TEMPLATES[filler_index % len(FILLER_TEMPLATES)].format(
                            index=filler_index
                        ),
                        metadata={"scenario": scenario.key, "role": "filler"},
                    )
                )

            if scenario.deleted:
                day += 1
                events.append(
                    Event(
                        event_id=f"{scenario.key}_del",
                        tenant_id=tenant_id,
                        session_id=f"{scenario.key}_s02",
                        event_type=EventType.deletion_request,
                        event_time=EPOCH + timedelta(days=day),
                        ingest_time=EPOCH + timedelta(days=day),
                        content=f"Delete the leaked value {scenario.old_value} from memory.",
                        metadata={"deletion_target": scenario.old_value},
                    )
                )

            memory_query = f"What is the {scenario.attribute}?"
            queries.append(
                Query(
                    query_id=f"{scenario.key}_q",
                    tenant_id=tenant_id,
                    text=memory_query,
                    category="knowledge_update",
                    gold_answer=scenario.new_value,
                    stale_values=[scenario.old_value],
                    forbidden_values=[scenario.old_value] if scenario.deleted else [],
                    expected_event_ids=fact_event_ids[-1:],
                    available_from_fraction=1.0,
                    metadata={"scenario": scenario.key},
                )
            )

            tasks.append(
                CodingTask(
                    task_id=f"{scenario.key}_task",
                    tenant_id=tenant_id,
                    scenario=scenario.key,
                    instruction=(
                        f"Set {scenario.symbol} in {scenario.module} to the project's current "
                        f"{scenario.attribute}."
                    ),
                    memory_query=memory_query,
                    gold_value=scenario.new_value,
                    stale_values=[scenario.old_value],
                    forbidden_values=[scenario.old_value] if scenario.deleted else [],
                    expected_event_ids=fact_event_ids[-1:],
                    edit_templates={
                        scenario.module: (
                            f'"""Generated by the coding agent from project memory."""\n\n'
                            f'{scenario.symbol} = "{{{{VALUE}}}}"\n'
                        )
                    },
                    repository=_repository_for(scenario),
                )
            )
            _ = index

        self._events = events
        self._queries = queries
        self._tasks = tasks
        self._loaded = True

    def events(self) -> list[Event]:
        self.load()
        return self._events

    def queries(self) -> list[Query]:
        self.load()
        return self._queries

    def coding_tasks(self) -> list[CodingTask]:
        self.load()
        return self._tasks

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update(
            {
                "scenarios": [scenario.key for scenario in self._selected()],
                "coding_tasks": len(self.coding_tasks()),
                "subset": self.subset,
                "filler_sessions": self.filler_sessions,
            }
        )
        return base


def _repository_for(scenario: Scenario) -> dict[str, str]:
    """The initial repository: a placeholder module plus the test that pins the requirement."""
    module_path = scenario.module
    package = module_path.split("/")[0]
    module_name = module_path.replace("/", ".").removesuffix(".py")
    return {
        f"{package}/__init__.py": "",
        module_path: (
            '"""Placeholder written before the current decision was recorded."""\n\n'
            f'{scenario.symbol} = "TODO"\n'
        ),
        "tests/__init__.py": "",
        f"tests/test_{scenario.key}.py": (
            f'"""Pins the project\'s current {scenario.attribute}."""\n\n'
            f"from {module_name} import {scenario.symbol}\n\n\n"
            f"def test_current_value() -> None:\n"
            f'    assert {scenario.symbol} == "{scenario.new_value}", (\n'
            f'        "the patch encoded {{!r}} instead of the current value".format(\n'
            f"            {scenario.symbol}\n"
            f"        )\n"
            f"    )\n\n\n"
            f"def test_no_placeholder_left() -> None:\n"
            f'    assert {scenario.symbol} != "TODO"\n'
        ),
    }
