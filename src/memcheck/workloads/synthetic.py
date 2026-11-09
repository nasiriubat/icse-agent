"""Synthetic longitudinal workload.

This is the only workload where store size can be swept cleanly while holding the *density* of
updates, deletions, duplicates, contradictions, and distractors fixed, which is what the growth
curves in RQ2 and RQ4 need. Everything is generated from a seed, so two runs of the same
configuration produce byte-identical streams.

The failure densities are chosen parameters, not measured field rates. That is stated in the
threats to validity and is why the growth curves are read as *relative* comparisons between
backends at equal density, never as absolute incidence estimates.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any

from memcheck.models import Event, EventType
from memcheck.workloads.base import Checkpoint, Query, Workload

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

#: Attribute phrases. A query is "What is the <attribute>?" and a fact is
#: "The <attribute> is <value>." so the attribute term set is exactly what a question carries.
ATTRIBUTES: tuple[str, ...] = (
    "primary database",
    "deployment target",
    "retention period",
    "message broker",
    "cache layer",
    "release cadence",
    "incident severity threshold",
    "backup window",
    "audit log destination",
    "feature flag provider",
    "search engine",
    "queue partition count",
    "session timeout",
    "encryption algorithm",
    "billing currency",
    "support tier",
    "monitoring vendor",
    "build system",
    "staging region",
    "api rate limit",
)

VALUES: tuple[str, ...] = (
    "postgres",
    "kubernetes",
    "thirty-days",
    "rabbitmq",
    "redis",
    "biweekly",
    "sev-two",
    "midnight-utc",
    "s3-archive",
    "launchdarkly",
    "opensearch",
    "sixteen",
    "forty-minutes",
    "aes-256-gcm",
    "eur",
    "platinum",
    "datadog",
    "bazel",
    "eu-west-1",
    "one-thousand-rpm",
    "mysql",
    "nomad",
    "ninety-days",
    "kafka",
    "memcached",
    "weekly",
    "sev-one",
    "noon-utc",
    "gcs-archive",
    "unleash",
    "elasticsearch",
    "thirty-two",
    "ten-minutes",
    "chacha20-poly1305",
    "usd",
    "gold",
    "grafana",
    "gradle",
    "us-east-2",
    "five-hundred-rpm",
)

DISTRACTOR_TEMPLATES: tuple[str, ...] = (
    "The {attribute} review meeting was productive.",
    "We discussed the {attribute} with the platform team.",
    "A ticket about the {attribute} was closed yesterday.",
    "The {attribute} documentation needs an update.",
)

#: Padding sentences that share no content token with any attribute or value, so they can only
#: dilute retrieval through corpus size and term statistics, never through a term match.
UNRELATED_PADDING_TEMPLATES: tuple[str, ...] = (
    "The quarterly offsite agenda was circulated to everyone.",
    "Lunch orders for the workshop close at eleven.",
    "The new hire orientation ran long this morning.",
    "Parking permits for the visitor lot were renewed.",
    "The book club picked a novel for next month.",
    "The volunteer day sign-up sheet is on the notice board.",
    "The office plants were watered by the facilities crew.",
    "A birthday card for the receptionist is going around.",
)

PADDING_STYLES = ("topical", "unrelated")


class SyntheticLongitudinalWorkload(Workload):
    """Deterministically generated multi-session, multi-tenant memory workload."""

    name = "synthetic_longitudinal"
    seed_sensitive = True

    def __init__(
        self,
        *,
        version: str = "v1",
        seed: int = 1,
        sessions: int = 24,
        facts_per_session: int = 3,
        distractors_per_session: int = 2,
        update_fraction: float = 0.35,
        deletion_fraction: float = 0.10,
        duplicate_fraction: float = 0.10,
        contradiction_fraction: float = 0.08,
        tenants: int = 2,
        padding_sessions: int = 0,
        padding_style: str = "topical",
        checkpoint_fractions: list[float] | None = None,
    ) -> None:
        if padding_sessions and version not in CHECKPOINT_VERSIONS:
            raise ValueError("padding_sessions requires workload version v3 or v5")
        if padding_style not in PADDING_STYLES:
            raise ValueError(f"padding_style must be one of {PADDING_STYLES}")
        self.version = version
        self.seed = seed
        self.sessions = sessions
        self.padding_sessions = padding_sessions
        self.padding_style = padding_style
        self.checkpoint_fractions = sorted(set(checkpoint_fractions or [1.0]))
        self.facts_per_session = facts_per_session
        self.distractors_per_session = distractors_per_session
        self.update_fraction = update_fraction
        self.deletion_fraction = deletion_fraction
        self.duplicate_fraction = duplicate_fraction
        self.contradiction_fraction = contradiction_fraction
        self.tenant_ids = [f"tenant_{index:02d}" for index in range(tenants)]
        self._events: list[Event] = []
        self._queries: list[Query] = []
        self._loaded = False

    # -- generation ---------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        rng = random.Random(self.seed)
        events: list[Event] = []
        # v4 reads the current value from the last real assertion, never the contradiction
        # marker; earlier versions emitted the marker itself as a fact value (DEVIATIONS V-024).
        real_current = self.version in ("v4", "v5")

        # Per-tenant state: attribute -> list of (value, event_id, event_time), plus deletions.
        history: dict[tuple[str, str], list[tuple[str, str, datetime]]] = {}
        deleted: set[tuple[str, str]] = set()
        counter = 0
        day = 0

        for session_index in range(self.sessions):
            for tenant_id in self.tenant_ids:
                session_id = f"s{session_index:03d}"
                day += 1
                event_time = EPOCH + timedelta(days=day)

                for _ in range(self.facts_per_session):
                    counter += 1
                    key_candidates = [
                        attribute
                        for attribute in ATTRIBUTES
                        if (tenant_id, attribute) not in deleted
                    ]
                    attribute = rng.choice(key_candidates)
                    key = (tenant_id, attribute)
                    known = history.get(key, [])

                    if known and rng.random() < self.update_fraction:
                        # A fact update: pick a value that is not the current one.
                        current_value = _current_value(known) if real_current else known[-1][0]
                        value = rng.choice([v for v in VALUES if v != current_value])
                    elif known:
                        # Re-assert the same fact; a well-behaved store must not duplicate it.
                        value = _current_value(known) if real_current else known[-1][0]
                    else:
                        value = rng.choice(VALUES)

                    event = _fact_event(
                        counter, tenant_id, session_id, event_time, attribute, value
                    )
                    events.append(event)
                    history.setdefault(key, []).append((value, event.event_id, event_time))

                    if rng.random() < self.duplicate_fraction:
                        # The same content re-delivered under a fresh id: content-hash duplicate.
                        counter += 1
                        duplicate = _fact_event(
                            counter, tenant_id, session_id, event_time, attribute, value
                        )
                        events.append(duplicate)

                    if rng.random() < self.contradiction_fraction and known:
                        # Two incompatible values at the *same* event_time: nothing in the
                        # stream establishes which is current.
                        counter += 1
                        other = rng.choice([v for v in VALUES if v != value])
                        events.append(
                            _fact_event(
                                counter, tenant_id, session_id, event_time, attribute, other
                            )
                        )
                        history[key].append((other, events[-1].event_id, event_time))
                        history[key].append((CONTRADICTION_MARKER, "", event_time))

                for _ in range(self.distractors_per_session):
                    counter += 1
                    attribute = rng.choice(ATTRIBUTES)
                    template = rng.choice(DISTRACTOR_TEMPLATES)
                    events.append(
                        Event(
                            event_id=f"ev_{counter:05d}",
                            tenant_id=tenant_id,
                            session_id=session_id,
                            event_type=EventType.assistant_message,
                            event_time=event_time,
                            ingest_time=event_time,
                            content=template.format(attribute=attribute),
                            metadata={"role": "distractor"},
                        )
                    )

                # Deletions happen after the fact has had time to reach summaries and indexes.
                if session_index >= self.sessions // 3 and rng.random() < self.deletion_fraction:
                    deletable = [
                        (tenant, attribute)
                        for (tenant, attribute) in history
                        if tenant == tenant_id and (tenant, attribute) not in deleted
                    ]
                    if deletable:
                        counter += 1
                        _, attribute = rng.choice(deletable)
                        deleted.add((tenant_id, attribute))
                        events.append(
                            Event(
                                event_id=f"ev_{counter:05d}",
                                tenant_id=tenant_id,
                                session_id=session_id,
                                event_type=EventType.deletion_request,
                                event_time=event_time,
                                ingest_time=event_time,
                                content=f"Delete everything about the {attribute}.",
                                metadata={"deletion_target": attribute},
                            )
                        )

        if self.padding_sessions:
            events.extend(self._padding_events(counter, day))

        self._events = events
        position_of = {event.event_id: index for index, event in enumerate(events)}
        if self.version in CHECKPOINT_VERSIONS:
            deletion_positions = {
                (event.tenant_id, str(event.metadata["deletion_target"])): index
                for index, event in enumerate(events)
                if event.event_type == EventType.deletion_request
            }
            self._queries = _build_checkpoint_queries(
                history,
                deletion_positions,
                position_of,
                self._checkpoint_specs(),
                temporal_from_last_change=self.version == "v5",
            )
        else:
            self._queries = _build_queries(
                history,
                deleted,
                position_of,
                len(events),
                cross_session=self.version == "v1",
                temporal_from_last_change=self.version == "v4",
            )
        self._loaded = True

    def _padding_events(self, counter: int, day: int) -> list[Event]:
        """Sessions appended after the fact stream that assert nothing.

        They are drawn from a generator seeded independently of the fact stream, so the facts,
        their order, and the derived query set are identical at every padding level. Each padding
        tenant-session carries as many events as a fact session does, so a padding session adds
        the same volume to the store as a real one.
        """
        rng = random.Random(self.seed * 1_000_003 + 7)
        events: list[Event] = []
        per_tenant_session = self.facts_per_session + self.distractors_per_session
        for offset in range(self.padding_sessions):
            session_id = f"s{self.sessions + offset:03d}"
            for tenant_id in self.tenant_ids:
                day += 1
                event_time = EPOCH + timedelta(days=day)
                for _ in range(per_tenant_session):
                    counter += 1
                    if self.padding_style == "topical":
                        content = rng.choice(DISTRACTOR_TEMPLATES).format(
                            attribute=rng.choice(ATTRIBUTES)
                        )
                    else:
                        content = rng.choice(UNRELATED_PADDING_TEMPLATES)
                    events.append(
                        Event(
                            event_id=f"ev_{counter:05d}",
                            tenant_id=tenant_id,
                            session_id=session_id,
                            event_type=EventType.assistant_message,
                            event_time=event_time,
                            ingest_time=event_time,
                            content=content,
                            metadata={"role": "padding", "padding_style": self.padding_style},
                        )
                    )
        return events

    # -- checkpoints (v3) -----------------------------------------------------------------
    @property
    def total_sessions(self) -> int:
        return self.sessions + self.padding_sessions

    def _session_checkpoint(self, fraction: float) -> tuple[str, int, int]:
        """Snap a fraction of the stream to the end of a session.

        Returns ``(label, event_index, sessions_ingested)``. Sessions differ slightly in length,
        so a fraction of the event count would land mid-session; snapping makes "the store at
        16 sessions" mean exactly that.
        """
        ends: dict[str, int] = {}
        for index, event in enumerate(self._events):
            ends[event.session_id] = index + 1
        sessions = min(self.total_sessions, max(1, round(fraction * self.total_sessions)))
        return f"cp{round(fraction * 100):03d}", ends[f"s{sessions - 1:03d}"], sessions

    def _checkpoint_specs(self) -> list[tuple[str, float, int, int]]:
        specs: list[tuple[str, float, int, int]] = []
        for fraction in self.checkpoint_fractions:
            label, event_index, sessions = self._session_checkpoint(fraction)
            specs.append((label, fraction, event_index, sessions))
        return specs

    def checkpoints(self, fractions: list[float]) -> list[Checkpoint]:
        if self.version not in CHECKPOINT_VERSIONS:
            return super().checkpoints(fractions)
        self.load()
        result: list[Checkpoint] = []
        for fraction in fractions:
            label, event_index, _ = self._session_checkpoint(fraction)
            result.append(Checkpoint(label=label, fraction=fraction, event_index=event_index))
        return result

    def events(self) -> list[Event]:
        self.load()
        return self._events

    def queries(self) -> list[Query]:
        self.load()
        return self._queries

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update(
            {
                "seed": self.seed,
                "sessions": self.sessions,
                "facts_per_session": self.facts_per_session,
                "update_fraction": self.update_fraction,
                "deletion_fraction": self.deletion_fraction,
                "duplicate_fraction": self.duplicate_fraction,
                "contradiction_fraction": self.contradiction_fraction,
                "tenants": self.tenant_ids,
            }
        )
        if self.version in CHECKPOINT_VERSIONS:
            base.update(
                {
                    "padding_sessions": self.padding_sessions,
                    "padding_style": self.padding_style,
                    "checkpoint_fractions": self.checkpoint_fractions,
                }
            )
        return base


CONTRADICTION_MARKER = "__contradiction__"

#: Versions that ask each query at the checkpoint it belongs to and support padding. v5 is v3
#: with v4's two label fixes (DEVIATIONS V-025).
CHECKPOINT_VERSIONS = ("v3", "v5")


def _current_value(entries: list[tuple[str, str, datetime]]) -> str:
    """The value of the last real assertion; marker entries carry no event id."""
    return next(value for value, event_id, _ in reversed(entries) if event_id)


def _temporal_target(
    entries: list[tuple[str, str, datetime]],
) -> tuple[str, str, datetime] | None:
    """The value current just before the latest change of value, and an instant inside it.

    Returns ``(previous_value, previous_event_id, as_of)`` or ``None`` when no instant exists at
    which the previous value was unambiguously current: the value never changed, the change
    happened at the same timestamp as the previous assertion, or that timestamp also carries a
    different value or a contradiction.
    """
    real = [(value, event_id, when) for value, event_id, when in entries if event_id]
    current = real[-1][0]
    start = len(real) - 1
    while start > 0 and real[start - 1][0] == current:
        start -= 1
    if start == 0:
        return None
    change_time = real[start][2]
    previous_value, previous_event_id, previous_time = real[start - 1]
    if previous_time >= change_time:
        return None
    if any(when == previous_time and value != previous_value for value, _, when in entries):
        return None
    return previous_value, previous_event_id, previous_time + (change_time - previous_time) / 2


def _fact_event(
    counter: int,
    tenant_id: str,
    session_id: str,
    event_time: datetime,
    attribute: str,
    value: str,
) -> Event:
    return Event(
        event_id=f"ev_{counter:05d}",
        tenant_id=tenant_id,
        session_id=session_id,
        event_type=EventType.user_message,
        event_time=event_time,
        ingest_time=event_time,
        content=f"The {attribute} is {value}.",
        metadata={"attribute": attribute, "value": value},
    )


def _build_queries(
    history: dict[tuple[str, str], list[tuple[str, str, datetime]]],
    deleted: set[tuple[str, str]],
    position_of: dict[str, int],
    total_events: int,
    *,
    cross_session: bool = True,
    temporal_from_last_change: bool = False,
) -> list[Query]:
    """Derive the graded query set from the generated ground truth.

    ``cross_session`` keeps v1's ``_xs`` query. It repeats the ``_cur`` query's text, tenant and
    gold answer at the full store, where ``_cur`` is also asked, so in v1 every such fact is
    graded twice at the final checkpoint. v2 drops it (DEVIATIONS V-018).

    Gold labels come from the generator's own state, never from a model, so they are exact.

    A query becomes available only once the event that supports its gold answer has actually been
    delivered. Asking about a fact the store has not yet seen would score every backend as wrong
    at early checkpoints and would say nothing about memory.
    """

    def available_from(event_id: str) -> float:
        # +1 so the checkpoint that *includes* the supporting event qualifies.
        return min(1.0, (position_of.get(event_id, 0) + 1) / max(1, total_events))

    queries: list[Query] = []
    index = 0
    for (tenant_id, attribute), entries in sorted(history.items()):
        real = [(value, event_id, when) for value, event_id, when in entries if event_id]
        if not real:
            continue
        # A contradiction only stands if nothing later resolved it. A strictly later dated
        # assertion is exactly the evidence that settles which value is current, so labelling
        # such a key "must abstain" would penalise a store for correctly resolving it.
        contradiction_times = {
            when for value, event_id, when in entries if value == "__contradiction__"
        }
        latest_assertion = max(when for _, _, when in real)
        contradicted = any(when >= latest_assertion for when in contradiction_times)
        index += 1
        text = f"What is the {attribute}?"
        current_value, current_event_id, current_time = real[-1]
        prior_values = [value for value, _, _ in real[:-1] if value != current_value]

        if (tenant_id, attribute) in deleted:
            # After deletion the only correct behaviour is to abstain; any stored value is a leak.
            queries.append(
                Query(
                    query_id=f"q{index:04d}_del",
                    tenant_id=tenant_id,
                    text=text,
                    category="abstention",
                    gold_answer=None,
                    expects_abstention=True,
                    forbidden_values=sorted({value for value, _, _ in real}),
                    expected_event_ids=[],
                    available_from_fraction=1.0,
                    metadata={"attribute": attribute, "reason": "deleted"},
                )
            )
            continue

        if contradicted:
            queries.append(
                Query(
                    query_id=f"q{index:04d}_con",
                    tenant_id=tenant_id,
                    text=text,
                    category="abstention",
                    gold_answer=None,
                    expects_abstention=True,
                    expected_event_ids=[event_id for _, event_id, _ in real],
                    available_from_fraction=1.0,
                    metadata={"attribute": attribute, "reason": "unresolved_contradiction"},
                )
            )
            continue

        category = "knowledge_update" if prior_values else "single_session_extraction"
        queries.append(
            Query(
                query_id=f"q{index:04d}_cur",
                tenant_id=tenant_id,
                text=text,
                category=category,
                gold_answer=current_value,
                stale_values=sorted(set(prior_values)),
                expected_event_ids=[current_event_id],
                available_from_fraction=available_from(current_event_id),
                metadata={"attribute": attribute},
            )
        )

        # Cross-session recall: the same fact asked only at the end of a much larger store.
        # v1 only; it duplicates the ``_cur`` query at the final checkpoint.
        if cross_session:
            queries.append(
                Query(
                    query_id=f"q{index:04d}_xs",
                    tenant_id=tenant_id,
                    text=text,
                    category="cross_session_reasoning",
                    gold_answer=current_value,
                    stale_values=sorted(set(prior_values)),
                    expected_event_ids=[current_event_id],
                    available_from_fraction=1.0,
                    metadata={"attribute": attribute, "asked_at_full_store": True},
                )
            )

        # Temporal reasoning: what was true before the latest update? v2 takes the assertion just
        # before the last one, which is often a re-assertion of the current value; v4 takes the
        # value current just before the last change and drops keys with no such instant.
        if temporal_from_last_change:
            target = _temporal_target(entries)
            if target is not None:
                previous_value, previous_event_id, as_of = target
                queries.append(
                    Query(
                        query_id=f"q{index:04d}_tmp",
                        tenant_id=tenant_id,
                        text=text,
                        category="temporal_reasoning",
                        gold_answer=previous_value,
                        stale_values=[],
                        expected_event_ids=[previous_event_id],
                        as_of_time=as_of,
                        available_from_fraction=1.0,
                        metadata={"attribute": attribute, "as_of": as_of.isoformat()},
                    )
                )
        elif prior_values:
            previous_value, previous_event_id, previous_time = real[-2]
            midpoint = previous_time + (current_time - previous_time) / 2
            queries.append(
                Query(
                    query_id=f"q{index:04d}_tmp",
                    tenant_id=tenant_id,
                    text=text,
                    category="temporal_reasoning",
                    gold_answer=previous_value,
                    stale_values=[],
                    expected_event_ids=[previous_event_id],
                    as_of_time=midpoint,
                    available_from_fraction=1.0,
                    metadata={"attribute": attribute, "as_of": midpoint.isoformat()},
                )
            )
    return queries


def _build_checkpoint_queries(
    history: dict[tuple[str, str], list[tuple[str, str, datetime]]],
    deletion_positions: dict[tuple[str, str], int],
    position_of: dict[str, int],
    specs: list[tuple[str, float, int, int]],
    *,
    temporal_from_last_change: bool = False,
) -> list[Query]:
    """One query set per checkpoint, each graded against the world as of that checkpoint (v3).

    v2 asks every question at the full store only, so a sweep over sizes compares separately
    generated streams. Here the same stream is asked at each checkpoint what was true *then*:
    the current value as of that event index, deletion if the request had arrived, abstention
    if a contradiction was still unresolved. Every query is answerable at the checkpoint it is
    asked, because its gold is derived from the events delivered by then.
    """
    queries: list[Query] = []
    key_index = {key: number for number, key in enumerate(sorted(history), start=1)}
    for label, fraction, event_index, sessions in specs:
        for key, entries in sorted(history.items()):
            tenant_id, attribute = key
            real: list[tuple[str, str, datetime]] = []
            delivered: list[tuple[str, str, datetime]] = []
            contradiction_times: set[datetime] = set()
            for offset, (value, event_id, when) in enumerate(entries):
                # A marker entry has no event of its own; it follows the event that contradicts
                # and is in force once that event has been delivered. Any delivered entry whose
                # value is the marker string counts as a contradiction, exactly as in v2.
                anchor = event_id or entries[offset - 1][1]
                if not anchor or position_of[anchor] >= event_index:
                    continue
                delivered.append((value, event_id, when))
                if event_id:
                    real.append((value, event_id, when))
                if value == "__contradiction__":
                    contradiction_times.add(when)
            if not real:
                continue
            latest_assertion = max(when for _, _, when in real)
            contradicted = any(when >= latest_assertion for when in contradiction_times)
            deleted = deletion_positions.get(key, event_index) < event_index
            prefix = f"q{key_index[key]:04d}"
            text = f"What is the {attribute}?"
            where = {"attribute": attribute, "checkpoint": label, "sessions": sessions}
            current_value, current_event_id, current_time = real[-1]
            prior_values = [value for value, _, _ in real[:-1] if value != current_value]

            if deleted:
                queries.append(
                    Query(
                        query_id=f"{prefix}_del_{label}",
                        tenant_id=tenant_id,
                        text=text,
                        category="abstention",
                        gold_answer=None,
                        expects_abstention=True,
                        forbidden_values=sorted({value for value, _, _ in real}),
                        expected_event_ids=[],
                        available_from_fraction=fraction,
                        asked_at_fraction=fraction,
                        metadata={**where, "reason": "deleted"},
                    )
                )
                continue
            if contradicted:
                queries.append(
                    Query(
                        query_id=f"{prefix}_con_{label}",
                        tenant_id=tenant_id,
                        text=text,
                        category="abstention",
                        gold_answer=None,
                        expects_abstention=True,
                        expected_event_ids=[event_id for _, event_id, _ in real],
                        available_from_fraction=fraction,
                        asked_at_fraction=fraction,
                        metadata={**where, "reason": "unresolved_contradiction"},
                    )
                )
                continue
            queries.append(
                Query(
                    query_id=f"{prefix}_cur_{label}",
                    tenant_id=tenant_id,
                    text=text,
                    category="knowledge_update" if prior_values else "single_session_extraction",
                    gold_answer=current_value,
                    stale_values=sorted(set(prior_values)),
                    expected_event_ids=[current_event_id],
                    available_from_fraction=fraction,
                    asked_at_fraction=fraction,
                    metadata=where,
                )
            )
            temporal: tuple[str, str, datetime] | None = None
            if temporal_from_last_change:
                temporal = _temporal_target(delivered)
            elif prior_values:
                previous_value, previous_event_id, previous_time = real[-2]
                temporal = (
                    previous_value,
                    previous_event_id,
                    previous_time + (current_time - previous_time) / 2,
                )
            if temporal is not None:
                previous_value, previous_event_id, midpoint = temporal
                queries.append(
                    Query(
                        query_id=f"{prefix}_tmp_{label}",
                        tenant_id=tenant_id,
                        text=text,
                        category="temporal_reasoning",
                        gold_answer=previous_value,
                        stale_values=[],
                        expected_event_ids=[previous_event_id],
                        as_of_time=midpoint,
                        available_from_fraction=fraction,
                        asked_at_fraction=fraction,
                        metadata={**where, "as_of": midpoint.isoformat()},
                    )
                )
    return queries
