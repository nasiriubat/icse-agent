"""LongMemEval adapter.

**No benchmark data is redistributed here.** The adapter reads a licensed download the user
supplies, verifies it against a recorded checksum, and maps it onto MemCheck's event/query
interface. Without the data the adapter raises a message telling the user exactly what to fetch
and where to put it, and the experiment runner records the workload as unavailable rather than
silently substituting something else.

See ``data/README.md`` for download instructions and checksum registration.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from memcheck.models import Event, EventType
from memcheck.utilities.hashing import file_sha256
from memcheck.workloads.base import Query, Workload

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

#: LongMemEval question types mapped onto MemCheck's categories.
CATEGORY_MAP: dict[str, str] = {
    "single-session-user": "single_session_extraction",
    "single-session-assistant": "single_session_extraction",
    "single-session-preference": "single_session_extraction",
    "multi-session": "cross_session_reasoning",
    "temporal-reasoning": "temporal_reasoning",
    "knowledge-update": "knowledge_update",
    "abstention": "abstention",
}

#: Accepted filenames per subset, in preference order. The authors publish the large subsets
#: under `_cleaned` names; both are accepted so a download from either release works.
EXPECTED_FILES: dict[str, tuple[str, ...]] = {
    "smoke": ("longmemeval_oracle.json",),
    "s": ("longmemeval_s_cleaned.json", "longmemeval_s.json"),
    "m": ("longmemeval_m_cleaned.json", "longmemeval_m.json"),
}

#: LongMemEval marks an abstention instance by suffixing its question id, not by giving it a
#: distinct `question_type`. Missing this would grade 30 of the 500 oracle instances as if an
#: answer existed, when the correct behaviour is to decline.
ABSTENTION_SUFFIX = "_abs"

#: Session timestamps look like "2023/04/10 (Mon) 17:50".
DATE_PATTERN = re.compile(r"(\d{4})/(\d{2})/(\d{2})[^\d]*(\d{2}):(\d{2})")


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse a LongMemEval timestamp, ignoring the weekday in parentheses."""
    if not value:
        return None
    match = DATE_PATTERN.search(str(value))
    if match is None:
        return None
    year, month, day, hour, minute = (int(part) for part in match.groups())
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


class LongMemEvalUnavailable(FileNotFoundError):
    """Raised when the licensed dataset has not been downloaded."""


def _stratified_subset(instances: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Deterministically sample across question types, keeping abstention instances represented.

    The released file is ordered by question type, so a head slice covers one category and says
    nothing about the other four. Round-robin over the groups keeps a small subset a fair
    miniature of the benchmark, and sorting by question id makes it reproducible.
    """
    if limit >= len(instances):
        return list(instances)
    groups: dict[str, list[dict[str, Any]]] = {}
    for instance in instances:
        question_id = str(instance.get("question_id", ""))
        # Abstention is a category in its own right, marked by the id suffix rather than by
        # question_type, so it gets its own stratum instead of being folded into a base type.
        key = (
            "abstention"
            if question_id.endswith(ABSTENTION_SUFFIX)
            else str(instance.get("question_type", "unknown"))
        )
        groups.setdefault(key, []).append(instance)
    for group in groups.values():
        group.sort(key=lambda item: str(item.get("question_id", "")))

    selected: list[dict[str, Any]] = []
    position = 0
    while len(selected) < limit:
        added = False
        for key in sorted(groups):
            group = groups[key]
            if position < len(group):
                selected.append(group[position])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        position += 1
    return selected


class LongMemEvalWorkload(Workload):
    """Adapter over a user-supplied LongMemEval download."""

    name = "longmemeval"

    def __init__(
        self,
        *,
        version: str = "v1",
        subset: str = "smoke",
        data_path: str | None = None,
        limit: int | None = None,
    ) -> None:
        self.version = version
        self.subset = subset
        self.limit = limit
        self.data_dir = Path(data_path or "data/raw")
        self._events: list[Event] = []
        self._queries: list[Query] = []
        self._loaded = False

    @property
    def candidate_files(self) -> tuple[Path, ...]:
        filenames = EXPECTED_FILES.get(self.subset)
        if filenames is None:
            raise KeyError(
                f"unknown LongMemEval subset {self.subset!r}; known: {sorted(EXPECTED_FILES)}"
            )
        return tuple(self.data_dir / name for name in filenames)

    @property
    def expected_file(self) -> Path:
        """The file that is present, or the preferred name if none is."""
        for candidate in self.candidate_files:
            if candidate.exists():
                return candidate
        return self.candidate_files[0]

    def is_available(self) -> bool:
        return any(candidate.exists() for candidate in self.candidate_files)

    def verify_checksum(self) -> tuple[bool, str]:
        """Compare the download against ``data/checksums/longmemeval.json`` if registered."""
        registry = Path("data/checksums/longmemeval.json")
        if not registry.exists():
            return False, "no checksum registered; run `memcheck data verify --register` first"
        recorded = json.loads(registry.read_text(encoding="utf-8"))
        expected = recorded.get(self.expected_file.name)
        if expected is None:
            return False, f"{self.expected_file.name} is not in the checksum registry"
        actual = file_sha256(self.expected_file)
        if actual != expected:
            return False, f"checksum mismatch: expected {expected}, found {actual}"
        return True, "checksum verified"

    def load(self) -> None:
        if self._loaded:
            return
        if not self.is_available():
            raise LongMemEvalUnavailable(
                f"LongMemEval subset {self.subset!r} not found at {self.expected_file}.\n"
                "LongMemEval is not redistributed with MemCheck. Follow data/README.md to "
                "download it, place it at the path above, then run:\n"
                "  uv run memcheck data verify --dataset longmemeval --register"
            )
        payload = json.loads(self.expected_file.read_text(encoding="utf-8"))
        instances = payload if isinstance(payload, list) else payload.get("instances", [])
        if self.limit is not None:
            instances = _stratified_subset(instances, self.limit)

        events: list[Event] = []
        queries: list[Query] = []
        day = 0
        for index, instance in enumerate(instances):
            tenant_id = f"lme_{index:04d}"
            sessions = instance.get("haystack_sessions") or instance.get("sessions") or []
            dates = instance.get("haystack_dates") or []
            evidence_ids: list[str] = []
            for session_index, session in enumerate(sessions):
                day += 1
                # Prefer the dataset's own session timestamp: temporal-reasoning questions are
                # about when things were said, so inventing an ordering would change the task.
                session_time = parse_timestamp(
                    dates[session_index] if session_index < len(dates) else None
                ) or (EPOCH + timedelta(days=day))
                turns = session if isinstance(session, list) else session.get("turns", [])
                for turn_index, turn in enumerate(turns):
                    role = str(turn.get("role", "user"))
                    content = str(turn.get("content", "")).strip()
                    if not content:
                        continue
                    event_id = f"{tenant_id}_s{session_index:03d}_t{turn_index:03d}"
                    if turn.get("has_answer"):
                        evidence_ids.append(event_id)
                    events.append(
                        Event(
                            event_id=event_id,
                            tenant_id=tenant_id,
                            session_id=f"{tenant_id}_s{session_index:03d}",
                            event_type=(
                                EventType.user_message
                                if role == "user"
                                else EventType.assistant_message
                            ),
                            event_time=session_time + timedelta(minutes=turn_index),
                            ingest_time=session_time + timedelta(minutes=turn_index),
                            content=content,
                            metadata={"role": role, "instance": instance.get("question_id")},
                        )
                    )
            question_id = str(instance.get("question_id", f"{tenant_id}_q"))
            question_type = str(instance.get("question_type", "multi-session"))
            is_abstention = question_id.endswith(ABSTENTION_SUFFIX)
            category = (
                "abstention"
                if is_abstention
                else CATEGORY_MAP.get(question_type, "cross_session_reasoning")
            )
            answer = instance.get("answer")
            queries.append(
                Query(
                    query_id=question_id,
                    tenant_id=tenant_id,
                    text=str(instance.get("question", "")),
                    category=category,
                    gold_answer=None
                    if is_abstention
                    else (str(answer) if answer is not None else None),
                    expects_abstention=is_abstention,
                    # LongMemEval gold answers are free text, so containment is not a sound
                    # equivalence test; semantic judging with a human audit path is required.
                    grader="llm_judge",
                    expected_event_ids=[] if is_abstention else evidence_ids,
                    as_of_time=parse_timestamp(instance.get("question_date")),
                    available_from_fraction=1.0,
                    metadata={
                        "question_type": question_type,
                        "abstention_marked_by": "question_id suffix" if is_abstention else None,
                    },
                )
            )

        self._events = sorted(events, key=lambda event: (event.event_time, event.event_id))
        self._queries = queries
        self._loaded = True

    def events(self) -> list[Event]:
        self.load()
        return self._events

    def queries(self) -> list[Query]:
        self.load()
        return self._queries

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        ok, message = self.verify_checksum() if self.is_available() else (False, "not downloaded")
        base.update(
            {
                "subset": self.subset,
                "source_file": str(self.expected_file),
                "checksum_verified": ok,
                "checksum_note": message,
            }
        )
        return base
