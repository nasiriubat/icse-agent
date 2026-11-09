"""Append-only JSONL streams for experiment artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any

from pydantic import BaseModel


class JsonlWriter:
    """Line-buffered append-only writer. Flushes on every record so a crash keeps partial data."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._count = 0

    def write(self, record: BaseModel | dict[str, Any]) -> None:
        payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        self._handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
        self._handle.flush()
        self._count += 1

    def write_many(self, records: list[BaseModel] | list[dict[str, Any]]) -> None:
        for record in records:
            self.write(record)

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file. Missing files yield nothing."""
    file_path = Path(path)
    if not file_path.exists():
        return
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
