"""Content hashing and deterministic identifier generation.

MemCheck never uses ``uuid4`` for anything that must be reproducible: identifiers are derived
from content so two runs of the same configuration produce byte-identical ids.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from memcheck.models import canonical_json


def stable_id(prefix: str, *parts: Any) -> str:
    """Deterministic identifier from a prefix and any hashable-to-JSON parts."""
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory(path: str | Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    """Map of relative path -> sha256 for every file under ``path``.

    Used to make ``results/raw/<run_id>/`` tamper-evident (DECISIONS D-011).
    """
    root = Path(path)
    result: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(root).as_posix()
        if relative in exclude or relative.endswith(".gitkeep"):
            continue
        result[relative] = file_sha256(item)
    return result
