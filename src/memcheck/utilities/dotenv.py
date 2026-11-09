"""Minimal ``.env`` loader.

MemCheck reads every secret and every cost control from the environment. A ``.env`` file is a
convenience for getting them *into* the environment on a development machine; it is never a
place MemCheck writes to, and its values are never logged.

Two rules make this safe to call unconditionally at CLI start-up:

1. **A real environment variable always wins.** An exported ``ALLOW_PAID_RUNS=false`` cannot be
   silently overridden by a stale ``.env`` sitting in the working directory.
2. **Only the file's own keys are touched.** Nothing is inferred, defaulted, or expanded.

Deliberately dependency-free: `python-dotenv` would be a runtime dependency for thirty lines.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Variables whose values must never be echoed back, even in a "loaded X" message.
SECRET_KEYS = frozenset({"ANTHROPIC_API_KEY", "OPENAI_API_KEY"})


def parse(text: str) -> dict[str, str]:
    """Parse ``.env`` text into a mapping. Supports ``KEY=value``, ``export KEY=value``,
    ``#`` comments, blank lines, and single- or double-quoted values."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip a matching pair of surrounding quotes, then an inline comment on unquoted values.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load(path: str | Path = ".env", *, override: bool = False) -> list[str]:
    """Load ``path`` into ``os.environ``. Returns the names of the variables it set.

    Existing environment variables are preserved unless ``override`` is explicitly requested,
    so the shell remains the authority over whether a paid run is allowed.
    """
    env_path = Path(path)
    if not env_path.exists():
        return []
    applied: list[str] = []
    for key, value in parse(env_path.read_text(encoding="utf-8")).items():
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied.append(key)
    return applied


def describe(applied: list[str]) -> str:
    """A human-readable summary that names the variables but never their values."""
    if not applied:
        return ""
    shown = [f"{key} (value hidden)" if key in SECRET_KEYS else key for key in sorted(applied)]
    return f"loaded {len(applied)} variable(s) from .env: {', '.join(shown)}"
