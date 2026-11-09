"""Environment capture for run manifests, plus SQLite capability probing."""

from __future__ import annotations

import platform
import sqlite3
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from memcheck.utilities.hashing import file_sha256


def _run(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=10, cwd=cwd, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_commit(repo_root: Path | None = None) -> str | None:
    return _run(["git", "rev-parse", "HEAD"], cwd=repo_root)


def working_tree_status(repo_root: Path | None = None) -> str:
    output = _run(["git", "status", "--porcelain"], cwd=repo_root)
    if output is None:
        return "not_a_git_repository"
    return "clean" if output == "" else f"dirty:{len(output.splitlines())}_files"


def operating_system() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def python_version() -> str:
    return sys.version.split()[0]


def cpu_information() -> str:
    processor = platform.processor() or platform.machine()
    try:
        import os

        cores = os.cpu_count() or 0
    except Exception:  # pragma: no cover - defensive
        cores = 0
    return f"{processor}, {cores} logical cores"


def gpu_information() -> str:
    """Report a GPU only if one is detectable without extra dependencies."""
    output = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if output:
        return output.replace("\n", "; ")
    return "none"


def dependency_lock_hash(repo_root: Path) -> str:
    """Hash the strongest available pin file, so the manifest identifies the dependency set."""
    for candidate in ("uv.lock", "requirements.lock.txt", "pyproject.toml"):
        path = repo_root / candidate
        if path.exists():
            return f"{candidate}:{file_sha256(path)[:32]}"
    return "unknown"


@lru_cache(maxsize=1)
def sqlite_has_fts5() -> bool:
    """Probe whether the local SQLite build includes FTS5 (DECISIONS D-003)."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE _probe USING fts5(content)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        connection.close()


def sqlite_version() -> str:
    return sqlite3.sqlite_version


def latex_engine() -> str | None:
    for engine in ("latexmk", "pdflatex", "tectonic", "xelatex"):
        if _run(["which", engine]):
            return engine
    return None


def repo_root() -> Path:
    """Repository root: walk up from this file until a pyproject.toml is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
