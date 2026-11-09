"""Sandboxed coding agent for RepoEvolutionBench.

Safety rules, enforced by construction rather than by prompt:

* The model returns **structured file edits**, never shell commands. Arbitrary model-generated
  commands are never executed.
* Edits are applied inside a temporary copy of the repository. Paths are resolved and confined
  to the sandbox root; an edit that escapes it is rejected.
* The only command ever run is the allowlisted ``python -m pytest`` invocation MemCheck itself
  constructs, with a timeout.
* The environment is stripped of proxy variables and the sandbox is a throwaway directory. This
  is best-effort isolation, not a security boundary — recorded in DEVIATIONS.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from memcheck.config import ModelConfig
from memcheck.models import ProviderCall, RetrievalResult, StrictModel
from memcheck.providers.base import ChatMessage, ChatRequest, Provider
from memcheck.providers.prompts import get_prompt
from memcheck.workloads.repo_evolution import CodingTask

ALLOWED_TEST_EXECUTABLE = sys.executable
DEFAULT_TIMEOUT_SECONDS = 60


#: What a real model is asked to return; the mock reader returns the same shape.
CODING_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "abstained": {"type": "boolean"},
        "reason": {"type": "string"},
        "resolved_value": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["abstained", "reason", "resolved_value", "edits"],
    "additionalProperties": False,
}


class PatchRejected(ValueError):
    """Raised when a proposed edit is malformed or escapes the sandbox."""


class CodingTaskResult(StrictModel):
    task_id: str
    scenario: str
    backend_name: str
    checkpoint: str
    tenant_id: str
    abstained: bool = False
    resolved_value: str | None = None
    gold_value: str = ""
    patch_applied: bool = False
    tests_passed: bool = False
    tests_run: bool = False
    return_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_ms: float = 0.0
    rejected_reason: str | None = None
    used_stale_value: bool = False
    used_forbidden_value: bool = False
    evidence_event_ids: list[str] = []
    provider_call: ProviderCall | None = None


class CodingAgent:
    """Resolves the current project decision from memory, patches, and runs the tests."""

    def __init__(
        self,
        provider: Provider,
        model: ModelConfig,
        *,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds
        # v1 asked for a unified diff that nothing parsed and promised file contents it was
        # never sent; v2 asks for structured edits and is sent the file it may edit.
        self.system_prompt = get_prompt("coding_v2")

    def run_task(
        self,
        task: CodingTask,
        retrieval: RetrievalResult,
        *,
        backend_name: str,
        checkpoint: str,
        artifacts_dir: Path | None = None,
    ) -> CodingTaskResult:
        started = time.perf_counter()
        result = CodingTaskResult(
            task_id=task.task_id,
            scenario=task.scenario,
            backend_name=backend_name,
            checkpoint=checkpoint,
            tenant_id=task.tenant_id,
            gold_value=task.gold_value,
            evidence_event_ids=sorted(
                {eid for item in retrieval.evidence for eid in item.source_event_ids}
            ),
        )

        request = ChatRequest(
            model_identifier=self.model.identifier,
            system_prompt=self.system_prompt,
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        f"Task: {task.instruction}\n\n"
                        f"Question the patch must answer: {task.memory_query}\n\n"
                        # Only the files the task may edit; never the tests, which hold the gold.
                        + "".join(
                            f"Current contents of {path}:\n{task.repository.get(path, '')}\n\n"
                            for path in task.edit_templates
                        )
                        + "Memory evidence:\n"
                        + (
                            "\n".join(
                                f"- {item.content} (valid_from="
                                f"{item.valid_from.isoformat() if item.valid_from else 'unknown'}, "
                                f"valid_to="
                                f"{item.valid_to.isoformat() if item.valid_to else 'open'})"
                                for item in retrieval.evidence
                            )
                            or "(no evidence retrieved)"
                        )
                    ),
                )
            ],
            temperature=self.model.temperature,
            max_output_tokens=self.model.max_output_tokens,
            purpose="code_patch",
            response_schema=CODING_RESPONSE_SCHEMA,
            structured_input={
                "query": task.memory_query,
                "abstain_on_conflict": True,
                "evidence": [
                    {
                        "memory_id": item.memory_id,
                        "content": item.content,
                        "confidence": item.confidence,
                        "valid_from": item.valid_from.isoformat() if item.valid_from else None,
                        "valid_to": item.valid_to.isoformat() if item.valid_to else None,
                    }
                    for item in retrieval.evidence
                ],
                "edits": [
                    {"path": path, "content": content}
                    for path, content in task.edit_templates.items()
                ],
            },
        )
        response = self.provider.chat(request)
        result.provider_call = response.call
        structured = response.structured or {}
        allowed = set(task.edit_templates)
        stray = [edit for edit in structured.get("edits", []) if edit.get("path") not in allowed]
        if stray:
            result.rejected_reason = (
                f"the agent edited files outside the task: {[e.get('path') for e in stray]}"
            )
            result.duration_ms = (time.perf_counter() - started) * 1000
            return result

        if structured.get("abstained", False):
            result.abstained = True
            result.rejected_reason = str(structured.get("reason", "agent abstained"))
            result.duration_ms = (time.perf_counter() - started) * 1000
            return result

        resolved = str(structured.get("resolved_value", "")).strip()
        result.resolved_value = resolved
        result.used_stale_value = any(
            stale.lower() in resolved.lower() for stale in task.stale_values if stale
        )
        result.used_forbidden_value = any(
            forbidden.lower() in resolved.lower()
            for forbidden in task.forbidden_values
            if forbidden
        )

        edits = structured.get("edits", [])
        if not edits:
            result.rejected_reason = "the agent returned no file edits"
            result.duration_ms = (time.perf_counter() - started) * 1000
            return result

        sandbox = Path(tempfile.mkdtemp(prefix=f"memcheck_task_{task.task_id}_"))
        try:
            self._materialize(sandbox, task.repository)
            self._apply_edits(sandbox, edits)
            result.patch_applied = True
            completed = self._run_tests(sandbox, task)
            result.tests_run = True
            result.return_code = completed.returncode
            result.tests_passed = completed.returncode == 0
            result.stdout_tail = completed.stdout[-2000:]
            result.stderr_tail = completed.stderr[-2000:]
            if artifacts_dir is not None:
                self._save_artifacts(
                    artifacts_dir, task, backend_name, checkpoint, edits, completed
                )
        except PatchRejected as rejection:
            result.rejected_reason = str(rejection)
        except subprocess.TimeoutExpired:
            result.rejected_reason = f"test command exceeded {self.timeout_seconds}s"
            result.tests_run = True
            result.tests_passed = False
        finally:
            import shutil

            shutil.rmtree(sandbox, ignore_errors=True)
        result.duration_ms = (time.perf_counter() - started) * 1000
        return result

    # -- sandbox helpers ----------------------------------------------------------------
    @staticmethod
    def _materialize(sandbox: Path, repository: dict[str, str]) -> None:
        for relative, content in repository.items():
            target = sandbox / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    @staticmethod
    def _apply_edits(sandbox: Path, edits: list[dict[str, str]]) -> None:
        root = sandbox.resolve()
        for edit in edits:
            raw_path = str(edit.get("path", "")).strip()
            if not raw_path:
                raise PatchRejected("an edit has no path")
            target = (root / raw_path).resolve()
            if not target.is_relative_to(root):
                raise PatchRejected(f"edit path escapes the sandbox: {raw_path!r}")
            if "tests/" in raw_path or raw_path.startswith("tests"):
                raise PatchRejected(f"the agent may not modify tests: {raw_path!r}")
            content = edit.get("content")
            if not isinstance(content, str) or not content.strip():
                raise PatchRejected(f"edit for {raw_path!r} has no content")
            if "{{VALUE}}" in content:
                raise PatchRejected(
                    f"edit for {raw_path!r} still contains an unresolved placeholder"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def _run_tests(self, sandbox: Path, task: CodingTask) -> subprocess.CompletedProcess[str]:
        # The command is constructed here, never taken from the model.
        command = [ALLOWED_TEST_EXECUTABLE, *task.test_command]
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "http_proxy",
                "https_proxy",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
            }
        }
        environment["PYTHONPATH"] = str(sandbox)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            command,
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=environment,
            check=False,
        )

    @staticmethod
    def _save_artifacts(
        artifacts_dir: Path,
        task: CodingTask,
        backend_name: str,
        checkpoint: str,
        edits: list[dict[str, str]],
        completed: subprocess.CompletedProcess[str],
    ) -> None:
        target = artifacts_dir / f"{backend_name}_{checkpoint}_{task.task_id}"
        target.mkdir(parents=True, exist_ok=True)
        for edit in edits:
            safe_name = str(edit["path"]).replace("/", "__")
            (target / f"patch_{safe_name}").write_text(str(edit["content"]), encoding="utf-8")
        (target / "pytest_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (target / "pytest_stderr.txt").write_text(completed.stderr, encoding="utf-8")
