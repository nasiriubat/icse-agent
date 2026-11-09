"""The experiment runner.

Guarantees it enforces, in order:

1. Fairness is validated before anything runs.
2. Cost is estimated and compared to the limit before anything is spent.
3. Every observation is streamed to ``results/raw/<run_id>/`` as it happens, so a crash keeps
   the work already done.
4. Raw outputs are written once and then checksummed; nothing reopens them for writing.
5. A resumed run skips conditions already recorded as complete and never recomputes them.
"""

from __future__ import annotations

import json
import platform
import shutil
import sqlite3
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from memcheck.agents.answer_agent import AnswerAgent, JudgeAgent
from memcheck.agents.coding_agent import CodingAgent
from memcheck.backends import build_backend, build_extractor
from memcheck.backends.base import MemoryBackend
from memcheck.backends.extraction import Extractor, LLMExtractor
from memcheck.backends.retrieval import RANKER_IDENTITY
from memcheck.config import ExperimentConfig
from memcheck.contracts import ContractContext, resolve_contracts
from memcheck.evaluation import cost as cost_module
from memcheck.evaluation.fairness import validate_conditions
from memcheck.evaluation.matrix import Condition, expand
from memcheck.faults import FaultContext, resolve_fault
from memcheck.models import (
    ExperimentRunManifest,
    ProviderCall,
    RunStatus,
    StoreStatistics,
    content_hash,
    utcnow,
)
from memcheck.providers import build_provider
from memcheck.providers.prompts import all_prompt_hashes
from memcheck.utilities.environment import (
    cpu_information,
    dependency_lock_hash,
    git_commit,
    gpu_information,
    operating_system,
    python_version,
    repo_root,
    working_tree_status,
)
from memcheck.utilities.hashing import hash_directory
from memcheck.utilities.jsonl import JsonlWriter
from memcheck.workloads import build_workload
from memcheck.workloads.base import Checkpoint, Query, Workload, grade_answer, query_is_asked

ARTIFACT_FILES = (
    "events.jsonl",
    "writes.jsonl",
    "retrievals.jsonl",
    "answers.jsonl",
    "grades.jsonl",
    "violations.jsonl",
    "costs.jsonl",
    "latency.jsonl",
    "store_statistics.jsonl",
    "faults.jsonl",
    "contracts.jsonl",
    "coding_tasks.jsonl",
)


@dataclass
class RunPaths:
    root: Path

    def artifact(self, name: str) -> Path:
        return self.root / name


def snapshot_store(source: Path, destination: Path) -> None:
    """Copy a live SQLite store to ``destination`` through a read-only connection.

    The online backup API reads the committed state, write-ahead log included, without taking a
    write lock, so the condition's store is unchanged by the copy.
    """
    reader = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        writer = sqlite3.connect(str(destination))
        try:
            reader.backup(writer)
        finally:
            writer.close()
    finally:
        reader.close()


class ExperimentRunner:
    """Executes a configuration into an immutable run directory."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        run_id: str | None = None,
        results_root: Path | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.results_root = results_root or Path(config.output.results_dir)
        self.run_id = run_id or self._make_run_id()
        self.paths = RunPaths(self.results_root / self.run_id)
        self.on_progress = on_progress or (lambda message: None)
        self._workspace = Path(tempfile.mkdtemp(prefix=f"memcheck_{self.run_id}_"))

    def _make_run_id(self) -> str:
        stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
        return f"{self.config.name}_{stamp}_{self.config.configuration_hash()[:8]}"

    # -- orchestration ------------------------------------------------------------------
    def run(self, *, resume: bool = False) -> RunStatus:
        conditions = expand(self.config)
        validate_conditions(self.config, conditions).raise_if_unfair()
        self.on_progress(f"fairness validated across {len(conditions)} condition(s)")

        workload = build_workload(self.config)
        workload.load()
        queries = workload.queries()
        # A seed-sensitive workload generates a different stream per seed, so each seed is a
        # separate sample. Building one stream and reusing it would make seeds replicas.
        workloads: dict[int, tuple[Workload, list[Query]]] = {}
        if workload.seed_sensitive:
            for seed in sorted({condition.seed for condition in conditions}):
                seeded = build_workload(self.config, seed=seed)
                seeded.load()
                workloads[seed] = (seeded, seeded.queries())

        estimate = cost_module.estimate_for_workload(self.config, conditions, workload)
        self.on_progress(estimate.render())
        cost_module.enforce(estimate)

        self.paths.root.mkdir(parents=True, exist_ok=True)
        status = self._load_status(resume=resume, conditions=conditions)
        status.estimated_cost_usd = estimate.usd

        self._write_config_copy()
        manifest = self._build_manifest(workload, conditions)
        self._write_manifest(manifest)

        provider = build_provider(self.config, cache_path=self._workspace / "provider_cache.db")
        agent = AnswerAgent(
            provider,
            self.config.model,
            abstain_on_conflict=self.config.retrieval.abstain_on_conflict,
        )
        self._coding_agent = CodingAgent(provider, self.config.model)
        # Queries whose gold answer is free text are judged semantically; the judge shares the
        # provider, so its calls are cost-recorded and cache-deduplicated like any other.
        self._judge = JudgeAgent(provider, self.config.model)
        # Retained so each condition can be given its own extractor: the write policy may issue
        # provider calls, and those must be attributed to the condition that made them.
        self._provider = provider

        writers = {name: JsonlWriter(self.paths.artifact(name)) for name in ARTIFACT_FILES}
        stdout_log = self.paths.artifact("stdout.log").open("a", encoding="utf-8")
        stderr_log = self.paths.artifact("stderr.log").open("a", encoding="utf-8")

        try:
            if not resume:
                if workloads:
                    for seed, (seeded, _) in workloads.items():
                        writers["events.jsonl"].write_many(
                            [
                                {"seed": seed, **event.model_dump(mode="json")}
                                for event in seeded.events()
                            ]
                        )
                else:
                    writers["events.jsonl"].write_many(workload.events())  # type: ignore[arg-type]

            for condition in conditions:
                if condition.condition_id in status.completed_condition_ids:
                    self.on_progress(f"skip (already complete): {condition.label}")
                    stdout_log.write(f"skip {condition.condition_id} {condition.label}\n")
                    continue
                self.on_progress(f"running: {condition.label}")
                stdout_log.write(f"start {condition.condition_id} {condition.label}\n")
                stdout_log.flush()
                try:
                    condition_workload, condition_queries = workloads.get(
                        condition.seed, (workload, queries)
                    )
                    actual_cost = self._run_condition(
                        condition, condition_workload, condition_queries, agent, writers
                    )
                    status.completed_condition_ids.append(condition.condition_id)
                    status.conditions_completed += 1
                    status.actual_cost_usd += actual_cost
                except Exception as error:
                    status.failed_condition_ids.append(condition.condition_id)
                    status.conditions_failed += 1
                    stderr_log.write(
                        f"condition {condition.condition_id} ({condition.label}) failed: "
                        f"{error!r}\n{traceback.format_exc()}\n"
                    )
                    stderr_log.flush()
                    self.on_progress(f"FAILED: {condition.label}: {error}")
                finally:
                    self._write_status(status)

            status.state = "completed" if not status.conditions_failed else "partial"
            status.finished_at = utcnow()
            self._write_status(status)
        finally:
            for writer in writers.values():
                writer.close()
            stdout_log.close()
            stderr_log.close()
            provider.close()
            self._finalize_checksums()
            shutil.rmtree(self._workspace, ignore_errors=True)
        return status

    # -- one condition ------------------------------------------------------------------
    def _run_condition(
        self,
        condition: Condition,
        workload: Workload,
        queries: list[Query],
        agent: AnswerAgent,
        writers: dict[str, JsonlWriter],
    ) -> float:
        events = workload.events()
        fault = resolve_fault(condition.fault)
        fault_context = FaultContext(seed=condition.seed, backend_name=condition.backend)
        delivered = fault.wrap_events(events, fault_context)
        checkpoints = workload.checkpoints(self.config.checkpoints)
        # Checkpoint indices are computed on the *delivered* stream so that, for example, a
        # duplicate-delivery run still measures at the same fractions of its own stream.
        scale = len(delivered) / max(1, len(events))
        scaled = [
            Checkpoint(
                label=checkpoint.label,
                fraction=checkpoint.fraction,
                event_index=min(len(delivered), max(1, round(checkpoint.event_index * scale))),
            )
            for checkpoint in checkpoints
        ]

        backend_path = self._workspace / f"{condition.condition_id}.db"
        # A fresh extractor per condition, so any provider calls it makes belong to this
        # condition rather than to whichever one happened to run first.
        extractor = self._build_extractor(condition.backend)
        backend = build_backend(
            condition.backend,
            backend_path,
            self.config.retrieval,
            ablations=list(condition.ablations),
            extractor=extractor,
        )
        actual_cost = 0.0
        try:
            position = 0
            for checkpoint in scaled:
                # Ingest up to this checkpoint.
                for event in delivered[position : checkpoint.event_index]:
                    write_result = backend.ingest_event(event)
                    write_result_payload = write_result.model_dump(mode="json")
                    write_result_payload["condition_id"] = condition.condition_id
                    write_result_payload["checkpoint"] = checkpoint.label
                    writers["writes.jsonl"].write(write_result_payload)
                    writers["latency.jsonl"].write(
                        {
                            "condition_id": condition.condition_id,
                            "checkpoint": checkpoint.label,
                            "phase": "ingestion",
                            "latency_ms": write_result.latency_ms,
                            "event_id": event.event_id,
                            "backend": condition.backend,
                        }
                    )
                position = checkpoint.event_index
                backend.checkpoint()

                # Perturb and recover at the final checkpoint, where the store is largest.
                if checkpoint is scaled[-1] and condition.fault != "none":
                    fault.perturb(backend, fault_context)
                    fault.recover(backend, fault_context)
                    recovered, observed = fault.verify(backend, fault_context)
                    report = fault.report(fault_context, recovered, observed)
                    payload = report.model_dump(mode="json")
                    payload["condition_id"] = condition.condition_id
                    payload["checkpoint"] = checkpoint.label
                    writers["faults.jsonl"].write(payload)

                actual_cost += self._evaluate_checkpoint(
                    condition, backend, workload, queries, checkpoint, agent, writers
                )
                actual_cost += self._evaluate_coding_tasks(
                    condition, backend, workload, checkpoint, writers
                )

            self._run_contracts(condition, writers, backend_path)
            actual_cost += self._record_extraction_cost(condition, extractor, writers)
        finally:
            backend.close()
        return actual_cost

    def _build_extractor(self, backend_name: str) -> Extractor | None:
        """Build the configured write policy, or ``None`` for a backend that has none.

        Only the guarded backend extracts memories; the baselines store transcripts or nothing,
        and handing them an extractor would imply a write policy they do not have.
        """
        if backend_name != "guarded_memory":
            return None
        if self.config.extraction.policy == "rule_based":
            return build_extractor("rule_based")
        return build_extractor(
            "llm",
            provider=self._provider,
            model_identifier=self.config.extraction.model_identifier
            or self.config.model.identifier,
            temperature=self.config.model.temperature,
            # The run's own budget, not the extractor's default: a reasoning model spends this
            # on reasoning before emitting anything, so it must match what the run configured.
            max_output_tokens=self.config.model.max_output_tokens,
        )

    def _record_extraction_cost(
        self,
        condition: Condition,
        extractor: Extractor | None,
        writers: dict[str, JsonlWriter],
    ) -> float:
        """Charge write-side provider calls to the condition, under the ingestion checkpoint.

        Extraction cost is real and is paid once per event rather than once per query, so
        folding it into the answering cost would misattribute where a configuration spends.
        """
        if not isinstance(extractor, LLMExtractor):
            return 0.0
        cost = 0.0
        for call in extractor.calls:
            if not isinstance(call, ProviderCall):
                continue
            cost += call.estimated_cost_usd
            writers["costs.jsonl"].write(
                {
                    "condition_id": condition.condition_id,
                    "checkpoint": "ingestion",
                    "query_id": None,
                    **call.model_dump(mode="json"),
                }
            )
        if extractor.malformed_responses:
            # A response we could not parse is indistinguishable from a model that correctly
            # decided to remember nothing, so it is surfaced rather than absorbed.
            self.on_progress(
                f"{condition.condition_id}: {extractor.malformed_responses} extraction "
                "response(s) could not be parsed and yielded no candidates"
            )
        return cost

    def _extractor_identity(self) -> str:
        """What the manifest records as the run's write policy.

        Built without a provider, because this is called before one exists and an identity is a
        property of the configuration rather than of a live object.
        """
        if self.config.extraction.policy == "rule_based":
            return "rule_copula_v1"
        model = self.config.extraction.model_identifier or self.config.model.identifier
        return f"llm:{model}:extract_v1"

    def _evaluate_checkpoint(
        self,
        condition: Condition,
        backend: MemoryBackend,
        workload: Workload,
        queries: list[Query],
        checkpoint: Checkpoint,
        agent: AnswerAgent,
        writers: dict[str, JsonlWriter],
    ) -> float:
        cost = 0.0
        for query in queries:
            if not query_is_asked(query, checkpoint):
                continue
            retrieval = backend.retrieve(
                query.text,
                query.tenant_id,
                as_of_time=query.as_of_time,
                token_budget=self.config.retrieval.token_budget,
                evidence_budget=self.config.retrieval.evidence_budget,
                query_id=query.query_id,
            )
            retrieval.checkpoint = checkpoint.label
            retrieval_payload = retrieval.model_dump(mode="json")
            retrieval_payload["condition_id"] = condition.condition_id
            retrieval_payload["query_id"] = query.query_id
            writers["retrievals.jsonl"].write(retrieval_payload)

            answer = agent.answer(
                query,
                retrieval,
                checkpoint=checkpoint.label,
                backend_name=condition.backend,
            )
            answer_payload = answer.model_dump(mode="json")
            answer_payload["condition_id"] = condition.condition_id
            writers["answers.jsonl"].write(answer_payload)

            judged_before = len(self._judge.calls)
            grade = grade_answer(query, answer, retrieval, judge=self._judge)
            for call in self._judge.calls[judged_before:]:
                cost += call.estimated_cost_usd
                writers["costs.jsonl"].write(
                    {
                        "condition_id": condition.condition_id,
                        "checkpoint": checkpoint.label,
                        "query_id": query.query_id,
                        **call.model_dump(mode="json"),
                    }
                )
            grade_payload = grade.model_dump(mode="json")
            grade_payload["condition_id"] = condition.condition_id
            grade_payload["ablations"] = list(condition.ablations)
            grade_payload["fault"] = condition.fault
            grade_payload["seed"] = condition.seed
            writers["grades.jsonl"].write(grade_payload)

            if answer.provider_call is not None:
                call = answer.provider_call
                cost += call.estimated_cost_usd
                writers["costs.jsonl"].write(
                    {
                        "condition_id": condition.condition_id,
                        "checkpoint": checkpoint.label,
                        "query_id": query.query_id,
                        **call.model_dump(mode="json"),
                    }
                )
            writers["latency.jsonl"].write(
                {
                    "condition_id": condition.condition_id,
                    "checkpoint": checkpoint.label,
                    "phase": "retrieval",
                    "latency_ms": retrieval.retrieval_latency_ms,
                    "query_id": query.query_id,
                    "backend": condition.backend,
                }
            )
            writers["latency.jsonl"].write(
                {
                    "condition_id": condition.condition_id,
                    "checkpoint": checkpoint.label,
                    "phase": "answer",
                    "latency_ms": answer.answer_latency_ms,
                    "query_id": query.query_id,
                    "backend": condition.backend,
                }
            )

        for tenant_id in [None, *workload.tenants()]:
            statistics: StoreStatistics = backend.get_statistics(tenant_id=tenant_id)
            statistics.checkpoint = checkpoint.label
            payload = statistics.model_dump(mode="json")
            payload["condition_id"] = condition.condition_id
            payload["fault"] = condition.fault
            payload["ablations"] = list(condition.ablations)
            writers["store_statistics.jsonl"].write(payload)
        return cost

    def _evaluate_coding_tasks(
        self,
        condition: Condition,
        backend: MemoryBackend,
        workload: Workload,
        checkpoint: Checkpoint,
        writers: dict[str, JsonlWriter],
    ) -> float:
        """Run the workload's coding tasks, graded by the repository's own unit tests."""
        tasks = workload.coding_tasks()
        if not tasks:
            return 0.0
        artifacts_dir = self.paths.root / "coding_artifacts"
        cost = 0.0
        for task in tasks:
            if task.available_from_fraction > checkpoint.fraction + 1e-9:
                continue
            retrieval = backend.retrieve(
                task.memory_query,
                task.tenant_id,
                token_budget=self.config.retrieval.token_budget,
                evidence_budget=self.config.retrieval.evidence_budget,
                query_id=task.task_id,
            )
            result = self._coding_agent.run_task(
                task,
                retrieval,
                backend_name=condition.backend,
                checkpoint=checkpoint.label,
                artifacts_dir=artifacts_dir,
            )
            payload = result.model_dump(mode="json")
            payload["condition_id"] = condition.condition_id
            payload["ablations"] = list(condition.ablations)
            payload["fault"] = condition.fault
            payload["seed"] = condition.seed
            writers["coding_tasks.jsonl"].write(payload)
            if result.provider_call is not None:
                cost += result.provider_call.estimated_cost_usd
        return cost

    def _run_contracts(
        self, condition: Condition, writers: dict[str, JsonlWriter], live_path: Path
    ) -> None:
        contracts = resolve_contracts(self.config.contracts)
        store_state = self.config.contract_store

        def factory(suffix: str) -> MemoryBackend:
            # Contracts always run with the rule-based extractor, whatever the run's write
            # policy is. They probe lifecycle invariants using scenarios whose phrasing is
            # controlled; substituting a model would make a contract result depend on
            # extraction quality, which is exactly the confound the study exists to separate.
            path = self._workspace / f"contract_{condition.condition_id}_{suffix}.db"
            if store_state == "aged":
                snapshot_store(live_path, path)
            return build_backend(
                condition.backend,
                path,
                self.config.retrieval,
                ablations=list(condition.ablations),
                extractor=build_extractor("rule_based")
                if condition.backend == "guarded_memory"
                else None,
            )

        context = ContractContext(
            run_id=self.run_id,
            backend_name=condition.backend,
            backend_factory=factory,
            seed=condition.seed,
            evidence_budget=self.config.retrieval.evidence_budget,
            token_budget=self.config.retrieval.token_budget,
            fault_name=condition.fault if condition.fault != "none" else None,
            condition_id=condition.condition_id,
            store_state=store_state,
        )
        for contract in contracts:
            result = contract.run(context)
            payload = result.model_dump(mode="json")
            payload["condition_id"] = condition.condition_id
            payload["fault"] = condition.fault
            payload["ablations"] = list(condition.ablations)
            payload["seed"] = condition.seed
            writers["contracts.jsonl"].write(payload)
            for violation in result.violations:
                violation_payload = violation.model_dump(mode="json")
                violation_payload["ablations"] = list(condition.ablations)
                violation_payload["seed"] = condition.seed
                writers["violations.jsonl"].write(violation_payload)

    # -- manifest, status, integrity ----------------------------------------------------
    def _build_manifest(
        self, workload: Workload, conditions: list[Condition]
    ) -> ExperimentRunManifest:
        root = repo_root()
        return ExperimentRunManifest(
            run_id=self.run_id,
            git_commit=git_commit(root),
            working_tree_status=working_tree_status(root),
            operating_system=operating_system(),
            python_version=python_version(),
            dependency_lock_hash=dependency_lock_hash(root),
            cpu_information=cpu_information(),
            gpu_information=gpu_information(),
            backend=",".join(sorted({condition.backend for condition in conditions})),
            model_provider=self.config.model.provider,
            model_identifier=self.config.model.identifier,
            embedding_identifier=self.config.embedding.identifier,
            temperature=self.config.model.temperature,
            random_seeds=list(self.config.seeds),
            prompt_hashes=all_prompt_hashes(),
            benchmark_name=self.config.workload.name,
            benchmark_version=self.config.workload.version,
            benchmark_parameters_hash=content_hash(
                {
                    "subset": self.config.workload.subset,
                    "params": self.config.workload.params,
                }
            ),
            configuration_hash=self.config.configuration_hash(),
            token_budget=self.config.retrieval.token_budget,
            evidence_budget=self.config.retrieval.evidence_budget,
            cost_limit_usd=self.config.cost.limit_usd,
            checkpoint_schedule=self.config.checkpoint_labels,
            repetition_number=self.config.repetitions,
            attempt=self.config.attempt,
            contract_store=self.config.contract_store,
            condition_id=",".join(condition.condition_id for condition in conditions),
            retrieval_mode=f"{self.config.retrieval.mode}:{RANKER_IDENTITY}",
            extractor_identity=self._extractor_identity(),
            ablations=list(self.config.ablations),
            faults=list(self.config.faults),
            notes=(f"platform={platform.platform()}; workload={json.dumps(workload.describe())}"),
        )

    def _write_manifest(self, manifest: ExperimentRunManifest) -> None:
        path = self.paths.artifact("manifest.json")
        path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def _write_config_copy(self) -> None:
        payload = self.config.model_dump(mode="json")
        self.paths.artifact("config.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
        )

    def _load_status(self, *, resume: bool, conditions: list[Condition]) -> RunStatus:
        path = self.paths.artifact("run_status.json")
        if resume and path.exists():
            status = RunStatus.model_validate(json.loads(path.read_text(encoding="utf-8")))
            status.state = "running"
            status.conditions_total = len(conditions)
            return status
        return RunStatus(run_id=self.run_id, state="running", conditions_total=len(conditions))

    def _write_status(self, status: RunStatus) -> None:
        self.paths.artifact("run_status.json").write_text(
            json.dumps(status.model_dump(mode="json"), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def _finalize_checksums(self) -> None:
        """Content-address the run directory so later modification is detectable."""
        checksums = hash_directory(self.paths.root, exclude=("checksums.json",))
        self.paths.artifact("checksums.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "algorithm": "sha256",
                    "generated_at": utcnow().isoformat(),
                    "files": checksums,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def verify_integrity(run_dir: Path) -> tuple[bool, list[str]]:
    """Re-check a run directory against its recorded checksums."""
    checksum_path = run_dir / "checksums.json"
    if not checksum_path.exists():
        return False, [f"{run_dir.name}: no checksums.json; integrity cannot be verified"]
    recorded = json.loads(checksum_path.read_text(encoding="utf-8"))["files"]
    current = hash_directory(run_dir, exclude=("checksums.json",))
    problems: list[str] = []
    for name, digest in recorded.items():
        if name not in current:
            problems.append(f"{run_dir.name}/{name}: missing")
        elif current[name] != digest:
            problems.append(f"{run_dir.name}/{name}: content changed since the run finished")
    for name in current:
        if name not in recorded:
            problems.append(f"{run_dir.name}/{name}: added after the run finished")
    return not problems, problems


def load_manifest(run_dir: Path) -> ExperimentRunManifest:
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return ExperimentRunManifest.model_validate(payload)


def discover_runs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    )
