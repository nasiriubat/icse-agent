"""Contracts executed on a snapshot of a grown store (DECISIONS D-029)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from memcheck.backends import build_backend, build_extractor
from memcheck.backends.base import MemoryBackend
from memcheck.config import RetrievalConfig, load_config
from memcheck.contracts import ContractContext
from memcheck.contracts.base import make_event
from memcheck.contracts.restart_equivalence import RestartEquivalence
from memcheck.evaluation import ExperimentRunner
from memcheck.evaluation.runner import snapshot_store


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in (path, path.with_name(path.name + "-wal")):
        if candidate.exists():
            digest.update(candidate.read_bytes())
    return digest.hexdigest()


class TestConfig:
    def test_default_is_fresh(self) -> None:
        assert load_config("configs/smoke.yaml").contract_store == "fresh"

    def test_aged_is_accepted_and_other_values_are_not(self) -> None:
        config = load_config("configs/smoke.yaml")
        aged = config.model_copy(update={"contract_store": "aged"})
        assert aged.contract_store == "aged"
        with pytest.raises(ValidationError):
            type(config).model_validate({**config.model_dump(), "contract_store": "stale"})


class TestSnapshot:
    def test_snapshot_carries_the_records_and_leaves_the_live_store_untouched(
        self, tmp_path: Path
    ) -> None:
        retrieval = RetrievalConfig()
        live_path = tmp_path / "live.db"
        live = build_backend(
            "guarded_memory", live_path, retrieval, extractor=build_extractor("rule_based")
        )
        try:
            for index, content in enumerate(
                [
                    "The primary database is mysql.",
                    "The deployment target is bare-metal.",
                    "The primary database is postgres.",
                ],
                start=1,
            ):
                live.ingest_event(make_event(index, content, tenant_id="tenant_00"))
            live.checkpoint()
            live_records = len(live.all_records())
            before = _digest(live_path)

            def factory(suffix: str) -> MemoryBackend:
                path = tmp_path / f"snapshot_{suffix}.db"
                snapshot_store(live_path, path)
                return build_backend(
                    "guarded_memory", path, retrieval, extractor=build_extractor("rule_based")
                )

            copy = factory("probe")
            try:
                assert len(copy.all_records()) == live_records > 0
            finally:
                copy.close()

            context = ContractContext(
                run_id="aged-unit",
                backend_name="guarded_memory",
                backend_factory=factory,
                store_state="aged",
            )
            result = RestartEquivalence().run(context)
            assert result.store_state == "aged"
            assert result.passed and not result.vacuous
            assert _digest(live_path) == before
            assert len(live.all_records()) == live_records
        finally:
            live.close()


class TestRunner:
    def test_aged_run_records_store_state_on_every_contract(self, tmp_path: Path) -> None:
        config = load_config("configs/smoke.yaml")
        config.seeds = [1]
        config.faults = ["none"]
        config.ablations = []
        config.backends = ["transcript_rag", "guarded_memory"]
        config.checkpoints = [1.0]
        config.workload.params["sessions"] = 3
        config.workload.params["tenants"] = 1
        config.contract_store = "aged"
        runner = ExperimentRunner(config, run_id="aged-itest", results_root=tmp_path)
        status = runner.run()
        assert status.state == "completed", status.error
        rows = [
            json.loads(line)
            for line in (runner.paths.root / "contracts.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert rows
        assert {row["store_state"] for row in rows} == {"aged"}
        guarded = [row for row in rows if row["backend_name"] == "guarded_memory"]
        assert all(row["passed"] for row in guarded)
