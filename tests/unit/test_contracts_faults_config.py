"""Contracts, faults, configuration validation, cost gating, and the claims registry."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from memcheck.config import (
    CORE_BACKENDS,
    ExperimentConfig,
    load_config,
    load_pricing,
    paid_run_authorization,
)
from memcheck.contracts import ALL_CONTRACTS, ContractContext, resolve_contracts
from memcheck.evaluation import cost as cost_module
from memcheck.evaluation.fairness import FairnessError, validate_conditions
from memcheck.evaluation.matrix import expand
from memcheck.faults import ALL_FAULTS, resolve_fault, run_fault
from memcheck.models import Severity


class TestContracts:
    def test_all_six_contracts_are_registered(self) -> None:
        assert len(ALL_CONTRACTS) == 6
        assert {contract.name for contract in ALL_CONTRACTS} == {
            "duplicate_replay_invariance",
            "restart_equivalence",
            "supersession_correctness",
            "deletion_closure",
            "tenant_isolation",
            "evidence_budget_compliance",
        }

    def test_every_contract_declares_its_required_fields(self) -> None:
        for contract_class in ALL_CONTRACTS:
            contract = contract_class()
            assert contract.description
            assert contract.preconditions
            assert contract.expected_invariant
            assert isinstance(contract.severity, Severity)

    def test_unknown_contract_name_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="unknown contract"):
            resolve_contracts(["not_a_contract"])

    @pytest.mark.parametrize("backend_name", list(CORE_BACKENDS))
    def test_contracts_run_against_every_backend(self, backend_name, backend_factory) -> None:
        context = ContractContext(
            run_id="test",
            backend_name=backend_name,
            backend_factory=backend_factory(backend_name),
            seed=1,
        )
        results = [contract().run(context) for contract in ALL_CONTRACTS]
        assert len(results) == 6
        for result in results:
            # Whatever the verdict, a contract must produce a usable result.
            assert result.contract_name
            if result.vacuous:
                # The scenario could not establish the precondition on this backend.
                assert result.vacuous_reason
                assert not result.violations
            elif not result.passed:
                assert result.violations
                for violation in result.violations:
                    assert violation.expected_behavior
                    assert violation.observed_behavior
                    assert violation.reproduction_steps

    def test_guarded_memory_satisfies_all_six(self, backend_factory) -> None:
        context = ContractContext(
            run_id="test",
            backend_name="guarded_memory",
            backend_factory=backend_factory("guarded_memory"),
            seed=1,
        )
        for contract in ALL_CONTRACTS:
            result = contract().run(context)
            assert result.passed, (
                f"{contract.name}: {[v.observed_behavior for v in result.violations]}"
            )

    def test_contracts_are_deterministic(self, backend_factory) -> None:
        def outcomes() -> list[tuple[str, bool, int]]:
            context = ContractContext(
                run_id="test",
                backend_name="transcript_rag",
                backend_factory=backend_factory("transcript_rag"),
                seed=1,
            )
            return [
                (contract.name, result.passed, len(result.violations))
                for contract in [c() for c in ALL_CONTRACTS]
                for result in [contract.run(context)]
            ]

        assert outcomes() == outcomes()


class TestFaults:
    def test_all_six_faults_are_registered(self) -> None:
        assert len(ALL_FAULTS) == 6

    def test_every_fault_declares_its_required_fields(self) -> None:
        for fault_class in ALL_FAULTS:
            fault = fault_class()
            assert fault.injection_point
            assert fault.activation_condition
            assert fault.expected_recovery_behavior

    def test_unknown_fault_name_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="unknown fault"):
            resolve_fault("not_a_fault")

    @pytest.mark.parametrize("fault_class", list(ALL_FAULTS))
    def test_faults_are_deterministic_under_a_fixed_seed(
        self, fault_class, tmp_path, retrieval, make_event
    ) -> None:
        from memcheck.backends import build_backend

        events = [
            make_event(1, "The primary database is postgres."),
            make_event(2, "The deployment target is docker-swarm."),
            make_event(3, "The deployment target is kubernetes."),
        ]

        def once(tag: str):
            backend = build_backend("guarded_memory", tmp_path / f"{tag}.db", retrieval)
            try:
                return run_fault(
                    fault_class(), backend, events, seed=42, backend_name="guarded_memory"
                )
            finally:
                backend.close()

        first, second = once("a"), once("b")
        assert first.recovered == second.recovered
        assert first.observed_recovery_behavior == second.observed_recovery_behavior

    def test_guarded_recovers_from_deletion_during_growth(
        self, tmp_path, retrieval, make_event
    ) -> None:
        from memcheck.backends import build_backend
        from memcheck.faults import DeletionDuringGrowth

        backend = build_backend("guarded_memory", tmp_path / "g.db", retrieval)
        try:
            report = run_fault(
                DeletionDuringGrowth(),
                backend,
                [make_event(1, "The primary database is postgres.")],
                seed=1,
                backend_name="guarded_memory",
            )
        finally:
            backend.close()
        assert report.recovered is True

    def test_transcript_rag_leaks_under_deletion_during_growth(
        self, tmp_path, retrieval, make_event
    ) -> None:
        from memcheck.backends import build_backend
        from memcheck.faults import DeletionDuringGrowth

        backend = build_backend("transcript_rag", tmp_path / "t.db", retrieval)
        try:
            report = run_fault(
                DeletionDuringGrowth(),
                backend,
                [make_event(1, "The primary database is postgres.")],
                seed=1,
                backend_name="transcript_rag",
            )
        finally:
            backend.close()
        assert report.recovered is False
        assert "resurfaced" in report.observed_recovery_behavior


class TestConfiguration:
    @pytest.mark.parametrize(
        "name",
        ["smoke.yaml", "repo_evolution.yaml", "experiment_matrix.yaml", "longmemeval.yaml"],
    )
    def test_shipped_configs_validate(self, name) -> None:
        config = load_config(Path("configs") / name)
        assert config.name
        assert config.backends

    def test_unknown_key_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            yaml.safe_dump({"name": "x", "workload": {"name": "synthetic_longitudinal"}, "oops": 1})
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            load_config(path)

    def test_unknown_backend_is_rejected(self) -> None:
        with pytest.raises(Exception, match="unknown backend"):
            ExperimentConfig.model_validate(
                {
                    "name": "x",
                    "backends": ["not_a_backend"],
                    "workload": {"name": "synthetic_longitudinal"},
                }
            )

    def test_ablations_require_the_guarded_backend(self) -> None:
        with pytest.raises(Exception, match="ablations apply to guarded_memory"):
            ExperimentConfig.model_validate(
                {
                    "name": "x",
                    "backends": ["transcript_rag"],
                    "ablations": ["abl_no_supersession"],
                    "workload": {"name": "synthetic_longitudinal"},
                }
            )

    def test_configuration_hash_is_stable_and_sensitive(self) -> None:
        first = load_config("configs/smoke.yaml")
        second = load_config("configs/smoke.yaml")
        assert first.configuration_hash() == second.configuration_hash()
        second.retrieval.token_budget += 1
        assert first.configuration_hash() != second.configuration_hash()


class TestFairness:
    def test_a_valid_matrix_passes(self) -> None:
        config = load_config("configs/smoke.yaml")
        report = validate_conditions(config, expand(config))
        assert report.ok, report.problems

    def test_condition_ids_are_stable_and_unique(self) -> None:
        config = load_config("configs/smoke.yaml")
        first, second = expand(config), expand(config)
        assert [c.condition_id for c in first] == [c.condition_id for c in second]
        assert len({c.condition_id for c in first}) == len(first)

    def test_undeclared_ablation_is_refused(self) -> None:
        from dataclasses import replace

        config = load_config("configs/smoke.yaml")
        conditions = expand(config)
        tampered = [
            replace(condition, declared_ablation=False) if condition.ablations else condition
            for condition in conditions
        ]
        report = validate_conditions(config, tampered)
        assert not report.ok
        with pytest.raises(FairnessError):
            report.raise_if_unfair()


class TestCostControls:
    def test_paid_runs_are_blocked_by_default(self) -> None:
        assert paid_run_authorization().allowed is False

    def test_both_controls_are_required(self, monkeypatch) -> None:
        monkeypatch.setenv("ALLOW_PAID_RUNS", "true")
        assert paid_run_authorization().allowed is False  # no ceiling
        monkeypatch.setenv("MAX_EXPERIMENT_COST_USD", "5.00")
        assert paid_run_authorization().allowed is True

    def test_non_numeric_ceiling_is_refused(self, monkeypatch) -> None:
        monkeypatch.setenv("ALLOW_PAID_RUNS", "true")
        monkeypatch.setenv("MAX_EXPERIMENT_COST_USD", "lots")
        assert paid_run_authorization().allowed is False

    def test_mock_provider_estimate_is_free_and_permitted(self) -> None:
        config = load_config("configs/smoke.yaml")
        estimate = cost_module.estimate(config, expand(config), query_count=10)
        assert estimate.usd == 0.0
        assert estimate.within_limit is True
        cost_module.enforce(estimate)

    def test_paid_estimate_is_refused_without_authorization(self) -> None:
        config = load_config("configs/smoke.yaml")
        config.model.provider = "anthropic"
        config.model.identifier = "claude-opus-5"
        estimate = cost_module.estimate(config, expand(config), query_count=10)
        assert estimate.usd > 0
        assert estimate.within_limit is False
        with pytest.raises(cost_module.CostLimitExceeded):
            cost_module.enforce(estimate)

    def test_pricing_for_an_unknown_model_raises_rather_than_assuming_zero(self) -> None:
        pricing = load_pricing("configs/pricing.example.yaml")
        with pytest.raises(KeyError, match="no pricing entry"):
            pricing.cost_for("openai:some-model", 1000, 1000)

    def test_building_a_paid_provider_is_refused(self) -> None:
        from memcheck.providers import ProviderError, build_provider

        config = load_config("configs/smoke.yaml")
        config.model.provider = "anthropic"
        with pytest.raises(ProviderError, match="ALLOW_PAID_RUNS"):
            build_provider(config)

    def test_environment_is_clean_for_tests(self) -> None:
        assert not os.environ.get("ANTHROPIC_API_KEY")
        assert not os.environ.get("OPENAI_API_KEY")


class TestFaultIdentityIsPerTenant:
    """Fault verification must scope fact identity to a tenant.

    Both the duplicate-amplification and out-of-order checks originally compared facts globally,
    so two tenants legitimately asserting the same fact were reported as a violation. The full
    matrix surfaced this as guarded_memory apparently failing out-of-order recovery on 54 keys
    while the per-tenant count was zero.
    """

    def _multi_tenant_events(self, make_event):
        return [
            make_event(1, "The deployment target is docker-swarm.", tenant_id="t1"),
            make_event(2, "The deployment target is docker-swarm.", tenant_id="t2"),
            make_event(3, "The deployment target is kubernetes.", tenant_id="t1", days=9),
            make_event(4, "The deployment target is kubernetes.", tenant_id="t2", days=9),
        ]

    def test_out_of_order_does_not_flag_two_tenants(self, tmp_path, retrieval, make_event) -> None:
        from memcheck.backends import build_backend
        from memcheck.faults import OutOfOrderDelivery

        backend = build_backend("guarded_memory", tmp_path / "oo.db", retrieval)
        try:
            report = run_fault(
                OutOfOrderDelivery(),
                backend,
                self._multi_tenant_events(make_event),
                seed=1,
                backend_name="guarded_memory",
            )
        finally:
            backend.close()
        assert report.recovered is True, report.observed_recovery_behavior
        assert "tenant" in report.observed_recovery_behavior

    def test_duplicate_delivery_does_not_flag_two_tenants(
        self, tmp_path, retrieval, make_event
    ) -> None:
        from memcheck.backends import build_backend
        from memcheck.faults import DuplicateEventDelivery

        backend = build_backend("guarded_memory", tmp_path / "dup.db", retrieval)
        try:
            report = run_fault(
                DuplicateEventDelivery(),
                backend,
                self._multi_tenant_events(make_event),
                seed=1,
                backend_name="guarded_memory",
            )
        finally:
            backend.close()
        assert report.recovered is True, report.observed_recovery_behavior
