"""The guarded backend's write and retrieval lifecycles."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memcheck.backends import build_backend
from memcheck.backends.guarded import GuardedMemoryBackend, InterruptedWrite
from memcheck.models import AbstentionReason, EventType, MemoryStatus, WriteOutcome

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
QUERY = "What is the deployment target?"


class TestIdempotentWrites:
    def test_duplicate_event_is_ignored(self, guarded, make_event) -> None:
        event = make_event(1, "The deployment target is docker-swarm.")
        assert guarded.ingest_event(event).outcome is WriteOutcome.created
        assert guarded.ingest_event(event).outcome is WriteOutcome.duplicate_ignored
        assert len(guarded.active_records()) == 1

    def test_repeated_delivery_does_not_amplify(self, guarded, make_event) -> None:
        event = make_event(1, "The deployment target is docker-swarm.")
        for _ in range(10):
            guarded.ingest_event(event)
        assert len(guarded.active_records()) == 1

    def test_reasserting_the_same_fact_later_creates_no_new_record(
        self, guarded, make_event
    ) -> None:
        guarded.ingest_event(make_event(1, "The deployment target is docker-swarm."))
        guarded.ingest_event(make_event(5, "The deployment target is docker-swarm."))
        assert len(guarded.active_records()) == 1


class TestVersioningAndSupersession:
    def test_update_creates_a_version_and_supersedes(self, guarded, make_event) -> None:
        guarded.ingest_event(make_event(1, "The deployment target is docker-swarm."))
        result = guarded.ingest_event(make_event(5, "The deployment target is kubernetes."))
        assert result.outcome is WriteOutcome.version_created
        assert result.superseded_memory_ids
        active = guarded.active_records()
        assert len(active) == 1
        assert active[0].version == 2
        assert "kubernetes" in active[0].content

    def test_superseded_record_keeps_a_closed_interval(self, guarded, make_event) -> None:
        guarded.ingest_event(make_event(1, "The deployment target is docker-swarm."))
        guarded.ingest_event(make_event(5, "The deployment target is kubernetes."))
        superseded = [
            record for record in guarded.all_records() if record.status is MemoryStatus.superseded
        ]
        assert len(superseded) == 1
        assert superseded[0].valid_to == EPOCH + timedelta(days=5)

    def test_current_query_returns_only_the_new_value(self, guarded, make_event) -> None:
        guarded.ingest_event(make_event(1, "The deployment target is docker-swarm."))
        guarded.ingest_event(make_event(5, "The deployment target is kubernetes."))
        contents = [item.content for item in guarded.retrieve(QUERY, "tenant_a").evidence]
        assert any("kubernetes" in text for text in contents)
        assert not any("docker-swarm" in text for text in contents)

    def test_historical_query_returns_the_old_value(self, guarded, make_event) -> None:
        guarded.ingest_event(make_event(1, "The deployment target is docker-swarm."))
        guarded.ingest_event(make_event(5, "The deployment target is kubernetes."))
        result = guarded.retrieve(QUERY, "tenant_a", as_of_time=EPOCH + timedelta(days=3))
        contents = [item.content for item in result.evidence]
        assert any("docker-swarm" in text for text in contents)
        assert not any("kubernetes" in text for text in contents)

    def test_a_later_fact_resolves_an_earlier_contradiction(self, guarded, make_event) -> None:
        # Two values at the same instant cannot be ordered by the evidence...
        guarded.ingest_event(make_event(1, "The deployment target is docker-swarm.", days=2))
        guarded.ingest_event(
            make_event(2, "The deployment target is nomad.", days=2, event_id="evt_conflict")
        )
        assert guarded.retrieve(QUERY, "tenant_a").abstained is True
        # ...but a strictly later assertion is exactly the evidence that settles it.
        guarded.ingest_event(make_event(3, "The deployment target is kubernetes.", days=9))
        resolved = guarded.retrieve(QUERY, "tenant_a")
        assert resolved.abstained is False
        assert any("kubernetes" in item.content for item in resolved.evidence)


class TestDeletion:
    def _seed(self, backend, make_event):
        backend.ingest_event(make_event(1, "The rotation key is hunter2."))
        backend.ingest_event(make_event(2, "The primary database is postgres."))

    def test_deletion_request_event_tombstones_the_fact(self, guarded, make_event) -> None:
        self._seed(guarded, make_event)
        guarded.ingest_event(
            make_event(
                3,
                "Delete everything about the rotation key.",
                event_type=EventType.deletion_request,
                metadata={"deletion_target": "key rotation"},
            )
        )
        assert guarded.retrieve("What is the rotation key?", "tenant_a").abstained is True

    def test_deletion_purges_index_summaries_and_cache(self, guarded, make_event) -> None:
        self._seed(guarded, make_event)
        guarded.retrieve("What is the rotation key?", "tenant_a")  # populate the cache
        result = guarded.delete("key rotation", "tenant_a")
        assert result.deleted_memory_ids
        assert result.purged_index_rows >= 1
        assert result.tombstone_ids
        assert not guarded.index.contains_text("hunter2")
        assert not any("hunter2" in text for text in guarded.summary_texts())
        assert not any("hunter2" in text for text in guarded.cache_texts())

    def test_unrelated_facts_survive_a_deletion(self, guarded, make_event) -> None:
        self._seed(guarded, make_event)
        guarded.delete("key rotation", "tenant_a")
        survivor = guarded.retrieve("What is the primary database?", "tenant_a")
        assert any("postgres" in item.content for item in survivor.evidence)

    def test_deletion_survives_restart_and_rebuild(self, guarded, make_event) -> None:
        self._seed(guarded, make_event)
        guarded.delete("key rotation", "tenant_a")
        guarded.restart()
        assert guarded.retrieve("What is the rotation key?", "tenant_a").abstained is True
        guarded.rebuild_indexes()
        assert not guarded.index.contains_text("hunter2")

    def test_a_deleted_key_is_not_resurrected_by_a_later_write(self, guarded, make_event) -> None:
        self._seed(guarded, make_event)
        guarded.delete("key rotation", "tenant_a")
        guarded.ingest_event(make_event(9, "The rotation key is hunter3."))
        assert guarded.retrieve("What is the rotation key?", "tenant_a").abstained is True


class TestRecovery:
    def test_interrupted_write_is_durable_in_the_log(self, guarded, make_event) -> None:
        event = make_event(1, "The escalation contact is oncall-team.")
        with pytest.raises(InterruptedWrite):
            guarded.ingest_event(event, abort_after="log_append")
        assert guarded.log.get("tenant_a", event.event_id) is not None
        assert guarded.retrieve("What is the escalation contact?", "tenant_a").abstained is True

    def test_restart_replays_the_interrupted_write(self, guarded, make_event) -> None:
        event = make_event(1, "The escalation contact is oncall-team.")
        with pytest.raises(InterruptedWrite):
            guarded.ingest_event(event, abort_after="log_append")
        guarded.restart()
        result = guarded.retrieve("What is the escalation contact?", "tenant_a")
        assert result.abstained is False
        assert any("oncall-team" in item.content for item in result.evidence)

    def test_rebuild_reconstructs_the_index_from_the_log(self, guarded, make_event) -> None:
        for index in range(1, 6):
            guarded.ingest_event(make_event(index, f"The service {index} owner is team{index}."))
        before = guarded.index.document_count()
        guarded.index.clear()
        assert guarded.index.document_count() == 0
        guarded.rebuild_indexes()
        assert guarded.index.document_count() == before


class TestBudgetsAndAbstention:
    def test_evidence_count_budget_is_enforced(self, guarded, make_event) -> None:
        for index in range(1, 21):
            guarded.ingest_event(
                make_event(index, f"The service level objective {index} is tier-{index}.")
            )
        result = guarded.retrieve(
            "What is the service level objective?", "tenant_a", evidence_budget=2
        )
        assert len(result.evidence) <= 2

    def test_token_budget_is_enforced_and_reported(self, guarded, make_event) -> None:
        for index in range(1, 21):
            guarded.ingest_event(
                make_event(index, f"The service level objective {index} is tier-{index}.")
            )
        result = guarded.retrieve(
            "What is the service level objective?", "tenant_a", token_budget=12
        )
        assert result.injected_tokens <= 12
        assert result.injected_tokens == sum(item.token_count for item in result.evidence)

    def test_abstains_with_no_evidence(self, guarded) -> None:
        result = guarded.retrieve("What is the deployment target?", "tenant_a")
        assert result.abstained is True
        assert result.abstention_reason is AbstentionReason.no_evidence

    def test_evidence_carries_provenance_and_validity(self, guarded, make_event) -> None:
        guarded.ingest_event(make_event(1, "The deployment target is kubernetes."))
        item = guarded.retrieve(QUERY, "tenant_a").evidence[0]
        assert item.source_event_ids == ["evt_0001"]
        assert item.valid_from == EPOCH + timedelta(days=1)
        assert item.retrieval_reason


class TestTenantIsolation:
    def test_retrieval_requires_a_tenant(self, guarded) -> None:
        with pytest.raises(Exception, match="tenant"):
            guarded.retrieve(QUERY, "")

    def test_one_tenant_never_sees_another(self, guarded, make_event) -> None:
        guarded.ingest_event(make_event(1, "The roadmap codename is acme-x.", tenant_id="a"))
        guarded.ingest_event(make_event(2, "The roadmap codename is globex-y.", tenant_id="b"))
        for tenant, own, foreign in (("a", "acme-x", "globex-y"), ("b", "globex-y", "acme-x")):
            evidence = guarded.retrieve("What is the roadmap codename?", tenant).evidence
            joined = " ".join(item.content for item in evidence)
            assert own in joined
            assert foreign not in joined


class TestAblations:
    def test_unknown_ablation_is_rejected(self, tmp_path, retrieval) -> None:
        with pytest.raises(Exception, match="unknown ablation"):
            GuardedMemoryBackend(tmp_path / "x.db", ablations=["abl_not_real"])

    def test_ablations_are_rejected_for_other_backends(self, tmp_path, retrieval) -> None:
        with pytest.raises(Exception, match="ablations are only defined"):
            build_backend(
                "transcript_rag", tmp_path / "y.db", retrieval, ablations=["abl_no_supersession"]
            )

    def test_no_supersession_leaves_both_versions_open(
        self, tmp_path, retrieval, make_event
    ) -> None:
        backend = build_backend(
            "guarded_memory",
            tmp_path / "abl.db",
            retrieval,
            ablations=["abl_no_supersession"],
        )
        try:
            backend.ingest_event(make_event(1, "The deployment target is docker-swarm."))
            backend.ingest_event(make_event(5, "The deployment target is kubernetes."))
            active = backend.active_records()
            assert len(active) == 2
            assert all(record.valid_to is None for record in active)
        finally:
            backend.close()

    def test_no_idempotency_stores_the_duplicate(self, tmp_path, retrieval, make_event) -> None:
        backend = build_backend(
            "guarded_memory",
            tmp_path / "abl2.db",
            retrieval,
            ablations=["abl_no_idempotency"],
        )
        try:
            event = make_event(1, "The cache layer is redis.", event_id="a")
            backend.ingest_event(event)
            backend.ingest_event(make_event(1, "The cache layer is redis.", event_id="b"))
            assert backend.log.count() == 2
        finally:
            backend.close()


class TestOutOfOrderDelivery:
    """A fact that arrives late is history, not a contradiction.

    Found by the out-of-order fault on the full matrix: treating an older late arrival as a
    conflict left the key permanently unresolved, because idempotency then made the corrective
    replay a no-op. Its own timestamp settles the order.
    """

    def test_late_older_fact_becomes_history(self, guarded, make_event) -> None:
        guarded.ingest_event(make_event(2, "The deployment target is kubernetes.", days=5))
        guarded.ingest_event(make_event(1, "The deployment target is docker-swarm.", days=1))

        active = guarded.active_records()
        assert len(active) == 1
        assert "kubernetes" in active[0].content

        history = [r for r in guarded.all_records() if r.status is MemoryStatus.superseded]
        assert len(history) == 1
        assert history[0].valid_to == EPOCH + timedelta(days=5)

    def test_current_and_historical_queries_are_both_right(self, guarded, make_event) -> None:
        guarded.ingest_event(make_event(2, "The deployment target is kubernetes.", days=5))
        guarded.ingest_event(make_event(1, "The deployment target is docker-swarm.", days=1))

        current = guarded.retrieve(QUERY, "tenant_a")
        assert current.abstained is False
        assert all("kubernetes" in item.content for item in current.evidence)

        historical = guarded.retrieve(QUERY, "tenant_a", as_of_time=EPOCH + timedelta(days=3))
        assert any("docker-swarm" in item.content for item in historical.evidence)

    def test_same_instant_is_still_a_contradiction(self, guarded, make_event) -> None:
        guarded.ingest_event(make_event(1, "The deployment target is kubernetes.", days=4))
        guarded.ingest_event(
            make_event(2, "The deployment target is nomad.", days=4, event_id="evt_same_instant")
        )
        assert guarded.retrieve(QUERY, "tenant_a").abstained is True
