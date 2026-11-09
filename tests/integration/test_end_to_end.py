"""End-to-end integration tests using only local and mock components.

These exercise the runner, the immutability guarantees, resumability, reporting, the claims
registry, and the paper validator. No network access and no credentials are involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from memcheck.cli import app
from memcheck.config import load_config
from memcheck.evaluation import ExperimentRunner, discover_runs, verify_integrity
from memcheck.evaluation.runner import ARTIFACT_FILES
from memcheck.workloads import build_workload

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def small_config():
    config = load_config("configs/smoke.yaml")
    config.seeds = [1]
    config.faults = ["none"]
    config.ablations = ["abl_no_supersession"]
    config.checkpoints = [0.5, 1.0]
    config.workload.params["sessions"] = 3
    config.workload.params["tenants"] = 1
    return config


@pytest.fixture(scope="module")
def completed_run(small_config, tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("results")
    runner = ExperimentRunner(small_config, run_id="itest", results_root=root)
    status = runner.run()
    assert status.state == "completed", status.error
    assert status.conditions_failed == 0
    return runner.paths.root


class TestRunArtifacts:
    def test_every_required_artifact_exists(self, completed_run) -> None:
        for name in (*ARTIFACT_FILES, "manifest.json", "config.yaml", "run_status.json"):
            assert (completed_run / name).exists(), name
        assert (completed_run / "stdout.log").exists()
        assert (completed_run / "stderr.log").exists()

    def test_stderr_is_empty_on_a_clean_run(self, completed_run) -> None:
        assert (completed_run / "stderr.log").read_text() == ""

    def test_manifest_records_the_required_provenance(self, completed_run) -> None:
        manifest = json.loads((completed_run / "manifest.json").read_text())
        for field in (
            "run_id",
            "timestamp",
            "operating_system",
            "python_version",
            "dependency_lock_hash",
            "cpu_information",
            "model_provider",
            "model_identifier",
            "temperature",
            "random_seeds",
            "prompt_hashes",
            "benchmark_name",
            "benchmark_version",
            "configuration_hash",
            "token_budget",
            "evidence_budget",
            "checkpoint_schedule",
            "retrieval_mode",
        ):
            assert field in manifest, field
        assert manifest["prompt_hashes"]
        assert manifest["retrieval_mode"].endswith("bm25_py_v1")

    def test_no_secret_appears_in_any_artifact(self, completed_run) -> None:
        for path in completed_run.rglob("*"):
            if not path.is_file():
                continue
            text = path.read_text(errors="ignore")
            assert "ANTHROPIC_API_KEY" not in text
            assert "sk-ant" not in text

    def test_results_are_checksummed_and_verify(self, completed_run) -> None:
        ok, problems = verify_integrity(completed_run)
        assert ok, problems

    def test_modification_is_detected(self, completed_run, tmp_path) -> None:
        import shutil

        copy = tmp_path / "copy"
        shutil.copytree(completed_run, copy)
        target = copy / "grades.jsonl"
        target.write_text(target.read_text() + '{"tampered": true}\n')
        ok, problems = verify_integrity(copy)
        assert ok is False
        assert any("content changed" in problem for problem in problems)

    def test_grades_cover_every_checkpoint(self, completed_run) -> None:
        grades = [
            json.loads(line) for line in (completed_run / "grades.jsonl").read_text().splitlines()
        ]
        assert {grade["checkpoint"] for grade in grades} == {"cp050", "cp100"}

    def test_contracts_ran_for_every_condition(self, completed_run) -> None:
        contracts = [
            json.loads(line)
            for line in (completed_run / "contracts.jsonl").read_text().splitlines()
        ]
        by_condition: dict[str, set[str]] = {}
        for record in contracts:
            by_condition.setdefault(record["condition_id"], set()).add(record["contract_name"])
        assert by_condition
        assert all(len(names) == 6 for names in by_condition.values())


class TestResume:
    def test_resume_skips_completed_conditions(self, small_config, tmp_path) -> None:
        runner = ExperimentRunner(small_config, run_id="resumable", results_root=tmp_path)
        runner.run()
        before = (tmp_path / "resumable" / "grades.jsonl").read_text().count("\n")

        messages: list[str] = []
        again = ExperimentRunner(
            small_config,
            run_id="resumable",
            results_root=tmp_path,
            on_progress=messages.append,
        )
        status = again.run(resume=True)
        after = (tmp_path / "resumable" / "grades.jsonl").read_text().count("\n")
        assert after == before, "a resumed run must not recompute completed conditions"
        assert any("skip (already complete)" in message for message in messages)
        assert status.state == "completed"


class TestWorkloads:
    def test_synthetic_is_deterministic(self) -> None:
        from memcheck.workloads import SyntheticLongitudinalWorkload

        first = SyntheticLongitudinalWorkload(seed=7, sessions=4)
        second = SyntheticLongitudinalWorkload(seed=7, sessions=4)
        assert [event.model_dump(mode="json") for event in first.events()] == [
            event.model_dump(mode="json") for event in second.events()
        ]
        assert [query.query_id for query in first.queries()] == [
            query.query_id for query in second.queries()
        ]

    def test_different_seeds_produce_different_streams(self) -> None:
        from memcheck.workloads import SyntheticLongitudinalWorkload

        first = SyntheticLongitudinalWorkload(seed=1, sessions=6)
        second = SyntheticLongitudinalWorkload(seed=2, sessions=6)
        assert [e.content for e in first.events()] != [e.content for e in second.events()]

    def test_repo_evolution_has_eight_scenarios_and_tasks(self) -> None:
        from memcheck.workloads import RepoEvolutionWorkload

        workload = RepoEvolutionWorkload()
        workload.load()
        assert len(workload.coding_tasks()) == 8
        for task in workload.coding_tasks():
            assert task.repository
            assert task.edit_templates
            assert task.gold_value

    def test_longmemeval_reports_unavailability_clearly(self) -> None:
        from memcheck.workloads import LongMemEvalUnavailable, LongMemEvalWorkload

        workload = LongMemEvalWorkload(data_path="data/raw")
        if workload.is_available():
            pytest.skip("LongMemEval is present in this environment")
        with pytest.raises(LongMemEvalUnavailable, match=r"data/README\.md"):
            workload.load()

    def test_checkpoints_are_monotonic(self, small_config) -> None:
        workload = build_workload(small_config)
        workload.load()
        checkpoints = workload.checkpoints([0.1, 0.5, 1.0])
        indices = [checkpoint.event_index for checkpoint in checkpoints]
        assert indices == sorted(indices)
        assert indices[-1] == len(workload.events())


class TestReporting:
    def test_report_generates_every_artifact(self, completed_run, tmp_path) -> None:
        from memcheck.reporting import build_report

        paper_dir = tmp_path / "paper"
        (paper_dir / "generated").mkdir(parents=True)
        result = build_report(
            completed_run.parent,
            paper_dir,
            run_dirs=[completed_run],
            processed_dir=tmp_path / "processed",
            figures_dir=tmp_path / "figures",
            tables_dir=tmp_path / "tables",
            audit_dir=tmp_path / "audit",
        )
        assert not result.integrity_problems
        assert result.processed_files
        assert len(result.table_results) == 6
        assert len(result.figure_results) == 8
        assert (tmp_path / "processed" / "task_utility.csv").exists()
        assert (tmp_path / "audit" / "audit_sample.csv").exists()

    def test_workbook_carries_the_data_behind_each_figure(self, completed_run, tmp_path) -> None:
        """A rasterised figure loses its numbers; the workbook is how they survive.

        The accuracy sheet is checked against the metric frame rather than merely for existence,
        because a spreadsheet that disagrees with the figure beside it is worse than none.
        """
        import openpyxl

        from memcheck.reporting import build_report

        paper_dir = tmp_path / "paper"
        (paper_dir / "generated").mkdir(parents=True)
        result = build_report(
            completed_run.parent,
            paper_dir,
            run_dirs=[completed_run],
            processed_dir=tmp_path / "processed",
            figures_dir=tmp_path / "figures",
            tables_dir=tmp_path / "tables",
            audit_dir=tmp_path / "audit",
        )
        assert result.workbook_result is not None
        path = result.workbook_result.path
        assert path is not None and path.exists()

        book = openpyxl.load_workbook(path)
        # The README carries the provenance stamp, so a workbook that leaves this repository
        # still names the runs behind it.
        assert book.sheetnames[0] == "README"
        readme = {row[0]: row[1] for row in book["README"].iter_rows(values_only=True)}
        assert readme["source run ids"] == completed_run.name

        assert "fig1_quality_vs_growth" in book.sheetnames
        sheet = book["fig1_quality_vs_growth"]
        header = [cell.value for cell in next(sheet.iter_rows(max_row=1))]
        assert header[:2] == ["checkpoint", "store_growth_pct"]
        assert "guarded_memory" in header

        # Every plotted value must equal the metric frame it was derived from.
        utility = pd.read_csv(tmp_path / "processed" / "task_utility.csv")
        unfaulted = utility[utility["fault"] == "none"]
        column = header.index("guarded_memory")
        for row in sheet.iter_rows(min_row=2, values_only=True):
            expected = unfaulted[
                (unfaulted["variant"] == "guarded_memory") & (unfaulted["checkpoint"] == row[0])
            ]["accuracy"]
            if not expected.empty:
                assert row[column] == pytest.approx(float(expected.iloc[0]))

        # Every group's workbook is copied beside the figures the paper includes.
        copies = list((paper_dir / "generated" / "figures").glob("figure_data_*.xlsx"))
        assert copies

    def test_audit_sample_is_deterministic(self, completed_run, tmp_path) -> None:
        from memcheck.reporting import aggregate
        from memcheck.reporting import audit as audit_module

        frames = aggregate.load_all(completed_run.parent, run_dirs=[completed_run])
        first = audit_module.build_sample(frames, sample_size=12)
        second = audit_module.build_sample(frames, sample_size=12)
        assert list(first["audit_id"]) == list(second["audit_id"])

    def test_audit_agreement_reports_missing_labels(self, completed_run, tmp_path) -> None:
        from memcheck.reporting import aggregate
        from memcheck.reporting import audit as audit_module

        frames = aggregate.load_all(completed_run.parent, run_dirs=[completed_run])
        export = audit_module.export(frames, tmp_path / "audit", sample_size=8)
        result = audit_module.agreement(export.path)
        assert result["labelled"] == 0
        assert "human_label" in str(result["note"])

    def test_pending_claims_do_not_render_a_number(self) -> None:
        """A claim that is not yet supported must never print a value.

        Constructed here rather than read from the shipped registry: which claims happen to be
        pending changes as experiments are run, and a test should not depend on that.
        """
        from memcheck.reporting import claims

        for status in ("pending", "planned", "unsupported"):
            claim = claims.Claim(
                claim_id="X9",
                claim_text="not yet established",
                status=status,  # type: ignore[arg-type]
                research_question="RQ1",
                metric="accuracy",
                run_ids=["some_run"],
                source_file="task_utility.csv",
                value_column="accuracy",
            )
            rendered, printable, _ = claims.resolve_value(claim, {})
            assert printable is False
            assert "RESULT PENDING" in rendered

    def test_the_shipped_registry_is_internally_consistent(self) -> None:
        """Whatever its statuses, every claim must carry what its status requires."""
        from memcheck.reporting import claims

        registry = claims.load_registry("paper/claims_registry.yaml")
        assert registry.claims
        for claim in registry.claims:
            if claim.status in claims.PRINTABLE_STATUS:
                assert claim.run_ids, f"{claim.claim_id} is {claim.status} but cites no run"
                assert claim.source_file, f"{claim.claim_id} is {claim.status} but names no file"
                assert claim.analysis_script, f"{claim.claim_id} names no analysis script"

    def test_registry_flags_a_supported_claim_with_no_artifact(self, tmp_path) -> None:
        from memcheck.reporting import claims

        registry = claims.ClaimsRegistry(
            claims=[
                claims.Claim(
                    claim_id="X1",
                    claim_text="something",
                    status="supported",
                    research_question="RQ1",
                    metric="accuracy",
                    run_ids=["nonexistent_run"],
                    source_file="not_there.csv",
                    value_column="accuracy",
                )
            ]
        )
        # The claim cites a run that is not present, so it has no processed directory to
        # resolve against — the registry must say so rather than reaching into another run's.
        problems = claims.validate_registry(registry, {}, set())
        assert any("does not exist" in problem for problem in problems)
        assert any("not present under results/raw" in problem for problem in problems)
        assert any("analysis_script" in problem for problem in problems)

        rendered, printable, _ = claims.resolve_value(registry.claims[0], {})
        assert printable is False
        assert "cites no run" in rendered


PAPER_SOURCES_SHIPPED = Path("paper/main.tex").exists()
NO_PAPER_SOURCES = "the paper's LaTeX sources are not shipped with this artifact"


class TestPaperValidation:
    @pytest.mark.skipif(not PAPER_SOURCES_SHIPPED, reason=NO_PAPER_SOURCES)
    def test_shipped_paper_validates(self) -> None:
        from memcheck.paper import validate_sources

        report = validate_sources(Path("paper"))
        assert report.ok, report.errors

    def test_an_undefined_citation_is_an_error(self, tmp_path) -> None:
        from memcheck.paper import validate_sources

        (tmp_path / "main.tex").write_text(
            r"\documentclass{article}\begin{document}See \cite{ghost}.\end{document}"
        )
        (tmp_path / "references.bib").write_text("")
        report = validate_sources(tmp_path)
        assert not report.ok
        assert any("ghost" in error for error in report.errors)

    def test_unbalanced_environments_are_detected(self, tmp_path) -> None:
        from memcheck.paper import validate_sources

        (tmp_path / "main.tex").write_text(r"\begin{document}\begin{table}\end{document}")
        (tmp_path / "references.bib").write_text("")
        report = validate_sources(tmp_path)
        assert not report.ok


class TestCli:
    runner = CliRunner()

    def test_doctor_succeeds(self) -> None:
        result = self.runner.invoke(app, ["doctor"])
        assert result.exit_code == 0

    def test_demo_succeeds(self) -> None:
        result = self.runner.invoke(app, ["demo"])
        assert result.exit_code == 0
        assert "rebuild indexes" in result.stdout

    def test_validate_exits_nonzero_on_a_violating_backend(self) -> None:
        result = self.runner.invoke(app, ["validate", "--backend", "transcript_rag"])
        assert result.exit_code != 0

    def test_validate_exits_zero_on_the_guarded_backend(self) -> None:
        result = self.runner.invoke(app, ["validate", "--backend", "guarded_memory"])
        assert result.exit_code == 0

    def test_dry_run_prints_a_cost_estimate(self) -> None:
        result = self.runner.invoke(app, ["run", "--config", "configs/smoke.yaml", "--dry-run"])
        assert result.exit_code == 0
        assert "Cost estimate" in result.stdout

    def test_unknown_config_exits_nonzero(self) -> None:
        result = self.runner.invoke(app, ["run", "--config", "configs/nope.yaml"])
        assert result.exit_code != 0

    @pytest.mark.skipif(not PAPER_SOURCES_SHIPPED, reason=NO_PAPER_SOURCES)
    def test_paper_validate_succeeds(self) -> None:
        result = self.runner.invoke(app, ["paper", "--validate"])
        assert result.exit_code == 0

    def test_inject_runs_all_faults(self) -> None:
        result = self.runner.invoke(
            app, ["inject", "--fault", "all", "--backend", "guarded_memory"]
        )
        assert result.exit_code == 0


def test_discover_runs_finds_the_completed_run(completed_run) -> None:
    assert completed_run in discover_runs(completed_run.parent)


class TestSeedsAreSamples:
    """Regression: every seed once replayed the first seed's stream, so seeds were replicas."""

    def test_each_seed_of_a_seed_sensitive_workload_gets_its_own_stream(
        self, small_config, tmp_path
    ) -> None:
        config = small_config.model_copy(deep=True)
        config.seeds = [1, 2]
        config.ablations = []
        config.backends = ["guarded_memory"]
        config.workload.params["sessions"] = 6
        runner = ExperimentRunner(config, run_id="seeds", results_root=tmp_path)
        assert runner.run().state == "completed"
        grades = pd.read_json(runner.paths.root / "grades.jsonl", lines=True)
        gold = {
            seed: sorted(map(str, frame["gold_answer"])) for seed, frame in grades.groupby("seed")
        }
        assert set(gold) == {1, 2}
        assert gold[1] != gold[2]
        events = pd.read_json(runner.paths.root / "events.jsonl", lines=True)
        assert set(events["seed"]) == {1, 2}
