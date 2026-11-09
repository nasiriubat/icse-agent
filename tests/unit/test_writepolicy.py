"""Tests for the write-policy comparison.

The module reaches across two comparability groups, which the rest of the reporting layer
refuses to do. These tests pin down when that is legitimate: the two runs must differ in the
extractor and nothing else, the same query is paired with itself across them, and anything that
does not fit is refused with a reason rather than silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from memcheck.models import ExperimentRunManifest
from memcheck.reporting import writepolicy
from memcheck.reporting.metrics import seeds_are_replicas
from memcheck.reporting.provenance import Provenance
from memcheck.utilities.hashing import hash_directory

RULE = "rule_copula_v1"
LLM = "llm:gpt-5-mini:extract_v1"
CATEGORIES = {
    "q1": "knowledge_update",
    "q2": "knowledge_update",
    "q3": "cross_session_reasoning",
    "q4": "abstention",
}


def _jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def fake_run(
    root: Path,
    run_id: str,
    *,
    extractor: str,
    outcomes: dict[str, dict[str, bool]],
    model: str = "gpt-5-mini",
    seeds: tuple[int, ...] = (1,),
    query_count: int | None = None,
    parameters_hash: str = "7d3a0e59",
    tokens: float = 100.0,
    cost: float = 0.01,
    write_cost: float = 0.0,
    checksum: bool = True,
) -> Path:
    """A minimal, checksummed run directory with the streams the analysis reads."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    queries = sorted({query for by_query in outcomes.values() for query in by_query})
    manifest = ExperimentRunManifest(
        run_id=run_id,
        benchmark_name="longmemeval",
        benchmark_version="v1",
        benchmark_parameters_hash=parameters_hash,
        model_provider="openai",
        model_identifier=model,
        extractor_identity=extractor,
        random_seeds=list(seeds),
        prompt_hashes={"answer_v1": "aaa", "judge_lme_official_v1": "jjj"}
        | ({"extract_v1": "eee"} if extractor.startswith("llm") else {}),
        checkpoint_schedule=["cp100"],
        faults=["none"],
        configuration_hash=f"cfg_{run_id}",
        notes=json.dumps({"query_count": query_count or len(queries)}).join(
            ("platform=test; workload=", "")
        ),
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), default=str), encoding="utf-8"
    )
    contracts, grades, answers, latency, costs = [], [], [], [], []
    for variant, by_query in outcomes.items():
        for seed in seeds:
            condition = f"cond_{variant}_{seed}"
            contracts.append(
                {
                    "condition_id": condition,
                    "backend_name": variant,
                    "ablations": [],
                    "fault": "none",
                    "seed": seed,
                    "contract_name": "restart_equivalence",
                    "passed": True,
                    "skipped": False,
                }
            )
            for query, correct in by_query.items():
                grades.append(
                    {
                        "condition_id": condition,
                        "checkpoint": "cp100",
                        "query_id": query,
                        "seed": seed,
                        "correct": correct,
                        "label": "correct" if correct else "abstained_wrongly",
                        "metadata": {"category": CATEGORIES.get(query, "misc")},
                        "retrieved_memory_ids": ["m1"],
                    }
                )
                answers.append(
                    {
                        "condition_id": condition,
                        "checkpoint": "cp100",
                        "query_id": query,
                        "injected_tokens": tokens,
                    }
                )
                latency.append(
                    {
                        "condition_id": condition,
                        "checkpoint": "cp100",
                        "phase": "retrieval",
                        "latency_ms": 2.0,
                    }
                )
                costs.append(
                    {
                        "condition_id": condition,
                        "checkpoint": "cp100",
                        "estimated_cost_usd": cost,
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "request_identifier": f"req_{condition}_{query}",
                    }
                )
            if write_cost:
                # An LLM extractor pays at ingestion time, under a checkpoint label that is
                # not part of the query schedule; the analysis must neither lose nor mislabel it.
                costs.append(
                    {
                        "condition_id": condition,
                        "checkpoint": "ingestion",
                        "estimated_cost_usd": write_cost,
                        "input_tokens": 500,
                        "output_tokens": 50,
                        "request_identifier": f"req_{condition}_extract",
                    }
                )
    _jsonl(run_dir / "contracts.jsonl", contracts)
    _jsonl(run_dir / "grades.jsonl", grades)
    _jsonl(run_dir / "answers.jsonl", answers)
    _jsonl(run_dir / "latency.jsonl", latency)
    _jsonl(run_dir / "costs.jsonl", costs)
    if checksum:
        (run_dir / "checksums.json").write_text(
            json.dumps({"files": hash_directory(run_dir, exclude=("checksums.json",))}),
            encoding="utf-8",
        )
    return run_dir


RULE_OUTCOMES = {
    "guarded_memory": {"q1": False, "q2": False, "q3": False, "q4": True},
    "no_memory": {"q1": False, "q2": False, "q3": False, "q4": True},
    "transcript_rag": {"q1": True, "q2": True, "q3": False, "q4": True},
}
LLM_OUTCOMES = {
    "guarded_memory": {"q1": True, "q2": True, "q3": True, "q4": False},
    "no_memory": {"q1": False, "q2": False, "q3": False, "q4": True},
    "transcript_rag": {"q1": True, "q2": True, "q3": True, "q4": True},
}


@pytest.fixture
def paired_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    fake_run(root, "lme_rule", extractor=RULE, outcomes=RULE_OUTCOMES, tokens=300.0, cost=0.01)
    fake_run(
        root,
        "lme_llm",
        extractor=LLM,
        outcomes=LLM_OUTCOMES,
        tokens=200.0,
        cost=0.015,
        write_cost=0.5,
    )
    return root


class TestPairSignature:
    def test_the_extractor_and_its_prompt_are_the_only_permitted_difference(self) -> None:
        common = {
            "benchmark_name": "longmemeval",
            "benchmark_parameters_hash": "abc",
            "model_identifier": "gpt-5-mini",
            "random_seeds": [1, 2, 3],
        }
        rule = ExperimentRunManifest(
            run_id="a", **common, extractor_identity=RULE, prompt_hashes={"answer_v1": "x"}
        )
        llm = ExperimentRunManifest(
            run_id="b",
            **common,
            extractor_identity=LLM,
            prompt_hashes={"answer_v1": "x", "extract_v1": "y"},
        )
        assert writepolicy.pair_signature(rule) == writepolicy.pair_signature(llm)

    def test_a_different_model_seed_set_or_answer_prompt_breaks_the_pair(self) -> None:
        base = ExperimentRunManifest(
            run_id="a", benchmark_name="longmemeval", random_seeds=[1], extractor_identity=RULE
        )
        for change in (
            {"model_identifier": "gpt-5"},
            {"random_seeds": [1, 2]},
            {"prompt_hashes": {"answer_v1": "changed"}},
            {"token_budget": 1024},
            {"faults": ["none", "index_loss"]},
        ):
            other = base.model_copy(update=change)
            assert writepolicy.pair_signature(base) != writepolicy.pair_signature(other), change

    def test_every_grouping_and_comparability_field_but_the_extractor_is_paired_on(self) -> None:
        from memcheck.evaluation.fairness import COMPARABLE_FIELDS, GROUPING_FIELDS

        expected = (set(GROUPING_FIELDS) | set(COMPARABLE_FIELDS)) - {"extractor_identity"}
        assert expected <= set(writepolicy.PAIRING_FIELDS)
        assert "extractor_identity" not in writepolicy.PAIRING_FIELDS


class TestAnalyse:
    def test_two_runs_differing_only_in_the_extractor_are_paired(self, paired_root: Path) -> None:
        analysis = writepolicy.analyse(paired_root)
        assert analysis.paired
        assert analysis.run_ids == ["lme_rule", "lme_llm"]
        assert analysis.extractors == {"rule": RULE, "llm": LLM}
        assert analysis.problems == []

        guarded = analysis.conditions.set_index("variant").loc["guarded_memory"]
        assert guarded["n"] == 4
        assert guarded["accuracy_rule"] == pytest.approx(0.25)
        assert guarded["accuracy_llm"] == pytest.approx(0.75)
        assert guarded["gained"] == 3 and guarded["lost"] == 1
        assert guarded["discordant_pairs"] == 4
        assert 0.0 < guarded["p_value"] <= guarded["p_value_holm"] <= 1.0
        assert guarded["accuracy_rule_ci_low"] < 0.25 < guarded["accuracy_rule_ci_high"]
        assert guarded["injected_tokens_rule"] == pytest.approx(300.0)
        assert guarded["injected_tokens_llm"] == pytest.approx(200.0)
        # Cost is summed over the condition: four queries at 0.01 and at 0.015 each.
        assert guarded["estimated_cost_usd_rule"] == pytest.approx(0.04)
        assert guarded["estimated_cost_usd_llm"] == pytest.approx(0.06)
        # The LLM policy's extraction calls are a write-side cost the rule-based policy never
        # pays; they are reported separately and in the total, never folded into answer time.
        assert guarded["write_cost_usd_rule"] == pytest.approx(0.0)
        assert guarded["write_cost_usd_llm"] == pytest.approx(0.5)
        assert guarded["total_cost_usd_llm"] == pytest.approx(0.56)
        assert guarded["latency_p95_ms_retrieval_llm"] == pytest.approx(2.0)

        # A baseline that reproduced exactly has no discordant pairs and p = 1.
        unchanged = analysis.conditions.set_index("variant").loc["no_memory"]
        assert unchanged["discordant_pairs"] == 0 and unchanged["p_value_holm"] == 1.0
        # The guarded condition comes first, because it is the one the manipulation acts on.
        assert next(iter(analysis.conditions["variant"])) == "guarded_memory"
        assert list(analysis.conditions["label"]) == [
            "guarded memory",
            "transcript chunking",
            "no memory",
        ]

    def test_categories_are_counted_from_the_paired_outcomes(self, paired_root: Path) -> None:
        analysis = writepolicy.analyse(paired_root)
        guarded = analysis.categories[analysis.categories["variant"] == "guarded_memory"]
        by_category = guarded.set_index("category")
        assert by_category.loc["knowledge_update", "n"] == 2
        assert by_category.loc["knowledge_update", "correct_rule"] == 0
        assert by_category.loc["knowledge_update", "correct_llm"] == 2
        assert by_category.loc["abstention", "accuracy_rule"] == 1.0
        assert by_category.loc["abstention", "accuracy_llm"] == 0.0
        # Category rows add up to the condition total.
        assert int(guarded["n"].sum()) == 4
        assert list(by_category.index)[:2] == ["knowledge_update", "cross_session_reasoning"]

    def test_saturation_is_reported_per_policy(self, paired_root: Path) -> None:
        analysis = writepolicy.analyse(paired_root)
        assert set(analysis.saturation["policy"]) == {"rule", "llm"}
        rule = analysis.saturation[
            (analysis.saturation["policy"] == "rule")
            & (analysis.saturation["variant"] == "guarded_memory")
        ].iloc[0]
        assert rule["reader_declined_with_evidence"] == 3

    def test_seed_replicas_are_counted_once(self, tmp_path: Path) -> None:
        """D-022: identical outcomes under every seed collapse to the query."""
        root = tmp_path / "raw"
        fake_run(root, "rule", extractor=RULE, outcomes=RULE_OUTCOMES, seeds=(1, 2, 3))
        fake_run(root, "llm", extractor=LLM, outcomes=LLM_OUTCOMES, seeds=(1, 2, 3))
        analysis = writepolicy.analyse(root)
        assert analysis.conditions.set_index("variant").loc["guarded_memory", "n"] == 4
        assert int(analysis.categories["n"].max()) == 2

    def test_seeds_that_disagree_keep_the_seed_in_the_key(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [("q1", 1), ("q1", 2), ("q2", 1), ("q2", 2)], names=["query_id", "seed"]
        )
        agreeing = pd.Series([True, True, False, False], index=index)
        disagreeing = pd.Series([True, False, False, False], index=index)
        assert seeds_are_replicas({"a": agreeing})
        assert not seeds_are_replicas({"a": agreeing, "b": disagreeing})

    def test_a_run_under_another_model_is_excluded_with_a_reason(self, paired_root: Path) -> None:
        """A third run that also changes the model is not a counterpart to anything."""
        fake_run(
            paired_root, "lme_llm_other_model", extractor=LLM, outcomes=LLM_OUTCOMES, model="gpt-5"
        )
        analysis = writepolicy.analyse(paired_root)
        assert analysis.run_ids == ["lme_rule", "lme_llm"]
        assert any("lme_llm_other_model" in problem for problem in analysis.problems)
        assert any("no counterpart" in problem for problem in analysis.problems)

    def test_the_largest_pair_is_kept_and_the_smaller_reported(self, paired_root: Path) -> None:
        """A 20-instance pair and a 100-instance pair are two experiments; the bigger wins."""
        big_rule = {v: {f"q{i}": i % 3 == 0 for i in range(10)} for v in RULE_OUTCOMES}
        big_llm = {v: {f"q{i}": i % 2 == 0 for i in range(10)} for v in RULE_OUTCOMES}
        fake_run(paired_root, "big_rule", extractor=RULE, outcomes=big_rule, parameters_hash="n100")
        fake_run(paired_root, "big_llm", extractor=LLM, outcomes=big_llm, parameters_hash="n100")
        analysis = writepolicy.analyse(paired_root)
        assert analysis.run_ids == ["big_rule", "big_llm"]
        assert any("excluded pair (lme_rule, lme_llm)" in problem for problem in analysis.problems)

    def test_a_run_without_checksums_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "raw"
        fake_run(root, "rule", extractor=RULE, outcomes=RULE_OUTCOMES)
        fake_run(root, "llm", extractor=LLM, outcomes=LLM_OUTCOMES, checksum=False)
        analysis = writepolicy.analyse(root)
        assert not analysis.paired
        assert any("no checksums.json" in problem for problem in analysis.problems)
        assert "no counterpart" in analysis.reason

    def test_an_absent_benchmark_yields_an_unpaired_analysis_not_an_exception(
        self, tmp_path: Path
    ) -> None:
        analysis = writepolicy.analyse(tmp_path, run_dirs=[])
        assert not analysis.paired
        assert "no two runs" in analysis.reason

    def test_a_condition_present_under_one_policy_only_is_reported(self, tmp_path: Path) -> None:
        root = tmp_path / "raw"
        fake_run(root, "rule", extractor=RULE, outcomes=RULE_OUTCOMES)
        llm_outcomes = {k: v for k, v in LLM_OUTCOMES.items() if k != "transcript_rag"}
        fake_run(root, "llm", extractor=LLM, outcomes=llm_outcomes)
        analysis = writepolicy.analyse(root)
        assert "transcript_rag" not in set(analysis.conditions["variant"])
        assert any("'transcript_rag'" in problem for problem in analysis.problems)


def _provenance() -> Provenance:
    return Provenance.create("test", "tests", ["lme_rule", "lme_llm"], ["cfg"])


class TestOutputs:
    def test_the_table_is_wide_labelled_and_free_of_raw_underscores(
        self, paired_root: Path, tmp_path: Path
    ) -> None:
        analysis = writepolicy.analyse(paired_root)
        table = writepolicy.write_latex(analysis, tmp_path / "generated", _provenance())
        assert not table.pending
        text = table.path.read_text(encoding="utf-8")
        assert r"\label{tab:write-policy}" in text
        assert r"\begin{table*}" in text and r"\end{table*}" in text
        assert "guarded memory" in text and "transcript chunking" in text
        assert "guarded_memory" not in text.split(r"\hline")[1]
        assert r"4 (3/1)" in text
        assert r"300$\rightarrow$200" in text
        assert r"0.040$\rightarrow$0.560" in text

    def test_an_unpaired_analysis_writes_a_pending_table_and_skips_the_figure(
        self, tmp_path: Path
    ) -> None:
        analysis = writepolicy.analyse(tmp_path, run_dirs=[])
        table = writepolicy.write_latex(analysis, tmp_path / "generated", _provenance())
        assert table.pending
        assert "[RESULT PENDING:" in table.path.read_text(encoding="utf-8")
        figure = writepolicy.write_figure(analysis, tmp_path / "figures", _provenance())
        assert figure.skipped and figure.path is None
        assert "no two runs" in figure.reason

    def test_the_figure_and_workbook_are_written_with_provenance(
        self, paired_root: Path, tmp_path: Path
    ) -> None:
        import openpyxl

        analysis = writepolicy.analyse(paired_root)
        figure = writepolicy.write_figure(analysis, tmp_path / "figures", _provenance())
        assert not figure.skipped and figure.path is not None
        assert figure.path.exists()
        assert figure.path.with_suffix(".pdf").exists()
        assert figure.path.with_suffix(".png.provenance.json").exists()

        workbook = writepolicy.write_workbook(
            analysis, tmp_path / "figures" / "figure_data_write_policy.xlsx", _provenance()
        )
        assert workbook.path is not None
        book = openpyxl.load_workbook(workbook.path)
        assert book.sheetnames == ["README", "conditions", "categories", "saturation"]
        header = [cell.value for cell in next(book["categories"].iter_rows(max_row=1))]
        assert {"variant", "category", "n", "accuracy_rule", "accuracy_llm"} <= set(header)

    def test_build_places_everything_and_records_it_in_the_index(
        self, paired_root: Path, tmp_path: Path
    ) -> None:
        paper = tmp_path / "paper"
        (paper / "generated").mkdir(parents=True)
        (paper / "generated" / "report_index.json").write_text(
            json.dumps(
                {"figures": [{"name": "fig1", "skipped": False, "reason": ""}], "tables": []}
            )
        )
        outcome = writepolicy.build(
            paired_root,
            paper,
            processed_dir=tmp_path / "processed" / "write_policy",
            figures_dir=tmp_path / "figures" / "write_policy",
        )
        assert (paper / "generated" / "table9_write_policy.tex").exists()
        assert (paper / "generated" / "figures" / "fig_write_policy.png").exists()
        assert (paper / "generated" / "figures" / "fig_write_policy.pdf").exists()
        assert (paper / "generated" / "figures" / "figure_data_write_policy.xlsx").exists()
        processed = pd.read_csv(tmp_path / "processed" / "write_policy" / "conditions.csv")
        assert set(processed["variant"]) == set(RULE_OUTCOMES)
        selection = json.loads(
            (tmp_path / "processed" / "write_policy" / "selection.json").read_text()
        )
        assert selection["paired_run_ids"] == ["lme_rule", "lme_llm"]

        index = json.loads((paper / "generated" / "report_index.json").read_text())
        assert [item["name"] for item in index["figures"]] == ["fig1", "fig_write_policy"]
        assert index["tables"] == [{"name": "table9_write_policy", "pending": False, "reason": ""}]
        assert index["write_policy"]["run_ids"] == ["lme_rule", "lme_llm"]
        assert "write policy pair" in outcome.summary()

    def test_build_without_a_pair_records_the_reason_like_a_skipped_figure(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "raw"
        fake_run(root, "rule_only", extractor=RULE, outcomes=RULE_OUTCOMES)
        paper = tmp_path / "paper"
        outcome = writepolicy.build(
            root,
            paper,
            processed_dir=tmp_path / "processed" / "write_policy",
            figures_dir=tmp_path / "figures" / "write_policy",
        )
        assert outcome.table.pending and outcome.figure.skipped
        index = json.loads((paper / "generated" / "report_index.json").read_text())
        figure = next(item for item in index["figures"] if item["name"] == "fig_write_policy")
        assert figure["skipped"] and "no counterpart" in figure["reason"]
        assert not (paper / "generated" / "figures" / "fig_write_policy.png").exists()
