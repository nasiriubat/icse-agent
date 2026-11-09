"""Tests for the cross-scale analysis.

The point of this module is that it reaches across comparability groups, which every other part
of the reporting layer refuses to do. These tests exist to pin down the conditions under which
that is legitimate: size must be the only thing that varies, and a separation that flickers must
not be reported as a detection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from memcheck.reporting import scale
from memcheck.reporting.provenance import Provenance


class TestWorkloadSize:
    def test_the_workload_summary_is_recovered_from_the_manifest_notes(self) -> None:
        notes = 'platform=macOS; workload={"name": "synthetic_longitudinal", "sessions": 40}'
        assert scale.workload_size(notes)["sessions"] == 40

    def test_absent_or_malformed_payloads_yield_nothing_rather_than_raising(self) -> None:
        assert scale.workload_size("platform=macOS") == {}
        assert scale.workload_size("workload={not json}") == {}
        assert scale.workload_size("") == {}


class TestSweepSignature:
    def test_size_is_excluded_so_two_sizes_share_a_signature(self) -> None:
        small = {"sessions": 8, "event_count": 100, "update_fraction": 0.35, "seed": 1}
        large = {"sessions": 64, "event_count": 900, "update_fraction": 0.35, "seed": 3}
        assert scale.sweep_signature(small) == scale.sweep_signature(large)

    def test_a_changed_failure_density_is_a_different_experiment(self) -> None:
        """The whole sweep argument collapses if density moves with size."""
        base = {"sessions": 8, "update_fraction": 0.35}
        assert scale.sweep_signature(base) != scale.sweep_signature(
            {"sessions": 8, "update_fraction": 0.50}
        )

    def test_tenants_are_compared_by_count_not_by_name(self) -> None:
        """Tenant identifiers are generated per run; their number is the parameter."""
        two = {"sessions": 8, "tenants": ["tenant_00", "tenant_01"]}
        two_again = {"sessions": 64, "tenants": ["tenant_aa", "tenant_bb"]}
        three = {"sessions": 8, "tenants": ["t0", "t1", "t2"]}
        assert scale.sweep_signature(two) == scale.sweep_signature(two_again)
        assert scale.sweep_signature(two) != scale.sweep_signature(three)


def _accuracy_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(
    variant: str,
    sessions: int,
    gap: float,
    separated: bool,
    discordant: int = 50,
) -> dict[str, object]:
    return {
        "variant": variant,
        "sessions": sessions,
        "gap_vs_reference": gap,
        "separated_from_reference": separated,
        "discordant_pairs": discordant,
    }


class TestLeadingIndicator:
    def test_a_defect_contracts_caught_early_is_marked_as_led(self) -> None:
        accuracy = _accuracy_frame(
            [
                _row("baseline", 8, 0.0, False),
                _row("baseline", 24, 0.06, True),
                _row("baseline", 64, 0.30, True),
                _row("guarded_memory", 8, 0.0, False),
            ]
        )
        table = scale._leading_indicator(
            accuracy, {8: pd.Series({"baseline": 21})}, "guarded_memory"
        )
        row = table[table["variant"] == "baseline"].iloc[0]
        assert row["violations_at_smallest_store"] == 21
        assert bool(row["separated_at_smallest_store"]) is False
        assert row["first_separating_sessions"] == 24
        assert bool(row["contracts_led_accuracy"]) is True
        assert bool(row["separation_is_stable"]) is True

    def test_a_separation_that_flickers_is_not_called_stable(self) -> None:
        """Significant, then not, then significant again is noise crossing a threshold."""
        accuracy = _accuracy_frame(
            [
                _row("flaky", 8, 0.0, False),
                _row("flaky", 24, 0.014, True, discordant=6),
                _row("flaky", 40, 0.0, False),
                _row("flaky", 64, 0.015, True, discordant=6),
                _row("guarded_memory", 8, 0.0, False),
            ]
        )
        table = scale._leading_indicator(accuracy, {8: pd.Series({"flaky": 6})}, "guarded_memory")
        row = table.iloc[0]
        assert row["first_separating_sessions"] == 24
        assert bool(row["separation_is_stable"]) is False
        assert row["discordant_pairs_at_first_separation"] == 6

    def test_a_condition_accuracy_never_sees_is_reported_as_never(self) -> None:
        accuracy = _accuracy_frame(
            [
                _row("invisible", 8, 0.0, False),
                _row("invisible", 64, 0.0, False),
                _row("guarded_memory", 8, 0.0, False),
            ]
        )
        table = scale._leading_indicator(
            accuracy, {8: pd.Series({"invisible": 42})}, "guarded_memory"
        )
        row = table.iloc[0]
        assert row["first_separating_sessions"] == scale.NEVER
        # Violating throughout while accuracy never notices is the strongest form of the claim.
        assert bool(row["contracts_led_accuracy"]) is True

    def test_a_condition_that_was_always_obviously_worse_did_not_need_contracts(self) -> None:
        accuracy = _accuracy_frame(
            [
                _row("no_memory", 8, 0.71, True),
                _row("no_memory", 64, 0.55, True),
                _row("guarded_memory", 8, 0.0, False),
            ]
        )
        table = scale._leading_indicator(
            accuracy, {8: pd.Series({"no_memory": 18})}, "guarded_memory"
        )
        assert bool(table.iloc[0]["contracts_led_accuracy"]) is False

    def test_the_reference_is_not_compared_against_itself(self) -> None:
        accuracy = _accuracy_frame([_row("guarded_memory", 8, 0.0, False)])
        table = scale._leading_indicator(accuracy, {8: pd.Series(dtype=int)}, "guarded_memory")
        assert table.empty


class TestAnalyseGuards:
    def test_a_missing_benchmark_yields_an_empty_analysis_not_an_exception(
        self, tmp_path: Path
    ) -> None:
        analysis = scale.analyse(tmp_path, run_dirs=[])
        assert analysis.accuracy.empty
        assert analysis.leading_indicator.empty
        assert analysis.violations.empty
        assert analysis.streams.empty


class TestStreamLevel:
    def test_one_difference_per_seed_not_per_query(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [("q1", 1), ("q2", 1), ("q1", 2), ("q2", 2)], names=["query_id", "seed"]
        )
        reference = pd.Series([True, True, True, False], index=index)
        variant = pd.Series([True, False, False, False], index=index)
        assert scale._stream_differences(reference, variant) == pytest.approx([0.5, 0.5])

    def test_without_seeds_there_is_one_stream(self) -> None:
        index = pd.Index(["q1", "q2"], name="query_id")
        reference = pd.Series([True, True], index=index)
        variant = pd.Series([True, False], index=index)
        assert scale._stream_differences(reference, variant) == pytest.approx([0.5])

    def test_a_separation_needs_both_the_pooled_and_the_stream_level_test(self) -> None:
        """Six discordant pairs can pass Holm while only half the streams move: not a detection."""
        rows: list[dict[str, object]] = [
            {"variant": "guarded_memory", "p_value": float("nan"), "sign_flip_p": 1.0},
            {"variant": "pooled_only", "p_value": 0.01, "sign_flip_p": 0.125},
            {"variant": "both", "p_value": 0.001, "sign_flip_p": 0.002},
            {"variant": "neither", "p_value": 0.6, "sign_flip_p": 0.5},
        ]
        scale._apply_verdicts(rows, "guarded_memory")
        verdict = {str(row["variant"]): row for row in rows}
        assert verdict["pooled_only"]["separated_pooled"] is True
        assert verdict["pooled_only"]["separated_from_reference"] is False
        assert verdict["both"]["separated_from_reference"] is True
        assert verdict["neither"]["separated_from_reference"] is False
        assert verdict["guarded_memory"]["separated_from_reference"] is False


class TestViolationsFrame:
    def test_every_graded_condition_appears_at_every_size_with_zero_as_a_value(self) -> None:
        """The reference raised nothing; the figure must still show that as a zero."""
        accuracy = _accuracy_frame(
            [
                _row("guarded_memory", 8, 0.0, False),
                _row("guarded_memory", 64, 0.0, False),
                _row("transcript_rag", 8, 0.0, False),
                _row("transcript_rag", 64, 0.3, True),
            ]
        )
        frame = scale._violations_frame(
            accuracy, {8: pd.Series({"transcript_rag": 21}), 64: pd.Series(dtype=int)}
        )
        assert list(frame.columns) == ["sessions", "variant", "violations"]
        lookup = {(row.variant, row.sessions): row.violations for row in frame.itertuples()}
        assert lookup == {
            ("guarded_memory", 8): 0,
            ("guarded_memory", 64): 0,
            ("transcript_rag", 8): 21,
            ("transcript_rag", 64): 0,
        }

    def test_an_empty_analysis_has_the_schema_but_no_rows(self) -> None:
        frame = scale._violations_frame(pd.DataFrame(), {})
        assert frame.empty
        assert list(frame.columns) == ["sessions", "variant", "violations"]


def _sweep_analysis() -> scale.ScaleAnalysis:
    """A plausible five-size sweep over the four conditions the figure draws, plus one it omits."""
    sizes = [8, 16, 24, 40, 64]
    curves = {
        "guarded_memory": [0.96, 0.88, 0.87, 0.90, 0.85],
        "transcript_rag": [0.97, 0.91, 0.81, 0.70, 0.55],
        "guarded_memory+abl_no_supersession": [0.96, 0.88, 0.83, 0.84, 0.71],
        "guarded_memory+abl_no_tombstones": [0.96, 0.87, 0.86, 0.90, 0.84],
        "no_memory": [0.26, 0.22, 0.22, 0.21, 0.30],
    }
    rows: list[dict[str, object]] = []
    for variant, values in curves.items():
        for size, value in zip(sizes, values, strict=True):
            separated = (
                variant != "guarded_memory"
                and abs(value - curves["guarded_memory"][sizes.index(size)]) > 0.03
            )
            rows.append(
                {
                    "sessions": size,
                    "run_id": f"scale_{size:03d}",
                    "event_count": size * 23,
                    "variant": variant,
                    "n": 400,
                    "accuracy": value,
                    "accuracy_ci_low": value - 0.04,
                    "accuracy_ci_high": min(1.0, value + 0.03),
                    "reference_accuracy": curves["guarded_memory"][sizes.index(size)],
                    "gap_vs_reference": curves["guarded_memory"][sizes.index(size)] - value,
                    "gap_ci_low": curves["guarded_memory"][sizes.index(size)] - value - 0.02,
                    "gap_ci_high": curves["guarded_memory"][sizes.index(size)] - value + 0.02,
                    "discordant_pairs": 30,
                    "p_value": 0.001 if separated else 0.5,
                    "cohens_g": 0.4,
                    "effect_label": "large",
                    "warnings": "",
                    "p_value_holm": float("nan") if variant == "guarded_memory" else 0.5,
                    "separated_from_reference": separated,
                }
            )
    accuracy = pd.DataFrame(rows)
    violations = {
        size: pd.Series(
            {
                "transcript_rag": 21,
                "guarded_memory+abl_no_supersession": 3,
                "guarded_memory+abl_no_tombstones": 6,
                "no_memory": 18,
            }
        )
        for size in sizes
    }
    streams = pd.DataFrame(
        [
            {
                "sessions": size,
                "run_id": f"scale_{size:03d}",
                "variant": variant,
                "n_streams": 10,
                "mean_diff": curves["guarded_memory"][sizes.index(size)] - value,
                "ci_low": curves["guarded_memory"][sizes.index(size)] - value - 0.02,
                "ci_high": curves["guarded_memory"][sizes.index(size)] - value + 0.02,
                "sign_flip_p": 0.002,
                "min_diff": 0.0,
                "max_diff": 0.1,
                "streams_favouring_reference": 10,
            }
            for variant, values in curves.items()
            if variant != "guarded_memory"
            for size, value in zip(sizes, values, strict=True)
        ]
    )
    return scale.ScaleAnalysis(
        accuracy=accuracy,
        leading_indicator=scale._leading_indicator(accuracy, violations, "guarded_memory"),
        run_ids=[f"scale_{size:03d}" for size in sizes],
        violations=scale._violations_frame(accuracy, violations),
        streams=streams,
    )


def _provenance() -> Provenance:
    return Provenance.create("test", "tests/unit/test_scale.py", ["scale_008"], [])


class TestSweepFigure:
    def test_the_figure_is_written_with_pdf_png_and_provenance_sidecar(
        self, tmp_path: Path
    ) -> None:
        path = scale.write_figure(_sweep_analysis(), tmp_path, _provenance())
        assert path is not None
        assert path.name == "fig_scale_sweep.png"
        assert path.with_suffix(".pdf").exists()
        assert path.with_name("fig_scale_sweep.png.provenance.json").exists()

    def test_the_figure_is_single_column_width(self, tmp_path: Path) -> None:
        """An IEEE column is 3.5in; a wider raster shrinks on the page until it is unreadable."""
        from PIL import Image

        path = scale.write_figure(_sweep_analysis(), tmp_path, _provenance())
        assert path is not None
        with Image.open(path) as image:
            width_in, height_in = (
                pixels / image.info.get("dpi", (200, 200))[0] for pixels in image.size
            )
        assert 3.2 <= width_in <= 3.7
        assert height_in <= 4.6

    def test_nothing_is_drawn_without_data_or_without_the_drawn_conditions(
        self, tmp_path: Path
    ) -> None:
        empty = scale.ScaleAnalysis(pd.DataFrame(), pd.DataFrame())
        assert scale.write_figure(empty, tmp_path, _provenance()) is None
        analysis = _sweep_analysis()
        only_baseline = analysis.accuracy[analysis.accuracy["variant"] == "no_memory"]
        absent = scale.ScaleAnalysis(only_baseline, pd.DataFrame(), violations=analysis.violations)
        assert scale.write_figure(absent, tmp_path, _provenance()) is None
        assert not list(tmp_path.iterdir())


class TestSweepWorkbook:
    def test_the_workbook_carries_the_accuracy_and_violation_series(self, tmp_path: Path) -> None:
        import openpyxl

        analysis = _sweep_analysis()
        path = scale.write_workbook(analysis, tmp_path, _provenance())
        assert path is not None and path.name == "figure_data_scale.xlsx"
        book = openpyxl.load_workbook(path)
        assert book.sheetnames == ["README", "accuracy", "violations", "streams"]
        stream_header = [cell.value for cell in next(book["streams"].iter_rows(max_row=1))]
        assert stream_header[:4] == ["sessions", "run_id", "variant", "n_streams"]
        header = [cell.value for cell in next(book["accuracy"].iter_rows(max_row=1))]
        assert header == [
            "sessions",
            "variant",
            "n",
            "accuracy",
            "ci_low",
            "ci_high",
            "discordant_pairs",
            "p_value_holm",
            "separated_from_reference",
        ]
        rows = list(book["accuracy"].iter_rows(min_row=2, values_only=True))
        assert len(rows) == len(analysis.accuracy)
        guarded_64 = next(row for row in rows if row[0] == 64 and row[1] == "guarded_memory")
        assert guarded_64[3] == pytest.approx(0.85)
        violation_rows = list(book["violations"].iter_rows(min_row=2, values_only=True))
        assert (8, "transcript_rag", 21) in violation_rows
        assert (8, "guarded_memory", 0) in violation_rows
        readme = {row[0]: row[1] for row in book["README"].iter_rows(values_only=True)}
        assert readme["source run ids"] == "scale_008"
        assert path.with_name("figure_data_scale.xlsx.provenance.json").exists()


class TestPublishFigure:
    def test_figure_and_workbook_are_copied_beside_the_paper_figures(self, tmp_path: Path) -> None:
        copies = scale.publish_figure(
            _sweep_analysis(), tmp_path / "figures", tmp_path / "paper" / "generated", _provenance()
        )
        names = sorted(path.name for path in copies)
        assert names == ["fig_scale_sweep.pdf", "fig_scale_sweep.png", "figure_data_scale.xlsx"]
        assert all(path.parent == tmp_path / "paper" / "generated" / "figures" for path in copies)


class TestSweepTables:
    def test_table_headings_name_the_actual_sizes_and_render_readable_values(
        self, tmp_path: Path
    ) -> None:
        written = scale.write_latex(_sweep_analysis(), tmp_path, _provenance())
        leading = (tmp_path / "table8_leading_indicator.tex").read_text(encoding="utf-8")
        assert len(written) == 2
        assert "viol." in leading and "same at every store size" in leading
        assert "gap" in leading and "@64" in leading
        assert r"\rlap{$^{*}$}" in (tmp_path / "table7_scale_verdict.tex").read_text()
        assert r"\footnotesize" in leading
        assert r"\begin{table}[t]" in leading
        # Booleans and sizes are words and integers, not 1/0 and 24.0.
        assert " yes " in leading or " yes \\\\" in leading
        assert "24.0" not in leading
        assert "guarded, no tombstones" in leading
        assert "guarded\\_memory+abl" not in leading
        # The verdict is defined once, in the verdict table's note, and the leading-indicator
        # table points at it rather than carrying a second, weaker rule of its own.
        assert "dagger" not in leading
        assert "both tests" in leading and r"\ref{tab:scale-verdict}" in leading
        verdict = (tmp_path / "table7_scale_verdict.tex").read_text(encoding="utf-8")
        assert r"\footnotesize" in verdict
        assert "transcript chunking" in verdict
        assert "sign-flip" in verdict and "10 independent streams" in verdict


@pytest.mark.integration
class TestAnalyseAcrossRealRuns:
    def test_the_sweep_excludes_runs_that_differ_in_more_than_size(self, tmp_path: Path) -> None:
        """Two sizes at one density form a curve; a third at another density must be dropped."""
        from memcheck.config import load_config
        from memcheck.evaluation import ExperimentRunner

        def execute(run_id: str, sessions: int, update_fraction: float) -> None:
            config = load_config("configs/smoke.yaml")
            config.seeds = [1]
            config.faults = ["none"]
            config.ablations = ["abl_no_supersession"]
            config.checkpoints = [1.0]
            config.workload.params["sessions"] = sessions
            config.workload.params["tenants"] = 1
            config.workload.params["update_fraction"] = update_fraction
            ExperimentRunner(config, run_id=run_id, results_root=tmp_path).run()

        execute("sweep_a", 4, 0.35)
        execute("sweep_b", 8, 0.35)
        execute("odd_density", 6, 0.60)

        analysis = scale.analyse(tmp_path)
        assert sorted(analysis.accuracy["sessions"].unique().tolist()) == [4, 8]
        assert any("more than store size" in problem for problem in analysis.problems)
        assert "odd_density" in " ".join(analysis.problems)


class TestPoolingGuard:
    def test_a_duplicated_query_names_the_cause_rather_than_the_symptom(self) -> None:
        """Pandas reports this as "non-unique multi-index", which says nothing useful."""
        import pandas as pd

        from memcheck.reporting.metrics import unique_outcomes

        grades = pd.DataFrame(
            [
                {"query_id": "q1", "seed": 1, "label": "correct"},
                {"query_id": "q1", "seed": 1, "label": "incorrect"},
            ]
        )
        with pytest.raises(ValueError, match="GROUPING_FIELDS"):
            unique_outcomes(grades, ["query_id", "seed"], "guarded_memory")


class TestComparabilityGrouping:
    def test_every_grouping_field_is_also_a_comparability_field(self) -> None:
        """Grouping on something the fairness validator does not check would be arbitrary."""
        from memcheck.evaluation.fairness import COMPARABLE_FIELDS, GROUPING_FIELDS

        # Two exemptions. The benchmark's name and version identify the experiment rather than
        # parameterise it. The condition set (ablations, faults) legitimately varies *within* a
        # run, so the validator cannot require it to be equal across runs -- but two runs with
        # different condition sets are still different experiments and must not be pooled.
        identity_fields = {"benchmark_name", "benchmark_version", "ablations", "faults"}
        assert set(GROUPING_FIELDS) - identity_fields <= set(COMPARABLE_FIELDS)

    def test_the_write_policy_separates_two_otherwise_identical_runs(self) -> None:
        """Regression: the extractor became a comparability field and grouping did not follow,
        which pooled a rule-based and a model-extracted run into one table."""
        from memcheck.models import ExperimentRunManifest
        from memcheck.reporting.report import _group_key

        common = {
            "benchmark_name": "longmemeval",
            "benchmark_version": "v1",
            "benchmark_parameters_hash": "abc123def456",
            "model_provider": "openai",
            "model_identifier": "gpt-5-mini",
        }
        rule = ExperimentRunManifest(run_id="a", **common, extractor_identity="rule_copula_v1")
        llm = ExperimentRunManifest(
            run_id="b", **common, extractor_identity="llm:gpt-5-mini:extract_v1"
        )
        assert _group_key(rule) != _group_key(llm)

    def test_two_samples_of_one_experiment_still_group_together(self) -> None:
        from memcheck.models import ExperimentRunManifest
        from memcheck.reporting.report import _group_key

        common = {
            "benchmark_name": "synthetic_longitudinal",
            "benchmark_version": "v1",
            "benchmark_parameters_hash": "abc123def456",
            "model_provider": "mock",
            "model_identifier": "oracle-reader-v1",
            "extractor_identity": "rule_copula_v1",
        }
        assert _group_key(ExperimentRunManifest(**common, run_id="a")) == _group_key(
            ExperimentRunManifest(**common, run_id="b")
        )

    def test_a_different_condition_set_is_a_different_experiment(self) -> None:
        """Regression: the 40-session sweep point and the full matrix share every workload
        parameter, so their unablated conditions are the same computation done twice."""
        from memcheck.models import ExperimentRunManifest
        from memcheck.reporting.report import _group_key

        common = {
            "benchmark_name": "synthetic_longitudinal",
            "benchmark_parameters_hash": "195f8c8d",
            "model_provider": "mock",
            "model_identifier": "oracle-reader-v1",
        }
        sweep = ExperimentRunManifest(
            run_id="scale_040",
            **common,
            ablations=["abl_no_supersession", "abl_no_tombstones"],
            faults=["none"],
        )
        matrix = ExperimentRunManifest(
            run_id="matrix",
            **common,
            ablations=["abl_no_supersession", "abl_no_tombstones", "abl_no_budget"],
            faults=["none", "index_loss"],
        )
        assert _group_key(sweep) != _group_key(matrix)
