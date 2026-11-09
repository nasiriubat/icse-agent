"""LaTeX table rendering.

Every generated table has to fit an IEEE column, so these tests pin the layout decisions that
decide whether it does: the environment, the font size, how many columns a table carries, and
how condition identifiers and measurements are rendered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from memcheck.reporting import tables
from memcheck.reporting.provenance import Provenance


@pytest.fixture
def provenance() -> Provenance:
    return Provenance.create("test", "tests/unit/test_tables.py", ["run_a"], ["abc"])


class TestDisplayName:
    @pytest.mark.parametrize(
        ("variant", "expected"),
        [
            ("guarded_memory", "guarded memory"),
            ("transcript_rag", "transcript chunking"),
            ("no_memory", "no memory"),
            ("guarded_memory+abl_no_supersession", "guarded, no supersession"),
            ("guarded_memory+abl_no_tenant_filter", "guarded, no tenant filter"),
            ("guarded_memory+abl_no_budget+abl_no_tombstones", "guarded, no budget, no tombstones"),
        ],
    )
    def test_identifiers_become_readable_names(self, variant: str, expected: str) -> None:
        assert tables.display_name(variant) == expected


class TestFormatting:
    def test_booleans_are_words_not_integers(self) -> None:
        assert tables._fmt(True) == "yes"
        assert tables._fmt(np.False_) == "no"

    def test_numpy_integers_are_integers(self) -> None:
        assert tables._fmt(np.int64(1234)) == "1,234"

    def test_special_characters_are_escaped(self) -> None:
        assert tables._fmt("a_b & 5%") == r"a\_b \& 5\%"


class TestTabular:
    def _frame(self, columns: int) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
        frame = pd.DataFrame(
            [
                {
                    "variant": "guarded_memory+abl_no_budget",
                    **{f"c{i}": 0.5 for i in range(columns - 1)},
                }
            ]
        )
        return frame, [("variant", "Condition")] + [(f"c{i}", f"C{i}") for i in range(columns - 1)]

    def test_default_is_a_single_column_small_table(self, provenance: Provenance) -> None:
        frame, columns = self._frame(4)
        tex = tables._tabular(frame, columns, "cap", "tab:x", provenance)
        assert r"\begin{table}[t]" in tex and r"\end{table}" in tex
        assert r"\small" in tex and r"\footnotesize" not in tex

    def test_wide_tables_span_both_columns_at_footnotesize(self, provenance: Provenance) -> None:
        frame, columns = self._frame(4)
        tex = tables._tabular(frame, columns, "cap", "tab:x", provenance, wide=True)
        assert r"\begin{table*}[t]" in tex and r"\end{table*}" in tex
        assert r"\footnotesize" in tex

    def test_more_than_four_columns_drops_to_footnotesize(self, provenance: Provenance) -> None:
        frame, columns = self._frame(7)
        tex = tables._tabular(frame, columns, "cap", "tab:x", provenance)
        assert r"\footnotesize" in tex
        assert r"\begin{table}[t]" in tex

    def test_an_explicit_fontsize_wins(self, provenance: Provenance) -> None:
        frame, columns = self._frame(7)
        tex = tables._tabular(frame, columns, "cap", "tab:x", provenance, fontsize=r"\scriptsize")
        assert r"\scriptsize" in tex and r"\footnotesize" not in tex

    def test_the_variant_column_is_rendered_through_display_name(
        self, provenance: Provenance
    ) -> None:
        frame, columns = self._frame(2)
        tex = tables._tabular(frame, columns, "cap", "tab:x", provenance)
        assert "guarded, no budget" in tex
        assert "abl\\_no" not in tex


def _utility(n: object = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant": variant,
                "fault": "none",
                "checkpoint": "cp100",
                "n": n,
                "accuracy": accuracy,
                "accuracy_ci_low": accuracy - 0.03,
                "accuracy_ci_high": accuracy + 0.03,
            }
            for variant, accuracy in (("guarded_memory", 0.903), ("transcript_rag", 0.701))
        ]
    )


class TestMainResults:
    def test_accuracy_and_interval_share_one_column_and_empty_n_is_dropped(
        self, tmp_path: Path, provenance: Provenance
    ) -> None:
        quality = pd.DataFrame(
            [
                {
                    "variant": "guarded_memory",
                    "fault": "none",
                    "checkpoint": "cp100",
                    "evidence_recall": 0.326,
                    "evidence_precision": 0.129,
                    "abstention_precision": 0.8,
                }
            ]
        )
        result = tables.table_main_results(
            {"task_utility": _utility(n=float("nan")), "memory_quality": quality},
            tmp_path,
            provenance,
        )
        tex = result.path.read_text(encoding="utf-8")
        header = next(line for line in tex.splitlines() if line.startswith("Backend"))
        assert header.count("&") == 4, header
        assert "$n$" not in header
        assert "0.903 [0.873, 0.933]" in tex
        assert "guarded memory & 0.903" in tex

    def test_n_is_kept_when_it_is_known(self, tmp_path: Path, provenance: Provenance) -> None:
        result = tables.table_main_results({"task_utility": _utility(n=432)}, tmp_path, provenance)
        tex = result.path.read_text(encoding="utf-8")
        header = next(line for line in tex.splitlines() if line.startswith("Backend"))
        assert "$n$" in header
        assert "& 432 &" in tex


class TestEfficiency:
    def test_sizes_are_kib_with_one_decimal_and_p50_and_cost_are_gone(
        self, tmp_path: Path, provenance: Provenance
    ) -> None:
        efficiency = pd.DataFrame(
            [
                {
                    "variant": "guarded_memory",
                    "fault": "none",
                    "checkpoint": "cp100",
                    "injected_tokens_mean": 26.08,
                    "latency_p50_ms_retrieval": 0.69,
                    "latency_p95_ms_retrieval": 0.93,
                    "latency_p95_ms_ingestion": 0.35,
                    "estimated_cost_usd": 0.0,
                }
            ]
        )
        storage = pd.DataFrame(
            [
                {
                    "variant": "guarded_memory",
                    "fault": "none",
                    "checkpoint": "cp100",
                    "database_size_bytes": 815104,
                    "index_size_bytes": 23036,
                }
            ]
        )
        result = tables.table_efficiency(
            {"efficiency": efficiency, "storage_growth": storage}, tmp_path, provenance
        )
        tex = result.path.read_text(encoding="utf-8")
        header = next(line for line in tex.splitlines() if line.startswith("Backend"))
        assert header.count("&") == 5
        assert "p50" not in header and "USD" not in header
        assert "DB" in header and "Index" in header and header.count("(KiB)") == 2
        assert "& 796.0 & 22.5" in tex
        assert "815104" not in tex


class TestFaultRecovery:
    def test_fault_names_are_short_and_the_table_is_wide(
        self, tmp_path: Path, provenance: Provenance
    ) -> None:
        faults = pd.DataFrame(
            [
                {"variant": "guarded_memory", "fault_name": fault, "recovered": True}
                for fault in (
                    "duplicate_event_delivery",
                    "out_of_order_delivery",
                    "interrupted_write",
                    "index_loss",
                    "contradictory_update",
                    "deletion_during_growth",
                )
            ]
        )
        result = tables.table_fault_recovery({}, {"faults": faults}, tmp_path, provenance)
        tex = result.path.read_text(encoding="utf-8")
        assert r"\begin{table*}[t]" in tex
        header = next(line for line in tex.splitlines() if line.startswith("Backend"))
        for short in tables.FAULT_SHORT_NAMES.values():
            assert short in header
        assert "duplicate" not in header


class TestContractViolations:
    def test_contract_headings_are_short_names(
        self, tmp_path: Path, provenance: Provenance
    ) -> None:
        contracts = pd.DataFrame(
            [
                {"variant": "guarded_memory", "contract_name": name, "passed": True}
                for name in tables.CONTRACT_SHORT_NAMES
            ]
        )
        violations = pd.DataFrame(
            [{"variant": "guarded_memory", "contract_name": "deletion_closure"}]
        )
        result = tables.table_contract_violations(
            {"contracts": contracts, "violations": violations}, tmp_path, provenance
        )
        tex = result.path.read_text(encoding="utf-8")
        header = next(line for line in tex.splitlines() if line.startswith("Condition"))
        assert header.count("&") == 6
        assert "Deletion" in header and "deletion~closure" not in header
        assert r"\footnotesize" in tex

    def test_a_cell_is_one_run_of_the_suite_not_the_sum_over_seeds(
        self, tmp_path: Path, provenance: Provenance
    ) -> None:
        """Two seeds each raised two deletion violations on the same scenario: the cell says 2,
        and the note says how many runs there were, so a repeat is not read as a discovery."""
        contracts = pd.DataFrame(
            [
                {"variant": "ablated", "contract_name": name, "passed": True, "seed": seed}
                for name in tables.CONTRACT_SHORT_NAMES
                for seed in (1, 2)
            ]
        )
        violations = pd.DataFrame(
            [
                {
                    "variant": "ablated",
                    "contract_name": "deletion_closure",
                    "fault": "none",
                    "seed": seed,
                }
                for seed in (1, 2)
                for _ in range(2)
            ]
        )
        result = tables.table_contract_violations(
            {"contracts": contracts, "violations": violations}, tmp_path, provenance
        )
        tex = result.path.read_text(encoding="utf-8")
        header = next(line for line in tex.splitlines() if line.startswith("Condition"))
        row = next(line for line in tex.splitlines() if line.startswith("ablated"))
        deletion = next(i for i, cell in enumerate(header.split("&")) if "Deletion" in cell)
        assert row.split("&")[deletion].strip().startswith("2")
        assert "4" not in row
        assert "in one run of the suite" in tex and "2 times" in tex
        assert "every run gave the same count" in tex
