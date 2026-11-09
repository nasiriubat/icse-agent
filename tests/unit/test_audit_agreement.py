"""Agreement scoring for the human judge audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from memcheck.reporting import audit


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    frame = pd.DataFrame(rows, columns=audit.AUDIT_COLUMNS)
    frame.to_csv(path, index=False)
    return path


def _row(audit_id: str, judge: str, human: str, rationale: object) -> dict[str, object]:
    return {
        "audit_id": audit_id,
        "stratum": "correct",
        "judge_label": judge,
        "judge_rationale": rationale,
        "human_label": human,
        "query": "q",
        "gold_answer": "g",
        "model_answer": "m",
    }


def test_agreement_scores_judge_decided_rows_separately(tmp_path: Path) -> None:
    rows = [
        _row("a", "correct", "correct", '{"label":"correct","rationale":"same"}'),
        _row("b", "incorrect", "correct", '{"label":"incorrect","rationale":"differs"}'),
        _row("c", "incorrect", "incorrect", '{"label":"incorrect","rationale":"differs"}'),
        # Deterministic grades: no rationale, or a grader marker.
        _row("d", "abstained_correctly", "abstained", None),
        _row("e", "unsupported", "correct", "grader=deterministic"),
        _row("f", "abstained_wrongly", "", None),  # unlabelled, ignored
    ]
    result = audit.agreement(_write(tmp_path / "sample.csv", rows))
    assert result["rows"] == 6
    assert result["labelled"] == 5
    assert result["agreements"] == 3
    assert result["judge_labelled"] == 3
    assert result["judge_agreements"] == 2
    assert result["judge_observed_agreement"] == 2 / 3


def test_agreement_without_labels_explains_itself(tmp_path: Path) -> None:
    rows = [_row("a", "correct", "", '{"label":"correct","rationale":"same"}')]
    result = audit.agreement(_write(tmp_path / "sample.csv", rows))
    assert result["labelled"] == 0
    assert "note" in result


def test_agreement_without_judge_rows_reports_zero(tmp_path: Path) -> None:
    rows = [_row("d", "abstained_correctly", "abstained", None)]
    result = audit.agreement(_write(tmp_path / "sample.csv", rows))
    assert result["judge_labelled"] == 0
    assert result["observed_agreement"] == 1.0
