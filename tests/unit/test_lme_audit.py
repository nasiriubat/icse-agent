"""The blinded LongMemEval audit: replicas shown once, judge labels hidden, kappa scored."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from memcheck.reporting import lme_audit


def _grade(qid: str, answer: str, label: str, source: str, seed: int = 1) -> dict[str, object]:
    return {
        "query_id": qid,
        "model_answer": answer,
        "label": label,
        "seed": seed,
        "backend_name": "guarded_memory",
        "ablations": [],
        "metadata": {"source_label": source},
    }


def test_export_and_agreement(tmp_path: Path) -> None:
    run = tmp_path / "run_rejudged_lme"
    run.mkdir()
    rows = [
        _grade("a", "3 months", "correct", "incorrect"),
        _grade("a", "3 months", "correct", "incorrect", seed=2),
        _grade("b", "blue", "incorrect", "incorrect"),
        _grade("c", "ABSTAIN: weak", "abstained_wrongly", "abstained_wrongly"),
        _grade("d_abs", "ABSTAIN: none", "abstained_correctly", "abstained_correctly"),
    ]
    (run / "grades.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    references = tmp_path / "refs.json"
    references.write_text(
        json.dumps(
            [
                {"question_id": q, "question_type": t, "question": f"{q}?", "answer": "x"}
                for q, t in (
                    ("a", "knowledge-update"),
                    ("b", "multi-session"),
                    ("c", "multi-session"),
                    ("d_abs", "multi-session"),
                )
            ]
        )
    )
    exported = lme_audit.export([run], references, tmp_path / "audit", sample_size=10)
    blind = pd.read_csv(exported.blind_path, dtype=str, keep_default_na=False)
    # The replica of "a" appears once; the abstention on answerable "c" is excluded.
    assert sorted(blind["question"]) == ["a?", "b?", "d_abs?"]
    assert "official_label" not in blind.columns
    assert "judge_v1_label" not in blind.columns
    blind["human_label"] = ["correct"] * len(blind)
    blind.to_csv(exported.blind_path, index=False)
    scored = lme_audit.agreement(exported.blind_path, exported.key_path)
    assert scored["labelled"] == 3
    assert scored["human_vs_official_judge"]["agreements"] == 2
    assert scored["human_vs_judge_v1"]["agreements"] == 1
