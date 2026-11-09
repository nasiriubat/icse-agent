"""The official-judge re-grade: verbatim prompts, one call per distinct answer, source untouched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcheck.evaluation import rejudge as rejudge_module
from memcheck.models import ProviderCall


def _row(query_id: str, answer: str, seed: int, label: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "model_answer": answer,
        "seed": seed,
        "label": label,
        "correct": label == "correct",
        "grader": "llm_judge",
        "metadata": {"category": "x"},
    }


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "raw" / "lme_run"
    run.mkdir(parents=True)
    rows = [
        _row("q1", "three months", 1, "incorrect"),
        _row("q1", "three months", 2, "incorrect"),
        _row("q2_abs", "ABSTAIN: no_matching_evidence", 1, "abstained_correctly"),
        _row("q3", "ABSTAIN: weak", 1, "abstained_wrongly"),
    ]
    (run / "grades.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run / "answers.jsonl").write_text("{}\n")
    (run / "manifest.json").write_text(
        json.dumps({"run_id": "lme_run", "prompt_hashes": {"judge_v1": "h"}})
    )
    references = tmp_path / "refs.json"
    references.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "knowledge-update",
                    "question": "How long?",
                    "answer": "3 months",
                },
                {
                    "question_id": "q2_abs",
                    "question_type": "multi-session",
                    "question": "Which?",
                    "answer": "not mentioned",
                },
                {
                    "question_id": "q3",
                    "question_type": "temporal-reasoning",
                    "question": "When?",
                    "answer": "May",
                },
            ]
        )
    )
    return run, references


def test_rejudge_writes_a_new_run_and_leaves_the_source_alone(source, tmp_path) -> None:
    run, references = source
    before = (run / "grades.jsonl").read_text()
    prompts: list[str] = []

    def judge(prompt: str) -> tuple[str, ProviderCall]:
        prompts.append(prompt)
        verdict = "Yes" if "How long?" in prompt or "unanswerable" in prompt else "No"
        return verdict, ProviderCall(
            provider="fake",
            model_identifier="m",
            request_identifier=str(len(prompts)),
            estimated_cost_usd=0.001,
        )

    result = rejudge_module.rejudge(run, references, tmp_path / "raw", judge)
    assert (run / "grades.jsonl").read_text() == before
    assert result.judged_pairs == 3  # the seed-2 replica of q1 is not judged again
    grades = [
        json.loads(line) for line in (result.run_dir / "grades.jsonl").read_text().splitlines()
    ]
    labels = {(g["query_id"], g["seed"]): g["label"] for g in grades}
    assert labels[("q1", 1)] == labels[("q1", 2)] == "correct"
    assert labels[("q2_abs", 1)] == "abstained_correctly"
    assert labels[("q3", 1)] == "abstained_wrongly"
    assert any("updated answer" in prompt for prompt in prompts)
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert "judge_v1" not in manifest["prompt_hashes"]
    assert rejudge_module.JUDGE_ID in manifest["prompt_hashes"]
    assert (result.run_dir / "checksums.json").exists()
    with pytest.raises(FileExistsError):
        rejudge_module.rejudge(run, references, tmp_path / "raw", judge)


def test_official_parser_matches_upstream() -> None:
    assert rejudge_module.official_verdict("Yes.")
    assert not rejudge_module.official_verdict("No")
