"""Paired comparison of transcript chunking against the guarded store on LongMemEval.

Both conditions come from the same re-graded run (model-based extraction, official judge), so
they answered the same 100 queries. Seeds are replicas and are collapsed. Writes
results/processed/write_policy/transcript_vs_guarded.json.

    uv run python scripts/lme_transcript_vs_guarded.py
"""

from __future__ import annotations

import json
from pathlib import Path

from memcheck.utilities.statistics import mcnemar, paired_difference_interval

RUN = Path("results/raw/longmemeval_openai_llm_extract_n100_20261022T094454Z_bc11e1d5_rejudged_lme")
OUTPUT = Path("results/processed/write_policy/transcript_vs_guarded.json")
CORRECT = {"correct", "abstained_correctly"}


def outcomes(backend: str) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for line in (RUN / "grades.jsonl").read_text(encoding="utf-8").splitlines():
        grade = json.loads(line)
        if grade["backend_name"] == backend and not grade.get("ablations") and grade["seed"] == 1:
            result[grade["query_id"]] = grade["label"] in CORRECT
    return result


def main() -> None:
    transcript, guarded = outcomes("transcript_rag"), outcomes("guarded_memory")
    shared = sorted(set(transcript) & set(guarded))
    a = [transcript[q] for q in shared]
    b = [guarded[q] for q in shared]
    test = mcnemar(a, b)
    difference, low, high = paired_difference_interval(a, b)
    payload = {
        "run": RUN.name,
        "n": len(shared),
        "transcript_accuracy": sum(a) / len(a),
        "guarded_accuracy": sum(b) / len(b),
        "transcript_only_correct": sum(x and not y for x, y in zip(a, b, strict=True)),
        "guarded_only_correct": sum(y and not x for x, y in zip(a, b, strict=True)),
        "difference": difference,
        "difference_ci95_newcombe": [low, high],
        "mcnemar_p": test.p_value,
        "method": test.note,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
