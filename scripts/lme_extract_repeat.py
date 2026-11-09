"""Two independent runs of the model-based extraction policy on LongMemEval.

Same configuration (hash bc11e1d5), separate provider caches, so every model call was repeated.
Reports per-design accuracy in each run and the discordant queries between the runs, beside the
rule-based to model-based swap. Official judge, seed 1 (seeds are cached replicas within a run).
Writes results/processed/write_policy/extract_repeat.json.

    uv run python scripts/lme_extract_repeat.py
"""

from __future__ import annotations

import json
from pathlib import Path

RAW = Path("results/raw")
RULE = RAW / "longmemeval_openai_n100_20261021T100428Z_295868b7_rejudged_lme"
EXTRACT = (
    RAW / "longmemeval_openai_llm_extract_n100_20261022T094454Z_bc11e1d5_rejudged_lme",
    RAW / "longmemeval_openai_llm_extract_n100_20261104T111729Z_bc11e1d5_rejudged_lme",
)
CONDITIONS = ("guarded_memory", "transcript_rag", "full_history", "no_memory")
CORRECT = {"correct", "abstained_correctly"}
OUTPUT = Path("results/processed/write_policy/extract_repeat.json")


def outcomes(run: Path, backend: str) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for line in (run / "grades.jsonl").read_text(encoding="utf-8").splitlines():
        grade = json.loads(line)
        if (
            grade["checkpoint"] == "cp100"
            and not grade.get("ablations")
            and grade.get("fault", "none") == "none"
            and grade["backend_name"] == backend
            and grade["seed"] == 1
        ):
            result[grade["query_id"]] = grade["label"] in CORRECT
    return result


def compare(first: dict[str, bool], second: dict[str, bool]) -> dict[str, object]:
    shared = sorted(set(first) & set(second))
    return {
        "n": len(shared),
        "accuracy_first": sum(first[q] for q in shared) / len(shared),
        "accuracy_second": sum(second[q] for q in shared) / len(shared),
        "gained": sum(second[q] and not first[q] for q in shared),
        "lost": sum(first[q] and not second[q] for q in shared),
    }


def main() -> None:
    payload: dict[str, object] = {
        "rule_run": RULE.name,
        "extract_runs": [run.name for run in EXTRACT],
        "repeat": {
            backend: compare(outcomes(EXTRACT[0], backend), outcomes(EXTRACT[1], backend))
            for backend in CONDITIONS
        },
        "swap_rule_to_extract": {
            run.name: compare(outcomes(RULE, "guarded_memory"), outcomes(run, "guarded_memory"))
            for run in EXTRACT
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
