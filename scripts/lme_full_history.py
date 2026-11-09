"""LongMemEval with the full distractor history (S split) beside the oracle split.

Same 100 questions, same reader, prompts, budgets and rule-based extractor; only the split
differs. Both runs are re-graded with the official judge. Seeds are cached replicas, so seed 1
is used. Writes results/processed/write_policy/full_history_split.csv and
results/processed/write_policy/full_history_transcript_vs_guarded.json.

    uv run python scripts/lme_full_history.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from memcheck.utilities.statistics import mcnemar, paired_difference_interval

RUNS = {
    "oracle": Path("results/raw/longmemeval_openai_n100_20261021T100428Z_295868b7_rejudged_lme"),
    "s": Path("results/raw/longmemeval_openai_s_n100_20261104T111853Z_2f3d473e_rejudged_lme"),
}
CONDITIONS = ("no_memory", "full_history", "transcript_rag", "guarded_memory")
CORRECT = {"correct", "abstained_correctly"}
OUT_DIR = Path("results/processed/write_policy")


def _final(record: dict[str, object]) -> bool:
    return (
        record.get("checkpoint") == "cp100"
        and not record.get("ablations")
        and record.get("fault", "none") == "none"
    )


def outcomes(run: Path, backend: str) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for line in (run / "grades.jsonl").read_text(encoding="utf-8").splitlines():
        grade = json.loads(line)
        if _final(grade) and grade["backend_name"] == backend and grade["seed"] == 1:
            result[grade["query_id"]] = grade["label"] in CORRECT
    return result


def mean_injected_tokens(run: Path, backend: str) -> float:
    values = []
    for line in (run / "retrievals.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if _final(record) and record["backend_name"] == backend:
            values.append(float(record["injected_tokens"]))
    return sum(values) / len(values)


def ingest_p95_ms(run: Path, backend: str) -> float:
    values = sorted(
        float(record["latency_ms"])
        for record in map(json.loads, (run / "latency.jsonl").read_text("utf-8").splitlines())
        if record["backend"] == backend and record["phase"] == "ingestion"
    )
    return values[max(0, math.ceil(0.95 * len(values)) - 1)]


def ingested_events(run: Path) -> int:
    return sum(1 for _ in (run / "events.jsonl").open(encoding="utf-8"))


def main() -> None:
    rows = []
    for split, run in RUNS.items():
        events = ingested_events(run)
        for backend in CONDITIONS:
            graded = outcomes(run, backend)
            rows.append(
                {
                    "split": split,
                    "run_id": run.name,
                    "condition": backend,
                    "n": len(graded),
                    "accuracy": sum(graded.values()) / len(graded),
                    "mean_injected_tokens": mean_injected_tokens(run, backend),
                    "ingest_p95_ms": ingest_p95_ms(run, backend),
                    "events": events,
                }
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "full_history_split.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    run = RUNS["s"]
    transcript, guarded = outcomes(run, "transcript_rag"), outcomes(run, "guarded_memory")
    shared = sorted(set(transcript) & set(guarded))
    a = [transcript[q] for q in shared]
    b = [guarded[q] for q in shared]
    test = mcnemar(a, b)
    difference, low, high = paired_difference_interval(a, b)
    payload = {
        "run": run.name,
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
    target = OUT_DIR / "full_history_transcript_vs_guarded.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
