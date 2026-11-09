"""Guarded-minus-transcript accuracy gap within each question type, at each sweep size.

The question mix shifts with store size (the attribute pool is fixed), so the pooled gap could
reflect the mix rather than the store. This separates the two. Writes
results/processed/scale/gap_by_category.csv and, standardised to a fixed question mix,
results/processed/scale/gap_standardised.csv.

    uv run python scripts/scale_gap_by_category.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

SIZES = ("008", "016", "024", "040", "064")
CORRECT = {"correct", "abstained_correctly"}
OUTPUT = Path("results/processed/scale/gap_by_category.csv")
STANDARDISED_OUTPUT = Path("results/processed/scale/gap_standardised.csv")


def sweep_run(size: str) -> Path:
    """The corrected-label sweep run at one size (workload v4, DEVIATIONS V-024/V-025)."""
    matches = sorted(Path("results/raw").glob(f"scale_v4_{size}_2*"))
    if len(matches) != 1:
        raise SystemExit(f"expected one scale_v4_{size} run, found {[m.name for m in matches]}")
    return matches[0]


def main() -> None:
    rows: list[dict[str, object]] = []
    for size in SIZES:
        outcomes: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
        grades = sweep_run(size) / "grades.jsonl"
        for line in grades.read_text(encoding="utf-8").splitlines():
            grade = json.loads(line)
            if grade["checkpoint"] != "cp100" or grade["ablations"]:
                continue
            if grade["backend_name"] in ("guarded_memory", "transcript_rag"):
                category = grade["metadata"]["category"]
                outcomes[category][grade["backend_name"]].append(grade["label"] in CORRECT)
        total = sum(len(by["guarded_memory"]) for by in outcomes.values())
        for category, by in sorted(outcomes.items()):
            guarded, transcript = by["guarded_memory"], by["transcript_rag"]
            rows.append(
                {
                    "sessions": int(size),
                    "category": category,
                    "n": len(guarded),
                    "share": round(len(guarded) / total, 4),
                    "guarded_accuracy": round(sum(guarded) / len(guarded), 4),
                    "transcript_accuracy": round(sum(transcript) / len(transcript), 4),
                    "gap": round(
                        sum(guarded) / len(guarded) - sum(transcript) / len(transcript), 4
                    ),
                }
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")
    standardised = _standardise(rows)
    with STANDARDISED_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(standardised[0]))
        writer.writeheader()
        writer.writerows(standardised)
    print(f"wrote {len(standardised)} rows to {STANDARDISED_OUTPUT}")


def _standardise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """The gap under a question mix held fixed across sizes.

    Two weightings: the 8-session mix (the smallest store's shares, applied at every size) and
    equal weights over the question types present. A type absent at a size contributes its
    weight times zero, which is the conservative reading.
    """
    by_size: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_size[int(row["sessions"])][str(row["category"])] = row  # type: ignore[arg-type]
    smallest = min(by_size)
    small_weights = {
        category: float(row["share"])
        for category, row in by_size[smallest].items()  # type: ignore[arg-type]
    }
    output: list[dict[str, object]] = []
    for sessions, by_category in sorted(by_size.items()):
        raw = sum(float(r["share"]) * float(r["gap"]) for r in by_category.values())  # type: ignore[arg-type]
        small = sum(
            weight * float(by_category[c]["gap"]) if c in by_category else 0.0  # type: ignore[arg-type]
            for c, weight in small_weights.items()
        )
        equal = sum(float(r["gap"]) for r in by_category.values()) / len(by_category)  # type: ignore[arg-type]
        output.append(
            {
                "sessions": sessions,
                "gap_raw_mix": round(raw, 4),
                f"gap_{smallest}_session_weights": round(small, 4),
                "gap_equal_weights": round(equal, 4),
                "types_present": len(by_category),
            }
        )
    return output


if __name__ == "__main__":
    main()
