"""Grade-label counts per design at the full store, for each sweep size.

Fault none, no ablations, final checkpoint, ten seeds pooled. The stale and leaked columns are
the answer-level counterparts of the supersession and deletion-closure contracts. Writes
results/processed/scale/label_counts.csv.

    uv run python scripts/scale_label_counts.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

SIZES = ("008", "016", "024", "040", "064")
LABELS = (
    "correct",
    "abstained_correctly",
    "incorrect",
    "stale",
    "leaked",
    "unsupported",
    "abstained_wrongly",
)
OUTPUT = Path("results/processed/scale/label_counts.csv")


def sweep_run(size: str) -> Path:
    """The corrected-label sweep run at one size (workload v4, DEVIATIONS V-024/V-025)."""
    matches = sorted(Path("results/raw").glob(f"scale_v4_{size}_2*"))
    if len(matches) != 1:
        raise SystemExit(f"expected one scale_v4_{size} run, found {[m.name for m in matches]}")
    return matches[0]


def main() -> None:
    rows: list[dict[str, object]] = []
    for size in SIZES:
        counts: dict[str, Counter[str]] = {}
        grades = sweep_run(size) / "grades.jsonl"
        for line in grades.read_text(encoding="utf-8").splitlines():
            grade = json.loads(line)
            if grade["checkpoint"] != "cp100" or grade["fault"] != "none" or grade["ablations"]:
                continue
            counts.setdefault(grade["backend_name"], Counter())[grade["label"]] += 1
        for variant, counter in sorted(counts.items()):
            row: dict[str, object] = {"sessions": int(size), "variant": variant}
            row["n"] = sum(counter.values())
            for label in LABELS:
                row[label] = counter.get(label, 0)
            unknown = set(counter) - set(LABELS)
            if unknown:
                raise SystemExit(f"unexpected labels {sorted(unknown)} in {grades}")
            rows.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sessions", "variant", "n", *LABELS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
