"""Human-versus-official-judge disagreements in the LongMemEval audit, by memory design.

A sampled answer may have been produced by several designs; such rows are counted under
"shared". Reads the labelled blind sample and its key. Writes
results/processed/audit/lme_disagreements_by_design.csv.

    uv run python scripts/lme_audit_by_design.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

AUDIT = Path("results/audit/lme_official_v1")
OUTPUT = Path("results/processed/audit/lme_disagreements_by_design.csv")
# Mirrors memcheck.reporting.lme_audit: a correct abstention is an accepted answer.
ACCEPT = {"correct", "abstained_correctly"}


def main() -> None:
    with (AUDIT / "audit_blind.csv").open(encoding="utf-8") as handle:
        human = {row["audit_id"]: row["human_label"].strip() for row in csv.DictReader(handle)}
    rows: Counter[str] = Counter()
    disagreements: Counter[str] = Counter()
    with (AUDIT / "audit_key.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = human[row["audit_id"]]
            if label not in {"correct", "incorrect"}:
                continue
            conditions = [c for c in row["conditions"].split(";") if c]
            design = conditions[0] if len(conditions) == 1 else "shared"
            rows[design] += 1
            if (label in ACCEPT) != (row["official_label"] in ACCEPT):
                disagreements[design] += 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["design", "rows", "disagreements"])
        for design in sorted(rows):
            writer.writerow([design, rows[design], disagreements[design]])
    print(OUTPUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
