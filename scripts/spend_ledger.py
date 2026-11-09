"""Billed provider spend of every paid run, failed attempts included (every reported number
must trace to a file under results/).

Billed means uncached calls only: a cached response repeats the original call's cost estimate.
Re-graded runs copy their source's costs.jsonl, so only their judge_calls.jsonl is counted.
Writes results/processed/spend/ledger.csv and prints the total.

    uv run python scripts/spend_ledger.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOTS = (Path("results/raw"), Path("results/raw_partial"))
OUTPUT = Path("results/processed/spend/ledger.csv")


def billed(path: Path) -> tuple[float, int]:
    total, calls = 0.0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("cached"):
            continue
        total += float(record.get("estimated_cost_usd") or 0.0)
        calls += 1
    return total, calls


def main() -> None:
    rows: list[dict[str, object]] = []
    for root in ROOTS:
        for run in sorted(path for path in root.iterdir() if path.is_dir()):
            files = ["judge_calls.jsonl"] if run.name.endswith("_rejudged_lme") else ["costs.jsonl"]
            for name in files:
                path = run / name
                if not path.exists():
                    continue
                usd, calls = billed(path)
                if usd > 0:
                    rows.append(
                        {
                            "root": root.name,
                            "run_id": run.name,
                            "file": name,
                            "billed_calls": calls,
                            "billed_usd": round(usd, 6),
                        }
                    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    total = sum(float(row["billed_usd"]) for row in rows)
    print(f"{len(rows)} paid run file(s); total billed ${total:.2f}")


if __name__ == "__main__":
    main()
