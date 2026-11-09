"""Contract outcomes on fresh stores versus snapshots of the grown 64-session stores.

Compares the corrected-label 64-session sweep run (results/raw/scale_v4_064_*, contract_store:
fresh) with results/raw/scale_v4_064_aged_contracts_* (contract_store: aged, same config
otherwise).
Writes results/processed/contracts/aged_vs_fresh.csv.

    uv run python scripts/aged_contracts_summary.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

RUN_PATTERNS = {"fresh": "scale_v4_064_2*", "aged": "scale_v4_064_aged_contracts_*"}
OUTPUT = Path("results/processed/contracts/aged_vs_fresh.csv")


def outcome(record: dict[str, object]) -> str:
    if record["skipped"]:
        return "skipped"
    if record["vacuous"]:
        return "vacuous"
    return "pass" if record["passed"] else "fail"


def main() -> None:
    counts: dict[tuple[str, str], Counter[str]] = {}
    for state, pattern in RUN_PATTERNS.items():
        matches = sorted(Path("results/raw").glob(pattern))
        if len(matches) != 1:
            raise SystemExit(f"expected one {pattern} run, found {[m.name for m in matches]}")
        path = matches[0] / "contracts.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["fault"] != "none":
                continue
            recorded = record.get("store_state", "fresh")
            if recorded != state:
                raise SystemExit(f"{path}: store_state {recorded!r}, expected {state!r}")
            variant = "+".join([str(record["backend_name"]), *record["ablations"]])
            key = (str(record["contract_name"]), variant)
            counts.setdefault(key, Counter())[f"{state}_{outcome(record)}"] += 1
    columns = [
        "contract",
        "variant",
        "fresh_pass",
        "fresh_fail",
        "fresh_vacuous",
        "aged_pass",
        "aged_fail",
        "aged_vacuous",
        "identical",
    ]
    rows = []
    for (contract, variant), counter in sorted(counts.items()):
        row: dict[str, object] = {"contract": contract, "variant": variant}
        for column in columns[2:-1]:
            row[column] = counter.get(column, 0)
        row["identical"] = all(
            counter.get(f"fresh_{o}", 0) == counter.get(f"aged_{o}", 0)
            for o in ("pass", "fail", "vacuous")
        )
        rows.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    differing = [r for r in rows if not r["identical"]]
    print(f"wrote {OUTPUT} ({len(rows)} rows, {len(differing)} differ)")


if __name__ == "__main__":
    main()
