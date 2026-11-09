"""Padding control: a fixed fact stream and query set, with empty sessions appended (V-023).

Reads the latest ``padding_v5_<style>_<n>`` run for each style and padding level: an 8-session
fact stream per seed followed by n sessions that assert nothing, so only the store's size changes
and the query set is identical at every level. Writes
results/processed/padding/gap_by_padding.csv: the guarded-minus-transcript accuracy gap at the
full store per style and level, with the pooled Newcombe interval and McNemar test (query in
stream) and the stream-level mean, cluster-bootstrap interval and exact sign-flip test (stream).

    uv run python scripts/padding_gap.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from memcheck.utilities.statistics import mcnemar, paired_difference_interval

sys.path.insert(0, str(Path(__file__).resolve().parent))
from longitudinal_checkpoints import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CORRECT,
    cluster_bootstrap,
    latest_run,
    read_jsonl,
    sign_flip_p,
)

STYLES = ("topical", "unrelated")
LEVELS = ("000", "008", "016", "032", "056")
OUTPUT = Path("results/processed/padding/gap_by_padding.csv")


def main() -> None:
    results_root = Path("results/raw")
    rows: list[dict[str, object]] = []
    for style in STYLES:
        for level in LEVELS:
            run_dir = latest_run(results_root, f"padding_v5_{style}_{level}_")
            outcomes: dict[str, dict[tuple[str, int], bool]] = defaultdict(dict)
            stale: dict[str, int] = defaultdict(int)
            events = len(read_jsonl(run_dir / "events.jsonl"))
            for grade in read_jsonl(run_dir / "grades.jsonl"):
                if grade["fault"] != "none" or grade["checkpoint"] != "cp100":
                    continue
                backend = str(grade["backend_name"])
                outcomes[backend][(str(grade["query_id"]), int(grade["seed"]))] = (  # type: ignore[arg-type]
                    grade["label"] in CORRECT
                )
                if grade["label"] == "stale":
                    stale[backend] += 1
            guarded, transcript = outcomes["guarded_memory"], outcomes["transcript_rag"]
            pairs = sorted(set(guarded) & set(transcript))
            a = [guarded[p] for p in pairs]
            b = [transcript[p] for p in pairs]
            diff, low, high = paired_difference_interval(a, b)
            test = mcnemar(a, b)
            per_seed: dict[int, list[tuple[bool, bool]]] = defaultdict(list)
            for pair in pairs:
                per_seed[pair[1]].append((guarded[pair], transcript[pair]))
            stream_diffs = [
                sum(x for x, _ in items) / len(items) - sum(y for _, y in items) / len(items)
                for _, items in sorted(per_seed.items())
            ]
            c_low, c_high = cluster_bootstrap(stream_diffs, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
            rows.append(
                {
                    "style": style,
                    "padding_sessions": int(level),
                    "total_sessions": 8 + int(level),
                    "run_id": run_dir.name,
                    "events_per_seed": round(events / len(per_seed), 1),
                    "n_pairs": len(pairs),
                    "guarded_accuracy": round(sum(a) / len(a), 4),
                    "transcript_accuracy": round(sum(b) / len(b), 4),
                    "transcript_stale": stale["transcript_rag"],
                    "gap": round(diff, 4),
                    "gap_ci_low": round(low, 4),
                    "gap_ci_high": round(high, 4),
                    "discordant_pairs": sum(x != y for x, y in zip(a, b, strict=True)),
                    "p_mcnemar": test.p_value,
                    "n_streams": len(stream_diffs),
                    "stream_mean": round(statistics.mean(stream_diffs), 4),
                    "stream_ci_low": round(c_low, 4),
                    "stream_ci_high": round(c_high, 4),
                    "sign_flip_p": round(sign_flip_p(stream_diffs), 4),
                    "streams_favouring_guarded": sum(d > 0 for d in stream_diffs),
                }
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
