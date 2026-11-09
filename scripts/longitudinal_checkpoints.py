"""One growing store, read at fixed checkpoints (DEVIATIONS V-023, challenge R3).

Reads the latest ``longitudinal_v5_064`` run: one 64-session stream per seed, asked at 8, 16,
24, 40 and 64 sessions what was true then. Writes

* results/processed/longitudinal/checkpoints.csv: per checkpoint and condition, accuracy and its
  error labels, record counts, storage, and ingestion latency, pooled over the ten seeds;
* results/processed/longitudinal/gap_vs_guarded.csv: the paired gap of each condition to the
  guarded store at each checkpoint, with the pooled Newcombe interval and McNemar test (unit:
  query in stream) and the stream-level mean with a cluster-bootstrap interval and an exact
  sign-flip test (unit: stream), beside the between-stream gap of the size sweep at the same
  session count (results/processed/scale/scale_accuracy.csv).

    uv run python scripts/longitudinal_checkpoints.py [run_dir]
"""

from __future__ import annotations

import csv
import itertools
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from memcheck.utilities.statistics import mcnemar, paired_difference_interval

RUN_PREFIX = "longitudinal_v5_064_"
OUTPUT_DIR = Path("results/processed/longitudinal")
SWEEP = Path("results/processed/scale/scale_accuracy.csv")
CORRECT = {"correct", "abstained_correctly"}
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261104


def variant_of(backend: str, ablations: list[str]) -> str:
    if backend == "guarded_memory" and ablations:
        return "_".join(a.removeprefix("abl_") for a in ablations)
    return backend


def latest_run(results_root: Path, prefix: str) -> Path:
    runs = sorted(p for p in results_root.iterdir() if p.name.startswith(prefix))
    if not runs:
        raise SystemExit(f"no run under {results_root} starts with {prefix}")
    return runs[-1]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def cluster_bootstrap(diffs: list[float], reps: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        sample = [rng.choice(diffs) for _ in diffs]
        means.append(sum(sample) / len(sample))
    return percentile(means, 0.025), percentile(means, 0.975)


def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided test of mean zero under all 2^k sign flips of the stream differences."""
    observed = abs(sum(diffs))
    count = 0
    total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        if abs(sum(s * d for s, d in zip(signs, diffs, strict=True))) >= observed - 1e-12:
            count += 1
    return count / total


def main() -> None:
    results_root = Path("results/raw")
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run(results_root, RUN_PREFIX)
    grades = read_jsonl(run_dir / "grades.jsonl")
    stats = read_jsonl(run_dir / "store_statistics.jsonl")
    latency = read_jsonl(run_dir / "latency.jsonl")
    sessions_at: dict[str, int] = {}

    # Outcomes keyed by (checkpoint, variant) -> {(query_id, seed): correct}
    outcomes: dict[tuple[str, str], dict[tuple[str, int], bool]] = defaultdict(dict)
    labels: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for grade in grades:
        if grade["fault"] != "none":
            continue
        variant = variant_of(str(grade["backend_name"]), list(grade["ablations"]))  # type: ignore[arg-type]
        key = (str(grade["checkpoint"]), variant)
        outcomes[key][(str(grade["query_id"]), int(grade["seed"]))] = grade["label"] in CORRECT  # type: ignore[arg-type]
        labels[key][str(grade["label"])] += 1

    condition_variant: dict[str, str] = {}
    for grade in grades:
        condition_variant[str(grade["condition_id"])] = variant_of(
            str(grade["backend_name"]),
            list(grade["ablations"]),  # type: ignore[arg-type]
        )
    storage: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in stats:
        if row.get("tenant_id") is not None or row["fault"] != "none":
            continue
        key = (str(row["checkpoint"]), condition_variant[str(row["condition_id"])])
        storage[key].append(row)
        sessions_at[str(row["checkpoint"])] = int(row["sessions_ingested"])  # type: ignore[arg-type]
    ingest_latency: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in latency:
        if row["phase"] != "ingestion":
            continue
        key = (str(row["checkpoint"]), condition_variant[str(row["condition_id"])])
        ingest_latency[key].append(float(row["latency_ms"]))  # type: ignore[arg-type]

    checkpoint_rows: list[dict[str, object]] = []
    for (checkpoint, variant), by_pair in sorted(outcomes.items()):
        n = len(by_pair)
        counts = labels[(checkpoint, variant)]
        rows = storage[(checkpoint, variant)]
        checkpoint_rows.append(
            {
                "checkpoint": checkpoint,
                "sessions": sessions_at.get(checkpoint),
                "variant": variant,
                "n": n,
                "accuracy": round(sum(by_pair.values()) / n, 4),
                "stale_rate": round(counts["stale"] / n, 4),
                "leak_rate": round(counts["leaked"] / n, 4),
                "wrongly_abstained_rate": round(counts["abstained_wrongly"] / n, 4),
                "unsupported_rate": round(counts["unsupported"] / n, 4),
                "active_records_mean": round(
                    statistics.mean(float(r["active_memory_count"]) for r in rows),
                    1,  # type: ignore[arg-type]
                ),
                "superseded_records_mean": round(
                    statistics.mean(float(r["superseded_record_count"]) for r in rows),
                    1,  # type: ignore[arg-type]
                ),
                "deleted_records_mean": round(
                    statistics.mean(float(r["deleted_record_count"]) for r in rows),
                    1,  # type: ignore[arg-type]
                ),
                "database_kib_mean": round(
                    statistics.mean(float(r["database_size_bytes"]) for r in rows) / 1024,
                    1,  # type: ignore[arg-type]
                ),
                "index_kib_mean": round(
                    statistics.mean(float(r["index_size_bytes"]) for r in rows) / 1024,
                    1,  # type: ignore[arg-type]
                ),
                "ingest_p95_ms": round(percentile(ingest_latency[(checkpoint, variant)], 0.95), 3),
            }
        )

    sweep: dict[tuple[int, str], dict[str, str]] = {}
    if SWEEP.exists():
        with SWEEP.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sweep[(int(row["sessions"]), row["variant"])] = row

    gap_rows: list[dict[str, object]] = []
    for checkpoint in sorted({cp for cp, _ in outcomes}):
        reference = outcomes[(checkpoint, "guarded_memory")]
        for variant in sorted(
            {v for cp, v in outcomes if cp == checkpoint and v != "guarded_memory"}
        ):
            other = outcomes[(checkpoint, variant)]
            pairs = sorted(set(reference) & set(other))
            a = [reference[p] for p in pairs]
            b = [other[p] for p in pairs]
            diff, low, high = paired_difference_interval(a, b)
            test = mcnemar(a, b)
            per_seed: dict[int, list[tuple[bool, bool]]] = defaultdict(list)
            for pair in pairs:
                per_seed[pair[1]].append((reference[pair], other[pair]))
            stream_diffs = [
                sum(x for x, _ in items) / len(items) - sum(y for _, y in items) / len(items)
                for _, items in sorted(per_seed.items())
            ]
            c_low, c_high = cluster_bootstrap(stream_diffs, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
            sessions = sessions_at.get(checkpoint)
            between = sweep.get((sessions, variant), {}) if sessions is not None else {}
            gap_rows.append(
                {
                    "checkpoint": checkpoint,
                    "sessions": sessions,
                    "variant": variant,
                    "n_pairs": len(pairs),
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
                    "sweep_gap": between.get("gap_vs_reference"),
                    "sweep_gap_ci_low": between.get("gap_ci_low"),
                    "sweep_gap_ci_high": between.get("gap_ci_high"),
                    "sweep_n": between.get("n"),
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("checkpoints.csv", checkpoint_rows), ("gap_vs_guarded.csv", gap_rows)):
        with (OUTPUT_DIR / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {OUTPUT_DIR / name} from {run_dir.name}")


if __name__ == "__main__":
    main()
