"""Workload v2 versus v4: does correcting the two generator defects change any result?

v4 differs from v2 only in two fixes (DEVIATIONS V-024): no fact asserts the internal
contradiction marker, and a temporal query's gold is the value current just before the last
change of value. The sweep is analysed with ``memcheck.reporting.scale.analyse``, the same code
that produces the paper's sweep tables, once per workload version. Writes, under
results/processed/v4/:

* sweep_comparison.csv: per size and variant, v2 and v4 side by side (accuracies, pooled gap and
  Newcombe interval, pooled Holm p, stream-level mean, cluster interval and sign-flip p, verdict).
* first_separation.csv: the smallest size at which each variant separates under the two-test rule.
* category_gap_064.csv: guarded-minus-transcript gap within each question type at 64 sessions.
* matrix_comparison.csv: per condition, contract violations per suite run and the fraction of
  fault injections that recovered, v2 against v4, in the 40-session matrix.

    uv run python scripts/v4_robustness.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from memcheck.reporting.scale import analyse

RAW = Path("results/raw")
OUT = Path("results/processed/v4")
SIZES = ("008", "016", "024", "040", "064")
VARIANTS = (
    "transcript_rag",
    "guarded_memory+abl_no_supersession",
    "guarded_memory+abl_no_tombstones",
    "full_history",
    "no_memory",
)
CORRECT = {"correct", "abstained_correctly"}


def _single(pattern: str) -> Path:
    matches = sorted(RAW.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected one run for {pattern!r}, found {[m.name for m in matches]}")
    return matches[0]


def _sweep_runs(version: str) -> list[Path]:
    if version == "v2":
        return [_single(f"scale_v2_{size}") for size in SIZES]
    return [_single(f"scale_v4_{size}_*") for size in SIZES]


def _matrix_run(version: str) -> Path:
    if version == "v2":
        return _single("experiment_matrix_2*")
    return _single("experiment_matrix_v4_*")


def _variant_name(record: dict[str, object]) -> str:
    ablations = record.get("ablations") or []
    return "+".join([str(record["backend_name"]), *[str(a) for a in ablations]])  # type: ignore[union-attr]


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sweep(version: str) -> pd.DataFrame:
    analysis = analyse(RAW, run_dirs=_sweep_runs(version))
    if not analysis.ok:
        raise SystemExit(f"{version}: {analysis.problems}")
    accuracy = analysis.accuracy.set_index(["sessions", "variant"])
    streams = analysis.streams.set_index(["sessions", "variant"])
    joined = accuracy.join(
        streams[["mean_diff", "ci_low", "ci_high"]].rename(
            columns={
                "mean_diff": "stream_mean",
                "ci_low": "cluster_ci_low",
                "ci_high": "cluster_ci_high",
            }
        ),
        how="left",
    )
    columns = {
        "n": "n",
        "reference_accuracy": "guarded_acc",
        "accuracy": "variant_acc",
        "gap_vs_reference": "gap",
        "gap_ci_low": "gap_ci_low",
        "gap_ci_high": "gap_ci_high",
        "p_value_holm": "p_holm",
        "stream_mean": "stream_mean",
        "cluster_ci_low": "cluster_ci_low",
        "cluster_ci_high": "cluster_ci_high",
        "sign_flip_p": "sign_flip_p",
        "separated_from_reference": "separated",
    }
    frame = joined[list(columns)].rename(columns=columns)
    frame.columns = [f"{version}_{name}" for name in frame.columns]
    return frame


def category_gap(version: str) -> dict[str, tuple[int, float]]:
    run = _sweep_runs(version)[-1]
    outcomes: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for grade in _jsonl(run / "grades.jsonl"):
        if grade["checkpoint"] != "cp100" or grade["fault"] != "none" or grade["ablations"]:
            continue
        backend = str(grade["backend_name"])
        if backend in ("guarded_memory", "transcript_rag"):
            category = str(grade["metadata"]["category"])  # type: ignore[index]
            outcomes[category][backend].append(grade["label"] in CORRECT)
    result = {}
    for category, by in outcomes.items():
        guarded, transcript = by["guarded_memory"], by["transcript_rag"]
        gap = sum(guarded) / len(guarded) - sum(transcript) / len(transcript)
        result[category] = (len(guarded), gap)
    return result


def matrix(version: str) -> dict[str, dict[str, float]]:
    run = _matrix_run(version)
    contracts = _jsonl(run / "contracts.jsonl")
    variant_of = {str(r["condition_id"]): _variant_name(r) for r in contracts}
    for grade in _jsonl(run / "grades.jsonl"):
        variant_of.setdefault(str(grade["condition_id"]), _variant_name(grade))
    violations: dict[str, int] = defaultdict(int)
    suite_runs: dict[str, set[tuple[object, object]]] = defaultdict(set)
    vacuous: dict[str, int] = defaultdict(int)
    for record in contracts:
        variant = _variant_name(record)
        suite_runs[variant].add((record["fault"], record["seed"]))
        violations[variant] += len(record["violations"])  # type: ignore[arg-type]
        vacuous[variant] += bool(record["vacuous"])
    recovered: dict[str, list[bool]] = defaultdict(list)
    for record in _jsonl(run / "faults.jsonl"):
        recovered[variant_of[str(record["condition_id"])]].append(bool(record["recovered"]))
    return {
        variant: {
            "violations_per_suite_run": violations[variant] / len(suite_runs[variant]),
            "vacuous": vacuous[variant],
            "fault_recovery": sum(recovered[variant]) / len(recovered[variant])
            if recovered[variant]
            else float("nan"),
        }
        for variant in sorted(suite_runs)
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    both = sweep("v2").join(sweep("v4"), how="outer").reset_index()
    both = both[both["variant"].isin(VARIANTS)].sort_values(["variant", "sessions"])
    both["verdict_changed"] = both["v2_separated"] != both["v4_separated"]
    both.to_csv(OUT / "sweep_comparison.csv", index=False)

    with (OUT / "first_separation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        # Direction matters: a separation with the variant ahead is not the guarded store
        # pulling away, so the two are reported apart.
        writer.writerow(
            [
                "variant",
                "v2_first_guarded_ahead",
                "v4_first_guarded_ahead",
                "v2_sizes_variant_ahead",
                "v4_sizes_variant_ahead",
            ]
        )
        for variant in VARIANTS:
            rows = both[both["variant"] == variant]
            first: dict[str, object] = {}
            ahead: dict[str, str] = {}
            for version in ("v2", "v4"):
                separated = rows[rows[f"{version}_separated"] == True]  # noqa: E712
                guarded = separated[separated[f"{version}_gap"] > 0]["sessions"]
                first[version] = int(guarded.min()) if not guarded.empty else ""
                behind = separated[separated[f"{version}_gap"] < 0]["sessions"]
                ahead[version] = " ".join(str(int(size)) for size in behind)
            writer.writerow([variant, first["v2"], first["v4"], ahead["v2"], ahead["v4"]])

    v2_categories, v4_categories = category_gap("v2"), category_gap("v4")
    with (OUT / "category_gap_064.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "v2_n", "v2_gap", "v4_n", "v4_gap"])
        for category in sorted(set(v2_categories) | set(v4_categories)):
            v2_n, v2_gap = v2_categories.get(category, (0, float("nan")))
            v4_n, v4_gap = v4_categories.get(category, (0, float("nan")))
            writer.writerow([category, v2_n, v2_gap, v4_n, v4_gap])

    v2_matrix, v4_matrix = matrix("v2"), matrix("v4")
    with (OUT / "matrix_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "v2_violations_per_suite_run",
                "v4_violations_per_suite_run",
                "v2_vacuous",
                "v4_vacuous",
                "v2_fault_recovery",
                "v4_fault_recovery",
            ]
        )
        for variant in sorted(set(v2_matrix) | set(v4_matrix)):
            a, b = v2_matrix.get(variant, {}), v4_matrix.get(variant, {})
            writer.writerow(
                [
                    variant,
                    a.get("violations_per_suite_run"),
                    b.get("violations_per_suite_run"),
                    a.get("vacuous"),
                    b.get("vacuous"),
                    a.get("fault_recovery"),
                    b.get("fault_recovery"),
                ]
            )
    print(
        f"wrote {OUT}/{{sweep_comparison,first_separation,category_gap_064,matrix_comparison}}.csv"
    )


if __name__ == "__main__":
    main()
