"""Turn immutable raw runs into processed tables.

Raw files are opened read-only. Everything written here goes to ``results/processed/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from memcheck.evaluation.runner import discover_runs, load_manifest
from memcheck.utilities.jsonl import read_jsonl

RAW_STREAMS = (
    "grades",
    "retrievals",
    "violations",
    "contracts",
    "faults",
    "store_statistics",
    "latency",
    "costs",
    "answers",
    "writes",
    "coding_tasks",
)


def load_run_frame(run_dir: Path, stream: str) -> pd.DataFrame:
    """Read one JSONL stream from one run into a frame, tagged with the run id."""
    records = list(read_jsonl(run_dir / f"{stream}.jsonl"))
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame["run_id"] = run_dir.name
    return frame


def condition_index(run_dir: Path) -> pd.DataFrame:
    """Map condition ids to their backend, ablations, fault, and seed.

    The contracts stream carries one row per condition and contract, so it is the cheapest
    complete index of the conditions a run actually executed.
    """
    frame = load_run_frame(run_dir, "contracts")
    if frame.empty:
        return pd.DataFrame(
            columns=["condition_id", "backend", "ablations", "fault", "seed", "run_id", "variant"]
        )
    index = (
        frame[["condition_id", "backend_name", "ablations", "fault", "seed", "run_id"]]
        .drop_duplicates(subset=["condition_id"])
        .rename(columns={"backend_name": "backend"})
        .reset_index(drop=True)
    )
    index["ablations"] = index["ablations"].apply(
        lambda value: tuple(sorted(value)) if isinstance(value, list) else ()
    )
    index["variant"] = index.apply(
        lambda row: row["backend"] + ("+" + "+".join(row["ablations"]) if row["ablations"] else ""),
        axis=1,
    )
    return index


def load_all(
    results_root: Path,
    streams: tuple[str, ...] = RAW_STREAMS,
    *,
    run_dirs: list[Path] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the given runs (default: all) into one frame per stream, joined to the index."""
    runs = run_dirs if run_dirs is not None else discover_runs(results_root)
    if not runs:
        raise FileNotFoundError(
            f"no runs found under {results_root}. Run `memcheck run --config configs/smoke.yaml` "
            "first."
        )
    indexes = pd.concat([condition_index(run) for run in runs], ignore_index=True)
    frames: dict[str, pd.DataFrame] = {"conditions": indexes}
    for stream in streams:
        parts = [load_run_frame(run, stream) for run in runs]
        parts = [part for part in parts if not part.empty]
        if not parts:
            frames[stream] = pd.DataFrame()
            continue
        frame = pd.concat(parts, ignore_index=True)
        if "condition_id" in frame.columns and not indexes.empty:
            frame = frame.merge(
                indexes[["condition_id", "backend", "variant", "ablations", "fault", "seed"]],
                on="condition_id",
                how="left",
                suffixes=("", "_cond"),
            )
        frames[stream] = frame
    return frames


def manifests_frame(results_root: Path, *, run_dirs: list[Path] | None = None) -> pd.DataFrame:
    rows = []
    for run in run_dirs if run_dirs is not None else discover_runs(results_root):
        manifest = load_manifest(run)
        payload = manifest.model_dump(mode="json")
        payload["prompt_hashes"] = json.dumps(payload["prompt_hashes"], sort_keys=True)
        payload["random_seeds"] = json.dumps(payload["random_seeds"])
        payload["checkpoint_schedule"] = json.dumps(payload["checkpoint_schedule"])
        payload["ablations"] = json.dumps(payload["ablations"])
        payload["faults"] = json.dumps(payload["faults"])
        rows.append(payload)
    return pd.DataFrame.from_records(rows)


def write_processed(frames: dict[str, pd.DataFrame], processed_dir: Path) -> list[Path]:
    """Write CSV (authoritative) and Parquet (when pyarrow is present) for each frame."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in frames.items():
        if frame.empty:
            continue
        csv_path = processed_dir / f"{name}.csv"
        serializable = frame.copy()
        for column in serializable.columns:
            if serializable[column].map(lambda value: isinstance(value, (list, dict, tuple))).any():
                serializable[column] = serializable[column].map(
                    lambda value: (
                        json.dumps(value, default=str, sort_keys=True)
                        if isinstance(value, (list, dict, tuple))
                        else value
                    )
                )
        serializable.to_csv(csv_path, index=False)
        written.append(csv_path)
        try:
            parquet_path = processed_dir / f"{name}.parquet"
            serializable.to_parquet(parquet_path, index=False)
            written.append(parquet_path)
        except (ImportError, ValueError):
            # pyarrow is an optional extra; CSV is authoritative (DEVIATIONS V-008).
            pass
    return written
