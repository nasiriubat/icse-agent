"""Spreadsheet export of every figure's underlying data.

Rasterised figures are a dead end for anyone who wants to re-plot: the numbers are gone. This
module writes the same data the figure generator plots, as an ``.xlsx`` workbook, so a figure can
be rebuilt in a spreadsheet, a different plotting library, or a paper's own TikZ pipeline without
re-running an experiment.

Two shapes are written, chosen so a chart can be inserted over a selected block with no further
work:

* **wide** for anything drawn against store growth --- one row per checkpoint, one column per
  condition. Selecting the block and inserting a line chart reproduces the figure.
* **long** for scatter and category plots, where a wide layout would be a lie about the data.

Every sheet is preceded by a ``README`` carrying the same provenance stamp as the figures, so a
workbook that outlives this repository still says which runs produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from memcheck.reporting.figures import CHECKPOINT_ORDER
from memcheck.reporting.provenance import Provenance

#: Excel rejects these in a sheet name, and truncates silently past 31 characters.
_ILLEGAL_SHEET_CHARS = str.maketrans(dict.fromkeys("[]:*?/\\", "_"))
MAX_SHEET_NAME = 31

#: Excel's hard row ceiling is 1,048,576. Raw event-level frames can approach it; we refuse to
#: write a truncated sheet, because a silently shortened table is worse than an absent one.
MAX_ROWS = 1_000_000


@dataclass
class SheetResult:
    name: str
    rows: int
    skipped: bool
    reason: str = ""


@dataclass
class WorkbookResult:
    path: Path | None
    sheets: list[SheetResult]
    skipped: bool = False
    reason: str = ""


def _sheet_name(name: str, taken: set[str]) -> str:
    cleaned = name.translate(_ILLEGAL_SHEET_CHARS)[:MAX_SHEET_NAME]
    candidate = cleaned
    suffix = 2
    while candidate.lower() in taken:
        tail = f"~{suffix}"
        candidate = cleaned[: MAX_SHEET_NAME - len(tail)] + tail
        suffix += 1
    taken.add(candidate.lower())
    return candidate


def _ordered_checkpoints(frame: pd.DataFrame) -> list[str]:
    present = [cp for cp in CHECKPOINT_ORDER if cp in set(frame["checkpoint"])]
    return present or sorted(str(cp) for cp in frame["checkpoint"].dropna().unique())


def _growth_pivot(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """One row per checkpoint, one column per condition --- the shape a line chart wants.

    The unfaulted condition is used when a fault column is present, matching the figure
    generator, so the workbook and the figure never disagree.
    """
    if frame.empty or value_column not in frame.columns:
        return pd.DataFrame()
    data = frame[frame["fault"] == "none"] if "fault" in frame.columns else frame
    if data.empty:
        data = frame
    checkpoints = _ordered_checkpoints(data)
    pivot = (
        data.pivot_table(index="checkpoint", columns="variant", values=value_column, aggfunc="mean")
        .reindex(checkpoints)
        .rename_axis(index="checkpoint", columns=None)
    )
    # A percentage label is what the figure's axis shows, so the spreadsheet shows it too. The
    # zero-padding in the checkpoint id exists to make ids sort lexically; it is noise on an axis.
    pivot.insert(0, "store_growth_pct", [int(cp.replace("cp", "")) for cp in pivot.index])
    return pivot.reset_index()


def _final_checkpoint(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "checkpoint" not in frame.columns:
        return frame
    final: pd.DataFrame = frame[frame["checkpoint"] == frame["checkpoint"].max()]
    return final


def _figure_sheets(
    frames: dict[str, pd.DataFrame], metrics: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """The exact series behind each of the eight figures, keyed by figure name."""
    utility = metrics.get("task_utility", pd.DataFrame())
    efficiency = metrics.get("efficiency", pd.DataFrame())
    storage = metrics.get("storage_growth", pd.DataFrame())
    violations = frames.get("violations", pd.DataFrame())

    sheets: dict[str, pd.DataFrame] = {
        "fig1_quality_vs_growth": _growth_pivot(utility, "accuracy"),
        "fig2_stale_vs_growth": _growth_pivot(utility, "stale_memory_error_rate"),
        "fig5_latency_vs_growth": _growth_pivot(efficiency, "latency_p95_ms_retrieval"),
    }

    # Figure 3: a single bar per condition at the full store.
    if not utility.empty and "deletion_leakage_rate" in utility.columns:
        leakage = (
            _final_checkpoint(utility)
            .groupby("variant", as_index=False)[["deletion_leakage_rate"]]
            .mean()
            .sort_values("deletion_leakage_rate")
        )
        sheets["fig3_deletion_leakage"] = leakage

    # Figure 4: stacked counts, so backends are rows and faults are columns.
    if not violations.empty:
        sheets["fig4_violations_by_fault"] = (
            violations.assign(fault_name=violations["fault_name"].fillna("none"))
            .groupby(["backend_name", "fault_name"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .rename_axis(columns=None)
        )

    # Figures 6 and 8 are scatters: one point per (condition, fault, checkpoint), so they stay
    # long. Wide would force a column per series and lose the pairing that makes the point.
    if not utility.empty and not efficiency.empty:
        merged = utility.merge(efficiency, on=["variant", "fault", "checkpoint"], how="inner")
        if not merged.empty and "injected_tokens_mean" in merged.columns:
            sheets["fig6_tokens_vs_quality"] = merged[
                ["variant", "fault", "checkpoint", "injected_tokens_mean", "accuracy"]
            ]
        if not merged.empty and "estimated_cost_usd" in merged.columns:
            # The figure is skipped when every run was free; the data sheet is still written,
            # because a column of zeros is a fact about the runs worth carrying.
            sheets["fig8_success_vs_cost"] = merged[
                ["variant", "fault", "checkpoint", "estimated_cost_usd", "accuracy"]
            ]

    if not storage.empty and "database_size_bytes" in storage.columns:
        scaled = storage.assign(database_size_kb=storage["database_size_bytes"] / 1024)
        sheets["fig7_storage_growth"] = _growth_pivot(scaled, "database_size_kb")

    return sheets


def _readme(provenance: Provenance, group: str, sheets: list[SheetResult]) -> pd.DataFrame:
    rows: list[tuple[str, str]] = [
        ("artifact", provenance.artifact),
        ("comparability group", group),
        ("generating script", provenance.script),
        ("source run ids", ", ".join(provenance.run_ids) or "none"),
        ("configuration hashes", ", ".join(provenance.configuration_hashes) or "none"),
        ("generated at", provenance.generated_at),
        (
            "regenerate with",
            "uv run memcheck report --results results/raw --paper paper",
        ),
        ("", ""),
        ("sheet", "rows (or reason for omission)"),
    ]
    rows.extend(
        (sheet.name, sheet.reason if sheet.skipped else str(sheet.rows)) for sheet in sheets
    )
    return pd.DataFrame(rows, columns=["field", "value"])


def write_workbook(
    frames: dict[str, pd.DataFrame],
    metrics: dict[str, pd.DataFrame],
    output: Path,
    provenance: Provenance,
    *,
    group: str = "all",
) -> WorkbookResult:
    """Write every figure series and every metric frame to one ``.xlsx`` file."""
    candidates: dict[str, pd.DataFrame] = dict(_figure_sheets(frames, metrics))
    # The aggregated metric frames are what the tables are built from, and they carry columns the
    # figures never plot (confidence bounds, counts, per-label rates), so they are exported whole.
    for name, frame in sorted(metrics.items()):
        candidates[f"metric_{name}"] = frame
    # Violations are the one raw frame small enough and useful enough to ship as-is: each row is
    # a reproduction trace, and it is the evidence behind every contract claim.
    raw_violations = frames.get("violations", pd.DataFrame())
    if not raw_violations.empty:
        candidates["raw_violations"] = raw_violations

    results: list[SheetResult] = []
    writable: dict[str, pd.DataFrame] = {}
    taken: set[str] = {"readme"}
    for name, frame in candidates.items():
        if frame is None or frame.empty:
            results.append(SheetResult(name, 0, True, "no data for this series in these runs"))
            continue
        if len(frame) > MAX_ROWS:
            results.append(
                SheetResult(name, len(frame), True, f"{len(frame)} rows exceeds the sheet limit")
            )
            continue
        writable[_sheet_name(name, taken)] = frame
        results.append(SheetResult(name, len(frame), False))

    if not writable:
        return WorkbookResult(None, results, True, "no sheet had data to write")

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _readme(provenance, group, results).to_excel(writer, sheet_name="README", index=False)
        for sheet_name, frame in writable.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
        _autofit(writer)
    provenance.write_sidecar(output)
    return WorkbookResult(output, results)


def _autofit(writer: pd.ExcelWriter) -> None:
    """Widen columns to their content, and freeze the header row.

    Purely cosmetic, but a workbook whose columns all read ``####`` is one the reader has to
    repair before it is usable.
    """
    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = "A2"
        for column in worksheet.columns:
            longest = max(
                (len(str(cell.value)) for cell in column if cell.value is not None), default=0
            )
            letter = column[0].column_letter
            worksheet.column_dimensions[letter].width = min(max(longest + 2, 10), 60)
