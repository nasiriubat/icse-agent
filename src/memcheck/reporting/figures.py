"""Figure generation (Section 19).

Design rules, chosen for a two-column IEEE paper that must also survive greyscale printing:

* one colourblind-safe categorical palette, used consistently for the same backend everywhere;
* every series additionally distinguished by marker and line style, so colour is never the only
  channel carrying meaning;
* no chartjunk — no gridlines heavier than the data, no 3-D, no dual axes;
* every figure carries a provenance footer naming its source runs and configuration hashes.

A figure whose data is absent is **not** drawn. The generator records it as skipped so the paper
renders ``[RESULT PENDING: ...]`` instead of an empty or misleading panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from memcheck.reporting.provenance import Provenance

#: Okabe-Ito, a colourblind-safe qualitative palette.
PALETTE: dict[str, str] = {
    "no_memory": "#999999",
    "full_history": "#E69F00",
    "transcript_rag": "#0072B2",
    "guarded_memory": "#009E73",
}
ABLATION_COLOR = "#CC79A7"
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES = ("-", "--", "-.", ":")

CHECKPOINT_ORDER = ["cp010", "cp025", "cp050", "cp075", "cp100"]


@dataclass
class FigureResult:
    name: str
    path: Path | None
    skipped: bool
    reason: str = ""


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (5.2, 3.2),
            "figure.dpi": 200,
            "font.size": 8,
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "savefig.bbox": "tight",
        }
    )


def _color(variant: str) -> str:
    return PALETTE.get(variant, ABLATION_COLOR)


def _ordered_checkpoints(frame: pd.DataFrame) -> list[str]:
    present = [cp for cp in CHECKPOINT_ORDER if cp in set(frame["checkpoint"])]
    return present or sorted(set(frame["checkpoint"].dropna()))


def _finish(figure: Figure, provenance: Provenance, path: Path, *, pdf_footer: bool = True) -> Path:
    """Save ``path`` (PNG) and its PDF twin, with the provenance footer and sidecar.

    With ``pdf_footer=False`` the PDF is saved before the footer is drawn: the PNG and the
    sidecar still carry the provenance, and the copy that goes into the paper carries only the
    plot, as a figure drawn by hand would.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not pdf_footer:
        figure.savefig(path.with_suffix(".pdf"))
    # Wrapped at the figure edge: a footer wider than the figure would otherwise widen the
    # saved canvas, and a single-column figure that is not single-column width shrinks on the
    # page until its tick labels are unreadable.
    figure.text(
        0.0,
        -0.06,
        provenance.figure_footer(),
        fontsize=4.5,
        color="#555555",
        transform=figure.transFigure,
        wrap=True,
        va="top",
    )
    figure.savefig(path)
    if pdf_footer:
        figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    provenance.write_sidecar(path)
    return path


def _growth_line(
    frame: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    output: Path,
    provenance: Provenance,
    *,
    x_column: str = "checkpoint",
) -> FigureResult:
    if frame.empty or value_column not in frame.columns:
        return FigureResult(output.stem, None, True, f"{value_column} is not present in results")
    data = frame[frame["fault"] == "none"] if "fault" in frame.columns else frame
    if data.empty:
        data = frame
    checkpoints = _ordered_checkpoints(data)
    figure, axes = plt.subplots()
    plotted = 0
    for index, (variant_key, group) in enumerate(sorted(data.groupby("variant"))):
        variant = str(variant_key)
        series = group.set_index(x_column).reindex(checkpoints)[value_column]
        if series.dropna().empty:
            continue
        axes.plot(
            list(range(len(checkpoints))),
            list(series.to_numpy()),
            label=variant,
            color=_color(variant),
            marker=MARKERS[index % len(MARKERS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            linewidth=1.2,
            markersize=3.5,
        )
        plotted += 1
    if not plotted:
        plt.close(figure)
        return FigureResult(output.stem, None, True, "no series had data to plot")
    axes.set_xticks(range(len(checkpoints)))
    axes.set_xticklabels([cp.replace("cp", "") + "%" for cp in checkpoints])
    axes.set_xlabel("store growth (fraction of the session stream ingested)")
    axes.set_ylabel(ylabel)
    axes.set_title(title, fontsize=8)
    axes.legend(ncol=2)
    return FigureResult(output.stem, _finish(figure, provenance, output), False)


def figure_quality_vs_growth(
    metrics: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> FigureResult:
    return _growth_line(
        metrics.get("task_utility", pd.DataFrame()),
        "accuracy",
        "task accuracy",
        "Task quality against store growth",
        output_dir / "fig1_quality_vs_growth.png",
        provenance,
    )


def figure_stale_vs_growth(
    metrics: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> FigureResult:
    return _growth_line(
        metrics.get("task_utility", pd.DataFrame()),
        "stale_memory_error_rate",
        "stale memory error rate",
        "Stale memory errors against store growth",
        output_dir / "fig2_stale_vs_growth.png",
        provenance,
    )


def figure_deletion_leakage(
    metrics: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> FigureResult:
    frame = metrics.get("task_utility", pd.DataFrame())
    output = output_dir / "fig3_deletion_leakage.png"
    if frame.empty or "deletion_leakage_rate" not in frame.columns:
        return FigureResult(output.stem, None, True, "no grades with a deletion-leakage label")
    final = frame[frame["checkpoint"] == frame["checkpoint"].max()]
    grouped = final.groupby("variant")["deletion_leakage_rate"].mean().sort_values()
    if grouped.dropna().empty:
        return FigureResult(output.stem, None, True, "deletion leakage rate is undefined")
    figure, axes = plt.subplots()
    axes.barh(
        list(grouped.index),
        list(grouped.values),
        color=[_color(str(name)) for name in grouped.index],
        height=0.6,
    )
    axes.set_xlabel("deletion leakage rate at the full store")
    axes.set_title("Deletion leakage by backend", fontsize=8)
    for position, value in enumerate(grouped.values):
        axes.text(value, position, f" {value:.3f}", va="center", fontsize=6)
    return FigureResult(output.stem, _finish(figure, provenance, output), False)


def figure_violations_by_fault(
    frames: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> FigureResult:
    violations = frames.get("violations", pd.DataFrame())
    output = output_dir / "fig4_violations_by_fault.png"
    if violations.empty:
        return FigureResult(output.stem, None, True, "no contract violations were recorded")
    pivot = (
        violations.assign(fault_name=violations["fault_name"].fillna("none"))
        .groupby(["backend_name", "fault_name"])
        .size()
        .unstack(fill_value=0)
    )
    figure, axes = plt.subplots(figsize=(5.6, 3.2))
    bottom = [0.0] * len(pivot.index)
    for index, fault in enumerate(pivot.columns):
        values = list(pivot[fault].values)
        axes.bar(
            list(pivot.index),
            values,
            bottom=bottom,
            label=fault,
            color=plt.cm.tab20(index / max(1, len(pivot.columns))),
            edgecolor="white",
            linewidth=0.4,
        )
        bottom = [b + v for b, v in zip(bottom, values, strict=True)]
    axes.set_ylabel("contract violations")
    axes.set_title("Contract violations by fault and backend", fontsize=8)
    axes.tick_params(axis="x", rotation=20)
    axes.legend(ncol=2, title="fault", title_fontsize=7)
    return FigureResult(output.stem, _finish(figure, provenance, output), False)


def figure_latency_vs_growth(
    metrics: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> FigureResult:
    frame = metrics.get("efficiency", pd.DataFrame())
    column = "latency_p95_ms_retrieval"
    if column not in frame.columns:
        return FigureResult(
            "fig5_latency_vs_growth", None, True, "no retrieval latency measurements"
        )
    return _growth_line(
        frame,
        column,
        "p95 retrieval latency (ms)",
        "p95 retrieval latency against store growth",
        output_dir / "fig5_latency_vs_growth.png",
        provenance,
    )


def figure_tokens_vs_quality(
    metrics: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> FigureResult:
    utility = metrics.get("task_utility", pd.DataFrame())
    efficiency = metrics.get("efficiency", pd.DataFrame())
    output = output_dir / "fig6_tokens_vs_quality.png"
    if utility.empty or efficiency.empty:
        return FigureResult(output.stem, None, True, "task utility or efficiency data is missing")
    merged = utility.merge(efficiency, on=["variant", "fault", "checkpoint"], how="inner")
    if merged.empty or "injected_tokens_mean" not in merged.columns:
        return FigureResult(output.stem, None, True, "no injected-token measurements to join")
    figure, axes = plt.subplots()
    for index, (variant_key, group) in enumerate(sorted(merged.groupby("variant"))):
        variant = str(variant_key)
        axes.scatter(
            group["injected_tokens_mean"],
            group["accuracy"],
            label=variant,
            color=_color(variant),
            marker=MARKERS[index % len(MARKERS)],
            s=22,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
    axes.set_xlabel("mean injected memory tokens per answer")
    axes.set_ylabel("task accuracy")
    axes.set_title("Injected tokens against task quality", fontsize=8)
    axes.legend(ncol=2)
    return FigureResult(output.stem, _finish(figure, provenance, output), False)


def figure_storage_growth(
    metrics: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> FigureResult:
    frame = metrics.get("storage_growth", pd.DataFrame())
    if frame.empty:
        return FigureResult("fig7_storage_growth", None, True, "no store statistics recorded")
    scaled = frame.assign(database_size_kb=frame["database_size_bytes"] / 1024)
    return _growth_line(
        scaled,
        "database_size_kb",
        "database size (KiB)",
        "Storage growth by backend",
        output_dir / "fig7_storage_growth.png",
        provenance,
    )


def figure_success_vs_cost(
    metrics: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> FigureResult:
    utility = metrics.get("task_utility", pd.DataFrame())
    efficiency = metrics.get("efficiency", pd.DataFrame())
    output = output_dir / "fig8_success_vs_cost.png"
    if utility.empty or efficiency.empty or "estimated_cost_usd" not in efficiency.columns:
        return FigureResult(output.stem, None, True, "no cost records to plot against success")
    merged = utility.merge(efficiency, on=["variant", "fault", "checkpoint"], how="inner")
    if merged["estimated_cost_usd"].fillna(0).abs().sum() == 0:
        return FigureResult(
            output.stem,
            None,
            True,
            "every run used the free mock provider, so monetary cost is identically zero and a "
            "success-against-cost plot would be meaningless",
        )
    figure, axes = plt.subplots()
    for index, (variant_key, group) in enumerate(sorted(merged.groupby("variant"))):
        variant = str(variant_key)
        axes.scatter(
            group["estimated_cost_usd"],
            group["accuracy"],
            label=variant,
            color=_color(variant),
            marker=MARKERS[index % len(MARKERS)],
            s=22,
        )
    axes.set_xlabel("estimated cost (USD)")
    axes.set_ylabel("task accuracy")
    axes.set_title("Task success against monetary cost", fontsize=8)
    axes.legend(ncol=2)
    return FigureResult(output.stem, _finish(figure, provenance, output), False)


def generate_all(
    frames: dict[str, pd.DataFrame],
    metrics: dict[str, pd.DataFrame],
    output_dir: Path,
    provenance: Provenance,
) -> list[FigureResult]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        figure_quality_vs_growth(metrics, output_dir, provenance),
        figure_stale_vs_growth(metrics, output_dir, provenance),
        figure_deletion_leakage(metrics, output_dir, provenance),
        figure_violations_by_fault(frames, output_dir, provenance),
        figure_latency_vs_growth(metrics, output_dir, provenance),
        figure_tokens_vs_quality(metrics, output_dir, provenance),
        figure_storage_growth(metrics, output_dir, provenance),
        figure_success_vs_cost(metrics, output_dir, provenance),
    ]
