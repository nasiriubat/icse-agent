"""LaTeX table generation (Section 19).

Tables are emitted as standalone ``.tex`` fragments that ``paper/main.tex`` includes. Each
carries a provenance comment block. A table whose data is absent is emitted as an explicit
``[RESULT PENDING: ...]`` note rather than as an empty tabular — the paper must never present a
blank cell as though it were a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from memcheck.reporting.provenance import Provenance

#: Columns beyond this many do not fit an IEEE column at ``\small``; the table drops a size.
MAX_SMALL_COLUMNS = 4


def _stack(*lines: str) -> str:
    """A two-line column heading, so a heading never sets the width of a numeric column."""
    return r"\begin{tabular}[b]{@{}r@{}}" + r"\\".join(lines) + r"\end{tabular}"


PENDING_TEMPLATE = r"""%% This table has no supporting artifact yet.
\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\begin{{tabular}}{{l}}
\hline
\texttt{{[RESULT PENDING: {reason}]}} \\
\hline
\end{{tabular}}
\end{{table}}
"""


@dataclass
class TableResult:
    name: str
    path: Path
    pending: bool
    reason: str = ""


def _escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "--"
    # ``bool`` is a subclass of ``int``, so it must be caught first or it prints as 1/0.
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        return f"{value:.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return _escape(str(value))


#: How the paper names the four backends; the prose uses these, so the tables must too.
BACKEND_DISPLAY_NAMES: dict[str, str] = {
    "no_memory": "no memory",
    "full_history": "full history",
    "transcript_rag": "transcript chunking",
    "guarded_memory": "guarded memory",
}


def display_name(variant: str) -> str:
    """A condition identifier as a reader should see it, not as the code spells it.

    ``guarded_memory+abl_no_supersession`` becomes ``guarded, no supersession``; a plain
    backend name just has its underscores replaced. The identifiers stay in every CSV and
    workbook, so the mapping is one-way and purely presentational.
    """
    base, _, rest = variant.partition("+")
    ablations = [part for part in rest.split("+") if part]
    if not ablations:
        return BACKEND_DISPLAY_NAMES.get(base, base.replace("_", " "))
    head = base.split("_")[0]
    tail = ", ".join(ablation.removeprefix("abl_").replace("_", " ") for ablation in ablations)
    return f"{head}, {tail}"


def _tabular(
    frame: pd.DataFrame,
    columns: list[tuple[str, str]],
    caption: str,
    label: str,
    provenance: Provenance,
    *,
    digits: int = 3,
    note: str = "",
    wide: bool = False,
    fontsize: str | None = None,
    rotate_headers: bool = False,
) -> str:
    """Render one LaTeX table.

    ``wide`` emits a ``table*`` spanning both columns. ``fontsize`` defaults to ``\\small``,
    dropping to ``\\footnotesize`` for wide tables and for any table with more than
    :data:`MAX_SMALL_COLUMNS` columns, which is where a ``\\small`` table stops fitting an IEEE
    column. ``rotate_headers`` turns every heading but the first through 90 degrees, for
    tables whose cells are a few digits wide and whose headings would otherwise set the width.
    Values in a ``variant`` column are rendered through :func:`display_name`.
    """
    environment = "table*" if wide else "table"
    if fontsize is None:
        fontsize = r"\footnotesize" if wide or len(columns) > MAX_SMALL_COLUMNS else r"\small"
    titles = [title for _, title in columns]
    if rotate_headers:
        titles = [titles[0]] + [f"\\rotatebox{{90}}{{{title}}}" for title in titles[1:]]
    header = " & ".join(titles) + r" \\"
    # ``@{}`` drops the padding outside the first and last column: it is wasted width in a
    # column that is already tight, and it lets the rules align with the caption's margin.
    alignment = "@{}l" + "r" * (len(columns) - 1) + "@{}"
    lines = [
        provenance.latex_comment(),
        f"\\begin{{{environment}}}[t]",
        r"\centering",
        fontsize,
        r"\setlength{\tabcolsep}{2pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        header,
        r"\hline",
    ]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if key == "variant" and isinstance(value, str):
                value = display_name(value)
            cells.append(_fmt(value, digits))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    if note:
        lines.append(
            f"\\\\[2pt]\n\\begin{{minipage}}{{\\linewidth}}\\footnotesize {note}\\end{{minipage}}"
        )
    lines.append(f"\\end{{{environment}}}")
    return "\n".join(lines) + "\n"


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _pending(path: Path, caption: str, label: str, reason: str) -> TableResult:
    _write(path, PENDING_TEMPLATE.format(caption=caption, label=label, reason=reason))
    return TableResult(path.stem, path, True, reason)


def table_main_results(
    metrics: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> TableResult:
    path = output_dir / "table1_main_results.tex"
    caption = "Main benchmark results at the full store, by backend."
    label = "tab:main-results"
    frame = metrics.get("task_utility", pd.DataFrame())
    if frame.empty:
        return _pending(path, caption, label, "no graded answers were found under results/raw")
    final = frame[(frame["checkpoint"] == frame["checkpoint"].max()) & (frame["fault"] == "none")]
    if final.empty:
        final = frame[frame["checkpoint"] == frame["checkpoint"].max()]
    quality = metrics.get("memory_quality", pd.DataFrame())
    if not quality.empty:
        final = final.merge(quality, on=["variant", "fault", "checkpoint"], how="left")
    final = final.sort_values("accuracy", ascending=False)
    # Accuracy and its interval are one measurement, and three columns for it is what pushed
    # the table past the column edge.
    final = final.assign(accuracy_with_ci=final.apply(_accuracy_with_ci, axis=1))
    columns = [("variant", "Backend")]
    if "n" in final.columns and final["n"].notna().any():
        columns.append(("n", "$n$"))
    columns.extend(
        [
            ("accuracy_with_ci", "Accuracy [95\\% CI]"),
            ("evidence_recall", _stack("Ev.", "recall")),
            ("evidence_precision", _stack("Ev.", "prec.")),
            ("abstention_precision", _stack("Abst.", "prec.")),
        ]
    )
    note = (
        "Accuracy counts a correct answer and a correct abstention alike. Brackets are Wilson "
        "score intervals at 95\\%. Results use the deterministic oracle reader, so they measure "
        "the memory layer and are an upper bound on end-to-end agent accuracy."
    )
    return TableResult(
        path.stem,
        _write(path, _tabular(final, columns, caption, label, provenance, note=note)),
        False,
    )


def _accuracy_with_ci(row: pd.Series) -> str:
    accuracy = row.get("accuracy")
    low = row.get("accuracy_ci_low")
    high = row.get("accuracy_ci_high")
    if accuracy is None or (isinstance(accuracy, float) and math.isnan(accuracy)):
        return "--"
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return f"{float(accuracy):.3f}"
    return f"{float(accuracy):.3f} [{float(low):.3f}, {float(high):.3f}]"


#: Column headings that fit a two-column page; the full contract names are in the caption text.
CONTRACT_SHORT_NAMES: dict[str, str] = {
    "duplicate_replay_invariance": "Dup.",
    "restart_equivalence": "Restart",
    "supersession_correctness": "Supers.",
    "deletion_closure": "Deletion",
    "tenant_isolation": "Tenant",
    "evidence_budget_compliance": "Budget",
}


def table_contract_violations(
    frames: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> TableResult:
    path = output_dir / "table2_contract_violations.tex"
    caption = "Contract violations by condition and contract at 40 sessions."
    label = "tab:contract-violations"
    violations = frames.get("violations", pd.DataFrame())
    contracts = frames.get("contracts", pd.DataFrame())
    if contracts.empty:
        return _pending(path, caption, label, "no contract results were found under results/raw")
    # Group by *variant*, not backend: an ablated guarded condition shares the guarded backend
    # name, and attributing its violations to the un-ablated system would be wrong.
    key = "variant" if "variant" in violations.columns else "backend_name"
    variants = (
        sorted(contracts["variant"].dropna().unique())
        if "variant" in contracts.columns
        else sorted(contracts["backend_name"].unique())
    )
    counts = (
        violations.groupby([key, "contract_name"]).size().unstack(fill_value=0)
        if not violations.empty
        else pd.DataFrame(index=variants)
    )
    for contract_name in sorted(contracts["contract_name"].unique()):
        if contract_name not in counts.columns:
            counts[contract_name] = 0
    counts = counts.reindex(variants, fill_value=0)
    counts = counts[sorted(counts.columns)]
    # Each condition runs the suite once per (fault, seed) on the same fixed scenario. A cell is
    # the count for one run, not the sum over runs: summing would only multiply by the number
    # of seeds and present one repeated observation as many.
    group_key = "variant" if "variant" in contracts.columns else "backend_name"
    suite_runs = contracts.groupby(group_key).size() // max(1, contracts["contract_name"].nunique())
    per_run = suite_runs.reindex(counts.index).fillna(1).clip(lower=1)
    counts = counts.div(per_run, axis=0).round().astype(int)
    uniform = True
    if not violations.empty and {"fault", "seed"} <= set(violations.columns):
        per_run_counts = violations.groupby([key, "contract_name", "fault", "seed"]).size()
        uniform = bool(per_run_counts.groupby(level=[0, 1]).nunique().eq(1).all())
    display = counts.reset_index().rename(columns={"index": key, key: "variant"})
    columns = [("variant", "Condition")] + [
        (column, CONTRACT_SHORT_NAMES.get(column, column.replace("_", "~")))
        for column in counts.columns
    ]
    runs_text = (
        f"{int(suite_runs.iloc[0])} times (fault settings $\\times$ seeds)"
        if not suite_runs.empty and suite_runs.nunique() == 1
        else "once per fault setting and seed"
    )
    vacuous = (
        int(contracts["vacuous"].fillna(False).astype(bool).sum())
        if "vacuous" in contracts.columns
        else 0
    )
    note = (
        "A cell is the number of violations that contract raised in one run of the suite. "
        f"Each condition ran the suite {runs_text}"
        + (
            " on the same fixed scenario, and every run gave the same count"
            if uniform
            else ", and the cell is the mean over those runs"
        )
        + ". Zero means the contract was evaluated and held"
        + (
            f"; {vacuous} evaluations whose precondition the scenario could not establish (for "
            "example, a budget on a store that keeps nothing) are vacuous and counted in no cell"
            if vacuous
            else ""
        )
        + ". Contracts: duplicate-replay invariance, restart equivalence, supersession "
        "correctness, deletion closure, tenant isolation, evidence-budget compliance."
    )
    return TableResult(
        path.stem,
        _write(
            path,
            _tabular(
                display,
                columns,
                caption,
                label,
                provenance,
                digits=0,
                note=note,
                rotate_headers=True,
            ),
        ),
        False,
    )


def table_fault_recovery(
    metrics: dict[str, pd.DataFrame],
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    provenance: Provenance,
) -> TableResult:
    path = output_dir / "table3_fault_recovery.tex"
    caption = "Fault recovery by backend: fraction of injections that recovered as expected."
    label = "tab:fault-recovery"
    faults = frames.get("faults", pd.DataFrame())
    if faults.empty:
        return _pending(
            path,
            caption,
            label,
            "no fault injections were recorded; run a configuration whose faults list is not "
            "only 'none'",
        )
    pivot = faults.groupby(["variant", "fault_name"])["recovered"].mean().unstack()
    display = pivot.reset_index()
    columns = [("variant", "Backend")] + [
        (column, FAULT_SHORT_NAMES.get(column, _escape(column.replace("_", " "))))
        for column in pivot.columns
    ]
    note = (
        "1.00 means every injection of that fault recovered as the fault's declared expected "
        "behaviour requires. A backend that has no such state to damage records recovery "
        "vacuously; the observed-behaviour string in faults.jsonl distinguishes the two. "
        "Faults: duplicate event delivery, out-of-order delivery, interrupted write, index "
        "loss, contradictory update, deletion during store growth."
    )
    return TableResult(
        path.stem,
        _write(path, _tabular(display, columns, caption, label, provenance, note=note, wide=True)),
        False,
    )


#: Column headings for the fault table; the full fault names are spelled out in its note.
FAULT_SHORT_NAMES: dict[str, str] = {
    "duplicate_event_delivery": "Dup.\\ deliv.",
    "out_of_order_delivery": "Out of order",
    "interrupted_write": "Interr.\\ write",
    "index_loss": "Index loss",
    "contradictory_update": "Contradict.",
    "deletion_during_growth": "Del.\\ + growth",
}


def table_longitudinal(
    metrics: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> TableResult:
    path = output_dir / "table4_longitudinal.tex"
    caption = "Accuracy and stale-memory error rate at each longitudinal checkpoint."
    label = "tab:longitudinal"
    frame = metrics.get("task_utility", pd.DataFrame())
    if frame.empty:
        return _pending(path, caption, label, "no graded answers were found under results/raw")
    data = frame[frame["fault"] == "none"] if "fault" in frame.columns else frame
    if data.empty:
        data = frame
    pivot = data.pivot_table(index="variant", columns="checkpoint", values="accuracy")
    display = pivot.reset_index()
    # The checkpoint id is zero-padded so it sorts; the padding is noise in a heading.
    columns = [("variant", "Backend")] + [
        (column, column.replace("cp", "").lstrip("0") + "\\%") for column in pivot.columns
    ]
    note = (
        "Each column is a fraction of the session stream ingested. Only the no-fault condition "
        "is shown; the fault conditions appear in Table~\\ref{tab:fault-recovery}."
    )
    return TableResult(
        path.stem,
        _write(path, _tabular(display, columns, caption, label, provenance, note=note)),
        False,
    )


def table_efficiency(
    metrics: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> TableResult:
    path = output_dir / "table5_efficiency.tex"
    caption = "Efficiency and storage at the full store, no fault, means over seeds."
    label = "tab:efficiency"
    efficiency = metrics.get("efficiency", pd.DataFrame())
    storage = metrics.get("storage_growth", pd.DataFrame())
    if efficiency.empty:
        return _pending(path, caption, label, "no efficiency measurements were found")
    final = efficiency[efficiency["checkpoint"] == efficiency["checkpoint"].max()]
    if not storage.empty:
        final = final.merge(storage, on=["variant", "fault", "checkpoint"], how="left")
    if "fault" in final.columns:
        no_fault = final[final["fault"] == "none"]
        final = no_fault if not no_fault.empty else final
    # Byte counts in the hundreds of thousands are unreadable at a glance and twice the width
    # of the same number in KiB; one decimal keeps the index sizes distinguishable.
    for source, target in (
        ("database_size_bytes", "database_size_kib"),
        ("index_size_bytes", "index_size_kib"),
    ):
        if source in final.columns:
            final = final.assign(**{target: final[source].map(_kib)})
    columns = [
        ("variant", "Backend"),
        ("injected_tokens_mean", _stack("Inj.", "tokens")),
        ("latency_p95_ms_retrieval", _stack("p95 retr.", "(ms)")),
        ("latency_p95_ms_ingestion", _stack("p95 ingest", "(ms)")),
        ("database_size_kib", _stack("DB", "(KiB)")),
        ("index_size_kib", _stack("Index", "(KiB)")),
    ]
    note = (
        "Token counts use the shared approximate counter recorded in each manifest, so budgets "
        "are comparable across backends. Latencies are 95th percentiles over the run; sizes are "
        "on-disk bytes divided by 1024. Monetary cost is zero for every mock-provider run and is "
        "reported with the paid LongMemEval runs."
    )
    return TableResult(
        path.stem,
        _write(path, _tabular(final, columns, caption, label, provenance, digits=2, note=note)),
        False,
    )


def _kib(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value) / 1024:.1f}"


def table_ablations(
    metrics: dict[str, pd.DataFrame], output_dir: Path, provenance: Provenance
) -> TableResult:
    path = output_dir / "table6_ablations.tex"
    caption = "Ablations: each row disables exactly one guard, all else held constant."
    label = "tab:ablations"
    frame = metrics.get("task_utility", pd.DataFrame())
    if frame.empty:
        return _pending(path, caption, label, "no graded answers were found under results/raw")
    final = frame[frame["checkpoint"] == frame["checkpoint"].max()]
    # One row per condition: the fault matrix would otherwise repeat every ablation once per
    # fault, and the unlabelled repeats read as duplicated rows.
    if "fault" in final.columns:
        unfaulted = final[final["fault"] == "none"]
        final = unfaulted if not unfaulted.empty else final
    ablated = final[final["variant"].str.contains(r"\+", regex=True, na=False)]
    baseline = final[final["variant"] == "guarded_memory"]
    if ablated.empty:
        return _pending(
            path,
            caption,
            label,
            "the pooled runs declare no ablations; add them to the configuration's `ablations` "
            "list and re-run",
        )
    display = pd.concat([baseline, ablated]).sort_values("variant")
    columns = [("variant", "Condition")]
    if "n" in display.columns and display["n"].notna().any():
        columns.append(("n", "$n$"))
    columns.extend(
        [
            ("accuracy", "Acc."),
            ("stale_memory_error_rate", "Stale"),
            ("deletion_leakage_rate", "Leaked"),
            ("unsupported_answer_rate", _stack("Unsup-", "ported")),
            ("wrong_abstention_rate", _stack("Wrong", "abst.")),
        ]
    )
    note = (
        "Each ablated condition differs from guarded memory in exactly one guard; "
        "the fairness validator rejects any run where more than one differs. Acc.\\ is task "
        "accuracy; the remaining columns are error rates at the full store."
    )
    return TableResult(
        path.stem,
        _write(path, _tabular(display, columns, caption, label, provenance, note=note)),
        False,
    )


def generate_all(
    frames: dict[str, pd.DataFrame],
    metrics: dict[str, pd.DataFrame],
    output_dir: Path,
    provenance: Provenance,
) -> list[TableResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        table_main_results(metrics, output_dir, provenance),
        table_contract_violations(frames, output_dir, provenance),
        table_fault_recovery(metrics, frames, output_dir, provenance),
        table_longitudinal(metrics, output_dir, provenance),
        table_efficiency(metrics, output_dir, provenance),
        table_ablations(metrics, output_dir, provenance),
    ]
