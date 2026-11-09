"""Cross-scale analysis: how a benchmark's verdict depends on how big the store was.

Everything else in the reporting layer works *within* a comparability group, because pooling runs
whose workload parameters differ would average incomparable numbers. This module deliberately
works *across* groups, which is legitimate only because it never pools them: each store size
keeps its own accuracy, its own paired tests, and its own violation counts, and what is compared
is the verdict each size produces.

Two questions are answered here.

**Does store size decide the verdict?** For every store size, each condition is tested against
the reference on the same query set with McNemar, Holm-adjusted within that size. A benchmark
that reports "no difference" at one size and "large difference" at another is not measuring a
property of the systems; it is measuring itself.

**Do contract violations lead accuracy?** For each condition we record the store size at which
its accuracy first separates from the reference, and the violations it had already accumulated at
the *smallest* size. A condition that violated contracts long before accuracy noticed is the
evidence for treating contracts as an early warning; a condition that never separates in accuracy
despite violating throughout is evidence that accuracy cannot see the property at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd

from memcheck.evaluation.runner import discover_runs, load_manifest, verify_integrity
from memcheck.reporting import aggregate
from memcheck.reporting.metrics import collapse_replicas, unique_outcomes
from memcheck.reporting.provenance import Provenance
from memcheck.utilities.statistics import (
    holm_bonferroni,
    mcnemar,
    paired_difference_interval,
    stream_level_test,
    wilson_interval,
)

DEFAULT_REFERENCE = "guarded_memory"
DEFAULT_BENCHMARK = "synthetic_longitudinal"
#: A separation must also hold at the level of independent streams (exact sign-flip test).
STREAM_ALPHA = 0.05

#: The workload summary is embedded in the manifest's notes field as a JSON object.
_WORKLOAD_PAYLOAD = re.compile(r"workload=(\{.*\})\s*$")

#: Printed instead of a store size when a condition never separates from the reference.
NEVER = "never"

#: Workload keys that are *expected* to vary across a size sweep, because they are the size or
#: are determined by it. Everything else must match, or the runs are not points on one curve.
SIZE_DEPENDENT_KEYS = frozenset({"sessions", "event_count", "query_count", "seed"})


@dataclass(frozen=True)
class SweepPoint:
    """One candidate point on the curve, before the sweep family has been chosen."""

    run_dir: Path
    sessions: int
    signature: str
    summary: dict[str, object]


@dataclass
class ScaleAnalysis:
    accuracy: pd.DataFrame
    leading_indicator: pd.DataFrame
    problems: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    #: One row per (store size, condition): the contract violations that condition raised at
    #: that size. Zero rows are kept, because "the guarded layer raised none" is a result.
    violations: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: One row per (store size, condition): the paired difference to the reference computed per
    #: independent stream, with the cluster-bootstrap interval and exact sign-flip p-value.
    streams: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class SweepSeries:
    """How one condition is drawn in the sweep figure."""

    variant: str
    label: str
    short_label: str
    color: str


#: The conditions the sweep figure shows, in drawing order. The two memory-less baselines sit
#: below 0.3 accuracy at every size and are left to the table: drawing them would squash the
#: axis so far that the inversion between the four conditions that matter becomes invisible.
#: Conditions are told apart by colour and by the name written at the end of each line, not by
#: marker shapes or dash patterns, which read as chart furniture at column width.
SWEEP_SERIES: tuple[SweepSeries, ...] = (
    SweepSeries("guarded_memory", "guarded memory", "guarded memory", "#009E73"),
    SweepSeries("transcript_rag", "transcript chunking", "transcript chunking", "#0072B2"),
    SweepSeries(
        "guarded_memory+abl_no_supersession",
        "guarded, no supersession",
        "no supersession",
        "#D55E00",
    ),
    SweepSeries(
        "guarded_memory+abl_no_tombstones", "guarded, no tombstones", "no tombstones", "#CC79A7"
    ),
)

SWEEP_FIGURE_NAME = "fig_scale_sweep"
SWEEP_WORKBOOK_NAME = "figure_data_scale.xlsx"


def workload_size(manifest_notes: str) -> dict[str, object]:
    """Recover the workload summary the runner recorded in the manifest.

    Read from the manifest rather than the config because the manifest is checksummed with the
    run and records what the workload *actually* produced, not what was requested.
    """
    match = _WORKLOAD_PAYLOAD.search(manifest_notes or "")
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def sweep_signature(summary: dict[str, object]) -> str:
    """Everything about a workload that must be constant across a size sweep.

    A sweep is only interpretable if store size is the *only* thing that changed. Two runs whose
    failure densities or tenant counts differ are two experiments, not two points on one curve,
    and silently plotting them together would attribute a parameter change to scale.
    """
    comparable = {
        key: value for key, value in sorted(summary.items()) if key not in SIZE_DEPENDENT_KEYS
    }
    # The tenant list grows with the tenant count, so compare its size rather than its members.
    tenants = comparable.pop("tenants", None)
    if isinstance(tenants, list):
        comparable["tenant_count"] = len(tenants)
    return json.dumps(comparable, sort_keys=True, default=str)


def _final_checkpoint_grades(grades: pd.DataFrame) -> pd.DataFrame:
    unfaulted = grades[grades["fault"] == "none"] if "fault" in grades.columns else grades
    if unfaulted.empty:
        unfaulted = grades
    final: pd.DataFrame = unfaulted[unfaulted["checkpoint"] == unfaulted["checkpoint"].max()]
    return final


def _outcomes_by_variant(grades: pd.DataFrame) -> dict[str, pd.Series]:
    keys = ["query_id", "seed"] if "seed" in grades.columns else ["query_id"]
    return {
        str(variant): unique_outcomes(group, keys, str(variant))
        for variant, group in grades.groupby("variant", dropna=False)
    }


def _stream_differences(reference: pd.Series, variant: pd.Series) -> list[float]:
    """Paired accuracy difference (reference minus variant) per independent stream.

    The stream is the seed: every query under one seed was answered from one store, so queries
    are clustered within it. Without a seed level there is one stream and one difference.
    """
    if "seed" not in (reference.index.names or []):
        return [float(reference.mean() - variant.mean())]
    frame = pd.DataFrame({"reference": reference.astype(float), "variant": variant.astype(float)})
    by_seed = frame.groupby(level="seed").mean()
    return [
        float(ref - var) for ref, var in zip(by_seed["reference"], by_seed["variant"], strict=True)
    ]


def _apply_verdicts(at_this_size: list[dict[str, object]], reference: str) -> None:
    """Set the Holm-adjusted p and the separation verdicts on each comparison row, in place.

    ``separated_pooled`` is the per-query McNemar verdict after Holm within the size.
    ``separated_from_reference``, the verdict every table uses, additionally requires the
    stream-level sign-flip test, so a difference carried by a handful of queries in a few
    streams cannot count as a detection.
    """
    comparisons = [row for row in at_this_size if row["variant"] != reference]
    adjusted = holm_bonferroni([float(cast(float, row["p_value"])) for row in comparisons])
    for row, p_adjusted in zip(comparisons, adjusted, strict=True):
        row["p_value_holm"] = p_adjusted
        row["separated_pooled"] = p_adjusted < 0.05
        row["separated_from_reference"] = bool(
            p_adjusted < 0.05 and float(cast(float, row["sign_flip_p"])) <= STREAM_ALPHA
        )
    for row in at_this_size:
        if row["variant"] == reference:
            row["p_value_holm"] = float("nan")
            row["separated_pooled"] = False
            row["separated_from_reference"] = False


def analyse(
    results_root: Path,
    *,
    benchmark: str = DEFAULT_BENCHMARK,
    reference: str = DEFAULT_REFERENCE,
    run_dirs: list[Path] | None = None,
    workload_version: str | None = None,
) -> ScaleAnalysis:
    """Compare the verdict each store size produces, for one benchmark and one reference.

    When ``workload_version`` is given, only sweeps of that version compete; two versions of
    one sweep cover the same sizes, so otherwise the choice would fall to a tie-break.
    """
    runs = run_dirs if run_dirs is not None else discover_runs(results_root)
    problems: list[str] = []
    rows: list[dict[str, object]] = []
    stream_rows: list[dict[str, object]] = []
    violations_by_size: dict[int, pd.Series] = {}
    used_runs: list[str] = []

    # Collect the candidates first, so the sweep family can be chosen before anything is
    # computed. A curve assembled from runs that differ in more than size would report a
    # parameter change as a scale effect.
    candidates: list[SweepPoint] = []
    for run_dir in runs:
        manifest = load_manifest(run_dir)
        if manifest.benchmark_name != benchmark:
            continue
        ok, integrity = verify_integrity(run_dir)
        if not ok:
            problems.extend(integrity)
            continue
        summary = workload_size(manifest.notes)
        sessions = summary.get("sessions")
        if not isinstance(sessions, int):
            problems.append(f"{run_dir.name}: manifest records no session count to scale against")
            continue
        candidates.append(
            SweepPoint(
                run_dir=run_dir,
                sessions=sessions,
                # The condition set belongs in the signature as much as the workload does: a run
                # carrying extra ablations or extra faults is a different experiment, and mixing
                # it in would leave conditions present at only one point of the curve.
                signature=sweep_signature(summary)
                + json.dumps(
                    {
                        "ablations": sorted(manifest.ablations),
                        "faults": sorted(manifest.faults),
                        "contract_store": manifest.contract_store,
                    },
                    sort_keys=True,
                ),
                summary=summary,
            )
        )

    if not candidates:
        return ScaleAnalysis(pd.DataFrame(), pd.DataFrame(), problems=problems, run_ids=[])

    # The family covering the most distinct sizes is the sweep; ties go to the one reaching
    # furthest, because a wider sweep answers the question better.
    families: dict[str, list[SweepPoint]] = {}
    for candidate in candidates:
        families.setdefault(candidate.signature, []).append(candidate)
    eligible = [
        key
        for key, family in families.items()
        if workload_version is None or family[0].summary.get("version") == workload_version
    ]
    if not eligible:
        problems.append(f"no sweep of workload version {workload_version!r}")
        return ScaleAnalysis(pd.DataFrame(), pd.DataFrame(), problems=problems, run_ids=[])
    chosen_key = max(
        eligible,
        key=lambda key: (
            len({point.sessions for point in families[key]}),
            max(point.sessions for point in families[key]),
        ),
    )
    for key, family in families.items():
        if key == chosen_key:
            continue
        names = ", ".join(sorted(point.run_dir.name for point in family))
        problems.append(
            f"excluded {len(family)} run(s) that differ from the sweep in more than store size "
            f"({names}): a size sweep is only interpretable when size is the only change"
        )

    # Two runs at the same size within one family are two samples, not two points; keeping the
    # first by name makes the selection deterministic rather than filesystem-ordered.
    by_size: dict[int, SweepPoint] = {}
    for candidate in sorted(families[chosen_key], key=lambda point: point.run_dir.name):
        if candidate.sessions in by_size:
            problems.append(
                f"{candidate.run_dir.name}: the sweep already has a run at {candidate.sessions} "
                "sessions; using the first by name"
            )
            continue
        by_size[candidate.sessions] = candidate

    for point in sorted(by_size.values(), key=lambda item: item.sessions):
        run_dir, sessions, summary = point.run_dir, point.sessions, point.summary
        frames = aggregate.load_all(results_root, run_dirs=[run_dir])
        grades = frames.get("grades", pd.DataFrame())
        if grades.empty:
            problems.append(f"{run_dir.name}: no grades")
            continue
        final = _final_checkpoint_grades(grades)
        outcomes, collapsed = collapse_replicas(_outcomes_by_variant(final))
        if collapsed:
            problems.append(
                f"{run_dir.name}: seeds reproduce each other exactly; tested on unique queries"
            )
        if reference not in outcomes:
            problems.append(f"{run_dir.name}: reference condition {reference!r} is absent")
            continue
        used_runs.append(run_dir.name)

        violations = frames.get("violations", pd.DataFrame())
        # Per run of the suite, not summed over seeds: a contract builds its own fixed scenario,
        # so every seed repeats the same evaluation and a sum would only count the seeds.
        evaluations = (
            final.groupby("variant")["seed"].nunique()
            if "seed" in final.columns
            else pd.Series(1, index=final["variant"].unique())
        )
        violations_by_size[sessions] = (
            (
                violations.groupby("variant").size()
                / evaluations.reindex(violations["variant"].unique()).fillna(1)
            )
            .round()
            .astype(int)
            if not violations.empty
            else pd.Series(dtype=int)
        )

        # Every comparison at this store size is one family, so the multiplicity correction is
        # applied within the size rather than across the whole sweep.
        at_this_size: list[dict[str, object]] = []
        baseline = outcomes[reference]
        for variant, series in sorted(outcomes.items()):
            shared = baseline.index.intersection(series.index)
            paired_reference = baseline.reindex(shared)
            paired_variant = series.reindex(shared)
            correct = int(paired_variant.sum())
            low, high = wilson_interval(correct, len(shared))
            test = mcnemar(list(paired_reference), list(paired_variant))
            _, gap_low, gap_high = paired_difference_interval(
                list(paired_reference), list(paired_variant)
            )
            stream = stream_level_test(_stream_differences(paired_reference, paired_variant))
            if variant != reference:
                stream_rows.append(
                    {
                        "sessions": sessions,
                        "run_id": run_dir.name,
                        "variant": variant,
                        "n_streams": stream.n_streams,
                        "mean_diff": stream.mean,
                        "ci_low": stream.ci_low,
                        "ci_high": stream.ci_high,
                        "sign_flip_p": stream.p_value,
                        "min_diff": stream.minimum,
                        "max_diff": stream.maximum,
                        "streams_favouring_reference": stream.favouring_reference,
                    }
                )
            at_this_size.append(
                {
                    "sessions": sessions,
                    "run_id": run_dir.name,
                    "event_count": summary.get("event_count"),
                    "variant": variant,
                    "n": len(shared),
                    "accuracy": correct / len(shared) if len(shared) else float("nan"),
                    "accuracy_ci_low": low,
                    "accuracy_ci_high": high,
                    "reference_accuracy": float(paired_reference.mean()),
                    "gap_vs_reference": float(paired_reference.mean() - paired_variant.mean()),
                    # Newcombe interval on the paired gap: a non-significant size is reported
                    # as the range of gaps it still allows, not as parity.
                    "gap_ci_low": gap_low,
                    "gap_ci_high": gap_high,
                    # Only discordant pairs carry evidence, so this is the real sample size of
                    # the comparison. A "significant" verdict resting on six of them deserves to
                    # be read differently from one resting on two hundred.
                    "discordant_pairs": int((paired_reference != paired_variant).sum()),
                    "p_value": test.p_value,
                    "cohens_g": test.effect_size,
                    "effect_label": test.effect_label,
                    "warnings": "; ".join(test.warnings),
                    # The same comparison at the level of independent streams.
                    "n_streams": stream.n_streams,
                    "stream_ci_low": stream.ci_low,
                    "stream_ci_high": stream.ci_high,
                    "sign_flip_p": stream.p_value,
                }
            )
        # The reference against itself is a tautology, not a comparison, and would waste an
        # element of the correction's family.
        _apply_verdicts(at_this_size, reference)
        rows.extend(at_this_size)

    accuracy = pd.DataFrame(rows).sort_values(["variant", "sessions"]) if rows else pd.DataFrame()
    streams = (
        pd.DataFrame(stream_rows).sort_values(["variant", "sessions"])
        if stream_rows
        else pd.DataFrame()
    )
    return ScaleAnalysis(
        accuracy=accuracy,
        leading_indicator=_leading_indicator(accuracy, violations_by_size, reference),
        problems=problems,
        run_ids=sorted(used_runs),
        violations=_violations_frame(accuracy, violations_by_size),
        streams=streams,
    )


def _violations_frame(
    accuracy: pd.DataFrame, violations_by_size: dict[int, pd.Series]
) -> pd.DataFrame:
    """Contract violations per (store size, condition), for every condition that was graded.

    Built from the graded conditions rather than from the violation log alone so that a
    condition with no violations appears with a zero instead of disappearing.
    """
    if accuracy.empty:
        return pd.DataFrame(columns=["sessions", "variant", "violations"])
    rows = [
        {
            "sessions": int(sessions),
            "variant": str(variant),
            "violations": int(
                violations_by_size.get(int(sessions), pd.Series(dtype=int)).get(str(variant), 0)
            ),
        }
        for variant, sessions in sorted(
            {
                (str(v), int(s))
                for v, s in zip(accuracy["variant"], accuracy["sessions"], strict=True)
            }
        )
    ]
    return pd.DataFrame(rows, columns=["sessions", "variant", "violations"])


def write_latex(analysis: ScaleAnalysis, output_dir: Path, provenance: Provenance) -> list[Path]:
    """Emit the two cross-scale tables the paper cites.

    Written from here rather than from ``tables.py`` because these are the only tables spanning
    comparability groups, and building them alongside the per-group ones would invite exactly the
    pooling the rest of the pipeline exists to prevent.
    """
    from memcheck.reporting.tables import _stack, _tabular, _write

    written: list[Path] = []
    if analysis.accuracy.empty:
        return written
    sizes = sorted(analysis.accuracy["sessions"].unique())

    # Table 1: accuracy by store size, one column per size, significance marked.
    rows: list[dict[str, object]] = []
    for variant, group in analysis.accuracy.groupby("variant"):
        by_size = group.set_index("sessions")
        row: dict[str, object] = {"variant": variant}
        for size in sizes:
            if size not in by_size.index:
                row[f"s{size}"] = None
                continue
            record = by_size.loc[size]
            # The star overhangs into the column gap rather than widening the column, so a
            # significant cell and a plain one keep the same width and the digits stay aligned.
            marker = r"\rlap{$^{*}$}" if bool(record["separated_from_reference"]) else ""
            row[f"s{size}"] = f"{float(record['accuracy']):.3f}{marker}"
        rows.append(row)
    columns = [("variant", "condition")] + [(f"s{size}", f"{size} sess.") for size in sizes]
    streams_text = (
        f"the {int(analysis.streams['n_streams'].max())} independent streams"
        if not analysis.streams.empty and "n_streams" in analysis.streams.columns
        else "the independent streams"
    )
    written.append(
        _write(
            output_dir / "table7_scale_verdict.tex",
            _tabular(
                pd.DataFrame(rows),
                columns,
                caption=(
                    "Task accuracy at the full store, by store size, at constant failure "
                    "density, pooled over ten independent seeds."
                ),
                label="tab:scale-verdict",
                provenance=provenance,
                note=(
                    r"$^{*}$ differs from guarded memory on the same query set by both tests: "
                    r"McNemar on the pooled pairs, Holm-adjusted within each store size, "
                    r"$p < 0.05$, and an exact sign-flip test on the paired difference across "
                    f"{streams_text}, $p \\le {STREAM_ALPHA}$."
                ),
                fontsize=r"\footnotesize",
            ),
        )
    )

    # Table 2: when contracts fired against when accuracy noticed.
    largest = int(sizes[-1])
    leading = analysis.leading_indicator.copy()
    if not leading.empty:
        leading["first_separating_sessions"] = [
            NEVER if first == NEVER else int(first)
            for first in leading["first_separating_sessions"]
        ]
        leading["separation_is_stable"] = leading["separation_is_stable"].map(
            lambda value: "yes" if bool(value) else "no"
        )
    written.append(
        _write(
            output_dir / "table8_leading_indicator.tex",
            _tabular(
                leading,
                [
                    ("variant", "condition"),
                    ("violations_at_smallest_store", "viol."),
                    ("first_separating_sessions", _stack("first", "sep.")),
                    ("discordant_pairs_at_first_separation", _stack("disc.", "pairs")),
                    ("separation_is_stable", "stable"),
                    ("gap_at_largest_store", _stack("gap", f"@{largest}")),
                ],
                caption=(
                    "Contract violations against the store size at which task accuracy first "
                    "separates from the guarded store. ``viol.'' is the number of violations in "
                    "one run of the contract suite; contracts build their own fixed scenario, so "
                    "it is the same at every store size. ``first sep.'' is the smallest store (in "
                    f"sessions) at which accuracy differs; ``gap @{largest}'' is the accuracy gap "
                    f"at {largest} sessions."
                ),
                label="tab:leading-indicator",
                provenance=provenance,
                note=(
                    "A separation requires both tests of Table~\\ref{tab:scale-verdict}: the "
                    "pooled McNemar test and the sign-flip test across independent streams. "
                    "``disc.\\ pairs'' is the number of queries the two conditions disagreed "
                    "on at the separating size, the comparison's effective sample size. "
                    "``stable'' is no if a separation disappears at a larger size and then "
                    "returns. A separation can favour either side; Fig.~\\ref{fig:scale-sweep} "
                    "shows which."
                ),
                fontsize=r"\footnotesize",
            ),
        )
    )
    return written


def write_figure(analysis: ScaleAnalysis, output_dir: Path, provenance: Provenance) -> Path | None:
    """Draw the paired accuracy gap to the reference against store size.

    Returns the ``.png`` path (the ``.pdf`` is written beside it), or ``None`` when there is
    nothing to draw. One panel: a solid line per condition in :data:`SWEEP_SERIES` (the
    reference itself is the zero line), its 95% Newcombe interval on the paired difference as a
    translucent band, and the condition's name at the end of the line. A band that straddles
    zero is a size at which the benchmark cannot tell the two apart; its width is how large a
    difference that size still allows. Significance marks stay in the verdict table. The x axis
    is categorical so the five sizes are equally spaced.
    """
    import matplotlib.pyplot as plt

    from memcheck.reporting.figures import PALETTE, _finish, _style

    if analysis.accuracy.empty or "gap_ci_low" not in analysis.accuracy.columns:
        return None
    accuracy = analysis.accuracy
    present = [
        series
        for series in SWEEP_SERIES
        if series.variant in set(accuracy["variant"]) and series.variant != DEFAULT_REFERENCE
    ]
    if not present:
        return None
    sizes = sorted(int(size) for size in accuracy["sessions"].unique())
    position = {size: index for index, size in enumerate(sizes)}

    _style()
    figure, axis = plt.subplots(figsize=(3.4, 2.3))
    axis.axhline(0.0, color="#888888", linewidth=0.7, zorder=0)
    ends: list[tuple[float, str, str]] = []
    for series in present:
        color = PALETTE.get(series.variant, series.color)
        rows = accuracy[accuracy["variant"] == series.variant].sort_values("sessions")
        x = [position[int(size)] for size in rows["sessions"]]
        y = rows["gap_vs_reference"].to_numpy(dtype=float)
        low = rows["gap_ci_low"].to_numpy(dtype=float)
        high = rows["gap_ci_high"].to_numpy(dtype=float)
        axis.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0, zorder=1)
        axis.plot(x, y, color=color, linewidth=1.3, solid_capstyle="round", zorder=3)
        axis.plot(x, y, color=color, linestyle="none", marker="o", markersize=2.6, zorder=4)
        ends.append((float(y[-1]), series.short_label, color))

    # Names go at the end of their line. Where two lines end too close together the labels are
    # pushed apart and a thin leader joins each to its line; the data points never move.
    ends.sort()
    placed = [value for value, _, _ in ends]
    gap = 0.022
    for index in range(1, len(placed)):
        if placed[index] - placed[index - 1] < gap:
            placed[index] = placed[index - 1] + gap
    last = len(sizes) - 1
    texts = []
    for (actual, label, color), slot in zip(ends, placed, strict=True):
        if abs(slot - actual) > 0.004:
            axis.plot(
                [last + 0.05, last + 0.17],
                [actual, slot],
                color=color,
                linewidth=0.5,
                alpha=0.7,
                zorder=2,
            )
        texts.append(
            axis.text(last + 0.2, slot, label, fontsize=6.3, color=color, va="center", ha="left")
        )

    drawn = accuracy[accuracy["variant"].isin([series.variant for series in present])]
    axis.set_ylim(
        min(-0.05, float(drawn["gap_ci_low"].min()) - 0.01),
        float(drawn["gap_ci_high"].max()) + 0.02,
    )
    axis.set_ylabel("accuracy gap to guarded store")
    axis.set_xlabel("store size (sessions)")
    axis.set_xticks(list(position.values()))
    axis.set_xticklabels([str(size) for size in sizes])
    axis.grid(axis="x", visible=False)
    figure.subplots_adjust(left=0.16, right=0.995, top=0.97, bottom=0.18)
    # The x range is widened to end just past the longest label, measured after layout, so the
    # labels sit inside the axes and the saved figure stays at column width whatever they say.
    left = -0.3
    axis.set_xlim(left, last + 2.0)
    figure.canvas.draw()
    axis_width = axis.get_window_extent().width
    label_width = max(text.get_window_extent().width for text in texts) + 4.0
    # Text keeps its pixel width whatever the data range, so the range is solved for, not read.
    axis.set_xlim(left, left + (last + 0.2 - left) / (1.0 - label_width / axis_width))
    return _finish(figure, provenance, output_dir / f"{SWEEP_FIGURE_NAME}.png", pdf_footer=False)


def write_workbook(
    analysis: ScaleAnalysis, output_dir: Path, provenance: Provenance
) -> Path | None:
    """The numbers behind the sweep figure, so it can be re-plotted without re-running."""
    from memcheck.reporting.workbook import SheetResult, _autofit, _readme

    if analysis.accuracy.empty:
        return None
    accuracy = analysis.accuracy.rename(
        columns={"accuracy_ci_low": "ci_low", "accuracy_ci_high": "ci_high"}
    )[
        [
            "sessions",
            "variant",
            "n",
            "accuracy",
            "ci_low",
            "ci_high",
            "discordant_pairs",
            "p_value_holm",
            "separated_from_reference",
        ]
    ].sort_values(["variant", "sessions"])
    sheets = {"accuracy": accuracy, "violations": analysis.violations}
    if not analysis.streams.empty:
        sheets["streams"] = analysis.streams
    results = [SheetResult(name, len(frame), False) for name, frame in sheets.items()]
    output = output_dir / SWEEP_WORKBOOK_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    group = "store-size sweep (one group per size, never pooled)"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _readme(provenance, group, results).to_excel(writer, sheet_name="README", index=False)
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        _autofit(writer)
    provenance.write_sidecar(output)
    return output


def publish_figure(
    analysis: ScaleAnalysis, figures_dir: Path, generated_dir: Path, provenance: Provenance
) -> list[Path]:
    """Write the sweep figure and its workbook, then copy them beside the paper's figures."""
    sources: list[Path] = []
    figure = write_figure(analysis, figures_dir, provenance)
    if figure is not None:
        sources.extend([figure, figure.with_suffix(".pdf")])
    book = write_workbook(analysis, figures_dir, provenance)
    if book is not None:
        sources.append(book)
    target_dir = generated_dir / "figures"
    copies: list[Path] = []
    for source in sources:
        if not source.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        target.write_bytes(source.read_bytes())
        copies.append(target)
    return copies


def _leading_indicator(
    accuracy: pd.DataFrame, violations_by_size: dict[int, pd.Series], reference: str
) -> pd.DataFrame:
    """When contracts fired, against when accuracy noticed.

    ``first_separating_sessions`` is the smallest store at which the condition's accuracy differs
    from the reference after correction. ``NEVER`` means the sweep never made the difference
    visible in accuracy --- which, for a condition that violates contracts throughout, is the
    strongest form of the argument rather than a null result.
    """
    if accuracy.empty or not violations_by_size:
        return pd.DataFrame()
    smallest = min(violations_by_size)
    smallest_violations = violations_by_size[smallest]

    rows: list[dict[str, object]] = []
    for variant, group in accuracy.groupby("variant"):
        if variant == reference:
            continue
        ordered = group.sort_values("sessions")
        separating = ordered[ordered["separated_from_reference"].astype(bool)]["sessions"]
        first = int(separating.min()) if not separating.empty else None
        at_smallest = group[group["sessions"] == smallest]

        # A separation that appears, vanishes at a larger store, and reappears is not a
        # detection; it is noise crossing a threshold. Reporting `first_separating_sessions`
        # without this flag would let a comparison resting on a handful of discordant pairs
        # masquerade as an accuracy signal.
        after_first = (
            ordered[ordered["sessions"] >= first]["separated_from_reference"].astype(bool)
            if first is not None
            else pd.Series(dtype=bool)
        )
        stable = bool(after_first.all()) if first is not None else False
        discordant_at_first = (
            int(ordered.loc[ordered["sessions"] == first, "discordant_pairs"].iloc[0])
            if first is not None
            else 0
        )
        rows.append(
            {
                "variant": variant,
                "violations_at_smallest_store": int(smallest_violations.get(variant, 0)),
                "smallest_store_sessions": smallest,
                "gap_at_smallest_store": float(at_smallest["gap_vs_reference"].iloc[0])
                if not at_smallest.empty
                else float("nan"),
                "separated_at_smallest_store": bool(at_smallest["separated_from_reference"].iloc[0])
                if not at_smallest.empty
                else False,
                "first_separating_sessions": first if first is not None else NEVER,
                "discordant_pairs_at_first_separation": discordant_at_first,
                "separation_is_stable": stable,
                "largest_store_sessions": int(group["sessions"].max()),
                "gap_at_largest_store": float(
                    group.sort_values("sessions")["gap_vs_reference"].iloc[-1]
                ),
                # The claim this table exists to test: contracts fired before accuracy did.
                "contracts_led_accuracy": bool(
                    smallest_violations.get(variant, 0) > 0
                    and (
                        first is None
                        or (
                            not at_smallest.empty
                            and not at_smallest["separated_from_reference"].iloc[0]
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("variant") if rows else pd.DataFrame()
