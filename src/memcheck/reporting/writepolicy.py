"""Write-policy comparison: one benchmark, one reader, two extractors.

The reporting layer works *within* a comparability group, because pooling runs that differ in a
comparability field would average incomparable numbers. The extractor identity is such a field
(DECISIONS D-017): a run under a different write policy is not another sample of the same
experiment, so the rule-based and the LLM-extracted runs of LongMemEval land in two groups and
are never pooled. That is correct, and it means the question RQ1 asks --- what moves when *only*
the write policy changes --- can only be answered across the boundary.

Like the cross-scale analysis (D-019), this module is permitted to reach across two groups
because it never pools them. Nothing is averaged across the boundary: the same query, under the
same seed, is paired with itself across the two runs, and what is reported is the paired
difference (McNemar on the discordant pairs, Holm-adjusted within the family of conditions)
alongside each policy's own accuracy, injected tokens, cost, latency, and reader saturation.
Every other field the fairness validator compares --- model, temperature, prompts other than the
extraction prompt, budgets, retrieval mode, checkpoint schedule, seeds, query set, and condition
set --- must match, or the two runs are not a pair and are refused with a reason.

The three baselines carry no write policy, so they appear in both runs unchanged by the
manipulation. Their paired movement measures run-to-run drift rather than an effect, which is
exactly why the table prints them next to the guarded condition.
"""

from __future__ import annotations

import inspect
import json
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from memcheck.evaluation.fairness import COMPARABLE_FIELDS, GROUPING_FIELDS
from memcheck.evaluation.runner import discover_runs, load_manifest, verify_integrity
from memcheck.models import ExperimentRunManifest, utcnow
from memcheck.reporting import aggregate
from memcheck.reporting import metrics as metrics_module
from memcheck.reporting.figures import PALETTE, FigureResult, _color, _style
from memcheck.reporting.metrics import seeds_are_replicas, unique_outcomes
from memcheck.reporting.provenance import Provenance
from memcheck.reporting.scale import workload_size
from memcheck.reporting.tables import TableResult, _pending, _tabular, _write
from memcheck.reporting.workbook import SheetResult, WorkbookResult, _autofit, _readme
from memcheck.utilities.statistics import holm_bonferroni, mcnemar, wilson_interval

DEFAULT_BENCHMARK = "longmemeval"

#: The two write policies this analysis knows how to pair, by extractor-identity prefix.
RULE_POLICY = "rule"
LLM_POLICY = "llm"

#: Prompt hashes under this prefix belong to the extractor and are *expected* to differ between
#: the two runs; every other prompt must match or the pair is refused.
EXTRACTION_PROMPT_PREFIX = "extract"

#: Grades with no category in their metadata are still paired; they are grouped under this name
#: so the per-category breakdown adds up to the condition total.
UNCATEGORISED = "uncategorised"

#: Everything the pair must agree on. Derived from the fairness validator's two field lists so
#: that a field added there cannot silently be left out here; the extractor is the one declared
#: difference and seeds are added because pairing a query against another draw of it is wrong.
PAIRING_FIELDS: tuple[str, ...] = (
    *(
        name
        for name in dict.fromkeys((*GROUPING_FIELDS, *COMPARABLE_FIELDS))
        if name != "extractor_identity"
    ),
    "random_seeds",
)

CONDITION_LABELS: dict[str, str] = {
    "no_memory": "no memory",
    "full_history": "full history",
    "transcript_rag": "transcript chunking",
    "guarded_memory": "guarded memory",
}
#: The guarded condition first, because it is the one the manipulation acts on.
CONDITION_ORDER: tuple[str, ...] = ("guarded_memory", "transcript_rag", "full_history", "no_memory")
#: Conditions drawn in the per-category figure. The two weak baselines only show drift, which
#: the table prints; drawing them doubles the figure's height for no information.
FIGURE_CONDITIONS: tuple[str, ...] = ("guarded_memory", "transcript_rag")

CATEGORY_LABELS: dict[str, str] = {
    "knowledge_update": "knowledge update",
    "cross_session_reasoning": "cross-session reasoning",
    "single_session_extraction": "single-session extraction",
    "temporal_reasoning": "temporal reasoning",
    "abstention": "abstention",
}
CATEGORY_ORDER: tuple[str, ...] = tuple(CATEGORY_LABELS)


def condition_label(variant: str) -> str:
    return CONDITION_LABELS.get(variant, variant.replace("_", " "))


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category.replace("_", " "))


def policy_of(extractor_identity: str) -> str | None:
    """Which side of the comparison an extractor belongs to, or ``None`` if neither."""
    for policy in (RULE_POLICY, LLM_POLICY):
        if extractor_identity.startswith(policy):
            return policy
    return None


def pair_signature(manifest: ExperimentRunManifest) -> str:
    """Everything about a run that must be identical to its counterpart under the other policy.

    The extractor itself and its prompt are the declared difference. The query count is taken
    from the workload summary as well as from the parameters hash, so a 20-instance run and a
    100-instance run form two pairs even if a future hash collision made them look alike.
    """
    comparable: dict[str, object] = {}
    for name in PAIRING_FIELDS:
        value = getattr(manifest, name, None)
        comparable[name] = sorted(str(item) for item in value) if isinstance(value, list) else value
    comparable["prompt_hashes"] = {
        name: digest
        for name, digest in sorted(manifest.prompt_hashes.items())
        if not name.startswith(EXTRACTION_PROMPT_PREFIX)
    }
    comparable["query_count"] = workload_size(manifest.notes).get("query_count")
    return json.dumps(comparable, sort_keys=True, default=str)


@dataclass(frozen=True)
class Candidate:
    run_dir: Path
    policy: str
    extractor: str
    signature: str
    query_count: int
    seeds: int
    configuration_hash: str


@dataclass
class WritePolicyAnalysis:
    conditions: pd.DataFrame
    categories: pd.DataFrame
    saturation: pd.DataFrame
    problems: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    configuration_hashes: list[str] = field(default_factory=list)
    extractors: dict[str, str] = field(default_factory=dict)

    @property
    def paired(self) -> bool:
        return not self.conditions.empty

    @property
    def reason(self) -> str:
        """Why no pair could be reported, for the pending table and the report index."""
        if self.paired:
            return ""
        if not self.problems:
            return "no two runs differing only in the guarded backend's write policy were found"
        return "; ".join(self.problems)


#: Checkpoints of the query schedule are labelled ``cpNNN``. Other labels (``ingestion``) mark
#: work done between checkpoints, such as an LLM extractor's calls, and must not be mistaken for
#: the last checkpoint merely because they sort after it.
SCHEDULED_PREFIX = "cp"


def _unfaulted(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "fault" not in frame.columns:
        return frame
    data: pd.DataFrame = frame[frame["fault"] == "none"]
    return data if not data.empty else frame


def _final_unfaulted(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = _unfaulted(frame)
    labels = [str(label) for label in data["checkpoint"].dropna().unique()]
    scheduled = [label for label in labels if label.startswith(SCHEDULED_PREFIX)]
    last = max(scheduled) if scheduled else max(labels, default="")
    final: pd.DataFrame = data[data["checkpoint"].astype(str) == last]
    return final


#: Billed spend: cached responses (seed replicas) are not counted again.
COST_COLUMN = "actual_cost_usd"


def _write_side_cost(frame: pd.DataFrame, variant: str) -> float:
    """Provider cost a condition paid outside the query schedule, i.e. while writing.

    The rule-based extractor makes no provider calls, so this is zero for it; for the LLM
    extractor it is the cost of the write policy itself, and leaving it out would present the
    policy as nearly free.
    """
    if frame.empty or COST_COLUMN not in frame.columns:
        return float("nan")
    data = _unfaulted(frame)
    rows = data[
        (data["variant"] == variant)
        & ~data["checkpoint"].astype(str).str.startswith(SCHEDULED_PREFIX)
    ]
    return float(rows[COST_COLUMN].fillna(0.0).sum())


def _category(value: object) -> str:
    """The query category, from the grade's metadata whether it is a dict or its JSON text."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return UNCATEGORISED
    if isinstance(value, dict):
        category = value.get("category")
        if category:
            return str(category)
    return UNCATEGORISED


def _outcomes(grades: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.Series]:
    """Per-variant paired outcomes at the final checkpoint, and each key's category."""
    final = _final_unfaulted(grades)
    keys = ["query_id", "seed"] if "seed" in final.columns else ["query_id"]
    outcomes = {
        str(variant): unique_outcomes(group, keys, str(variant))
        for variant, group in final.groupby("variant", dropna=False)
    }
    categories = (
        final.set_index(keys)["metadata"].map(_category)
        if "metadata" in final.columns
        else pd.Series(UNCATEGORISED, index=final.set_index(keys).index)
    )
    categories = categories[~categories.index.duplicated()]
    if "seed" in keys and seeds_are_replicas(outcomes):
        # On a deterministic benchmark every seed re-issues the same requests and the run's
        # cache answers them identically, so pairing on (query, seed) would count each query
        # once per seed and overstate the evidence threefold. The query is the unit of
        # evidence; collapse to it (D-022).
        outcomes = {
            variant: series.groupby(level="query_id").first()
            for variant, series in outcomes.items()
        }
        categories = categories.groupby(level="query_id").first()
    return outcomes, categories


def _ordered(values: list[str], order: tuple[str, ...]) -> list[str]:
    known = [value for value in order if value in values]
    return known + sorted(value for value in values if value not in order)


def _select_pair(
    candidates: list[Candidate], problems: list[str]
) -> tuple[Candidate, Candidate] | None:
    """Choose the (rule, llm) pair with the most paired outcomes; explain every exclusion."""
    families: dict[str, dict[str, list[Candidate]]] = {}
    for candidate in candidates:
        families.setdefault(candidate.signature, {}).setdefault(candidate.policy, []).append(
            candidate
        )

    pairs: list[tuple[Candidate, Candidate]] = []
    for family in families.values():
        if RULE_POLICY not in family or LLM_POLICY not in family:
            for members in family.values():
                for member in members:
                    problems.append(
                        f"excluded {member.run_dir.name}: no counterpart under the other write "
                        "policy shares its model, prompts, budgets, seeds, query set, and "
                        "condition set"
                    )
            continue
        chosen: dict[str, Candidate] = {}
        for policy in (RULE_POLICY, LLM_POLICY):
            # Two runs of one policy with one signature are two samples, not two pairs; the
            # first by name keeps the selection deterministic rather than filesystem-ordered.
            ordered = sorted(family[policy], key=lambda item: item.run_dir.name)
            chosen[policy] = ordered[0]
            for extra in ordered[1:]:
                problems.append(
                    f"excluded {extra.run_dir.name}: {ordered[0].run_dir.name} already holds "
                    f"the {policy} side of this pair; using the first by name"
                )
        pairs.append((chosen[RULE_POLICY], chosen[LLM_POLICY]))

    if not pairs:
        return None
    pairs.sort(
        key=lambda pair: (pair[0].query_count * pair[0].seeds, pair[0].run_dir.name),
        reverse=True,
    )
    for rule, llm in pairs[1:]:
        problems.append(
            f"excluded pair ({rule.run_dir.name}, {llm.run_dir.name}) with "
            f"{rule.query_count} queries x {rule.seeds} seed(s): the pair kept has more paired "
            "outcomes, and two pairs of different sizes are two experiments"
        )
    return pairs[0]


#: The benchmark's own judge (DECISIONS D-025). Runs graded by another judge are a separate
#: analysis, selected by name, never by whichever pair happens to sort first.
PRIMARY_JUDGE = "judge_lme_official_v1"
SECONDARY_JUDGE = "judge_v1"


def analyse(
    results_root: Path,
    *,
    benchmark: str = DEFAULT_BENCHMARK,
    run_dirs: list[Path] | None = None,
    judge: str | None = PRIMARY_JUDGE,
) -> WritePolicyAnalysis:
    """Pair the two runs of ``benchmark`` that differ only in the guarded write policy.

    ``judge`` keeps only runs whose manifest records that judge prompt; ``None`` keeps all.
    """
    runs = run_dirs if run_dirs is not None else discover_runs(results_root)
    problems: list[str] = []
    candidates: list[Candidate] = []
    for run_dir in runs:
        manifest = load_manifest(run_dir)
        if manifest.benchmark_name != benchmark:
            continue
        if judge is not None and judge not in manifest.prompt_hashes:
            continue
        ok, integrity = verify_integrity(run_dir)
        if not ok:
            problems.extend(integrity)
            continue
        policy = policy_of(manifest.extractor_identity)
        if policy is None:
            problems.append(
                f"{run_dir.name}: extractor {manifest.extractor_identity!r} is neither a "
                "rule-based nor an LLM write policy"
            )
            continue
        query_count = workload_size(manifest.notes).get("query_count")
        candidates.append(
            Candidate(
                run_dir=run_dir,
                policy=policy,
                extractor=manifest.extractor_identity,
                signature=pair_signature(manifest),
                query_count=query_count if isinstance(query_count, int) else 0,
                seeds=max(1, len(manifest.random_seeds)),
                configuration_hash=manifest.configuration_hash,
            )
        )

    empty = pd.DataFrame()
    pair = _select_pair(candidates, problems)
    if pair is None:
        return WritePolicyAnalysis(empty, empty, empty, problems)
    rule, llm = pair

    frames = {
        RULE_POLICY: aggregate.load_all(results_root, run_dirs=[rule.run_dir]),
        LLM_POLICY: aggregate.load_all(results_root, run_dirs=[llm.run_dir]),
    }
    grades = {policy: frame.get("grades", empty) for policy, frame in frames.items()}
    for policy, side in ((RULE_POLICY, rule), (LLM_POLICY, llm)):
        if grades[policy].empty:
            problems.append(f"{side.run_dir.name}: no grades")
            return WritePolicyAnalysis(empty, empty, empty, problems)
    outcomes_rule, categories_rule = _outcomes(grades[RULE_POLICY])
    outcomes_llm, _ = _outcomes(grades[LLM_POLICY])

    all_efficiency = {
        policy: metrics_module.efficiency(
            frame.get("answers", empty), frame.get("latency", empty), frame.get("costs", empty)
        )
        for policy, frame in frames.items()
    }
    efficiency = {policy: _final_unfaulted(frame) for policy, frame in all_efficiency.items()}
    storage = {
        policy: _final_unfaulted(
            metrics_module.storage_growth(frame.get("store_statistics", empty))
        )
        for policy, frame in frames.items()
    }

    def stored(policy: str, variant: str, column: str) -> float:
        frame = storage[policy]
        if frame.empty or column not in frame.columns:
            return float("nan")
        rows = frame[frame["variant"] == variant]
        return float(rows[column].iloc[0]) if not rows.empty else float("nan")

    saturation = {
        policy: _final_unfaulted(metrics_module.reader_saturation(grades[policy]))
        for policy in frames
    }

    def measure(policy: str, variant: str, column: str) -> float:
        frame = efficiency[policy]
        if frame.empty or column not in frame.columns:
            return float("nan")
        rows = frame[frame["variant"] == variant]
        return float(rows[column].iloc[0]) if not rows.empty else float("nan")

    def costs(policy: str, variant: str) -> dict[str, float]:
        answer_time = measure(policy, variant, COST_COLUMN)
        write_side = _write_side_cost(all_efficiency[policy], variant)
        total = (0.0 if math.isnan(answer_time) else answer_time) + (
            0.0 if math.isnan(write_side) else write_side
        )
        return {
            f"estimated_cost_usd_{policy}": answer_time,
            f"write_cost_usd_{policy}": write_side,
            f"total_cost_usd_{policy}": (
                float("nan") if math.isnan(answer_time) and math.isnan(write_side) else total
            ),
        }

    condition_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    variants = _ordered(sorted(set(outcomes_rule) | set(outcomes_llm)), CONDITION_ORDER)
    for variant in variants:
        if variant not in outcomes_rule or variant not in outcomes_llm:
            problems.append(f"condition {variant!r} is present under only one write policy")
            continue
        shared = outcomes_rule[variant].index.intersection(outcomes_llm[variant].index)
        unpaired = len(outcomes_rule[variant].index.symmetric_difference(shared)) + len(
            outcomes_llm[variant].index.symmetric_difference(shared)
        )
        if unpaired:
            problems.append(
                f"condition {variant!r}: {unpaired} outcome(s) had no counterpart in the other "
                "run and were left out of the pairing"
            )
        if shared.empty:
            problems.append(f"condition {variant!r}: the two runs share no queries")
            continue
        a = outcomes_rule[variant].reindex(shared).astype(bool)
        b = outcomes_llm[variant].reindex(shared).astype(bool)
        n = len(shared)
        correct_a, correct_b = int(a.sum()), int(b.sum())
        low_a, high_a = wilson_interval(correct_a, n)
        low_b, high_b = wilson_interval(correct_b, n)
        test = mcnemar(list(a), list(b))
        condition_rows.append(
            {
                "variant": variant,
                "label": condition_label(variant),
                "n": n,
                "correct_rule": correct_a,
                "correct_llm": correct_b,
                "accuracy_rule": correct_a / n,
                "accuracy_rule_ci_low": low_a,
                "accuracy_rule_ci_high": high_a,
                "accuracy_llm": correct_b / n,
                "accuracy_llm_ci_low": low_b,
                "accuracy_llm_ci_high": high_b,
                "delta": (correct_b - correct_a) / n,
                # Only discordant pairs carry evidence: the comparison's real sample size.
                "discordant_pairs": int((a != b).sum()),
                "gained": int((b & ~a).sum()),
                "lost": int((a & ~b).sum()),
                "p_value": test.p_value,
                "cohens_g": test.effect_size,
                "effect_label": test.effect_label,
                "method": test.note,
                "warnings": "; ".join(test.warnings),
                "injected_tokens_rule": measure(RULE_POLICY, variant, "injected_tokens_mean"),
                "injected_tokens_llm": measure(LLM_POLICY, variant, "injected_tokens_mean"),
                **costs(RULE_POLICY, variant),
                **costs(LLM_POLICY, variant),
                "latency_p95_ms_retrieval_rule": measure(
                    RULE_POLICY, variant, "latency_p95_ms_retrieval"
                ),
                "latency_p95_ms_retrieval_llm": measure(
                    LLM_POLICY, variant, "latency_p95_ms_retrieval"
                ),
                # Ingestion latency over the stream's final checkpoint window, and on-disk size
                # at the full store: the remaining efficiency measures challenge R2 asks for.
                "latency_p95_ms_ingestion_rule": measure(
                    RULE_POLICY, variant, "latency_p95_ms_ingestion"
                ),
                "latency_p95_ms_ingestion_llm": measure(
                    LLM_POLICY, variant, "latency_p95_ms_ingestion"
                ),
                "database_bytes_rule": stored(RULE_POLICY, variant, "database_size_bytes"),
                "database_bytes_llm": stored(LLM_POLICY, variant, "database_size_bytes"),
                "run_id_rule": rule.run_dir.name,
                "run_id_llm": llm.run_dir.name,
            }
        )
        by_category = categories_rule.reindex(shared).fillna(UNCATEGORISED)
        for category in _ordered(sorted(set(by_category)), CATEGORY_ORDER):
            members = by_category[by_category == category].index
            ca, cb = int(a.reindex(members).sum()), int(b.reindex(members).sum())
            category_rows.append(
                {
                    "variant": variant,
                    "label": condition_label(variant),
                    "category": category,
                    "category_label": category_label(category),
                    "n": len(members),
                    "correct_rule": ca,
                    "correct_llm": cb,
                    "accuracy_rule": ca / len(members),
                    "accuracy_llm": cb / len(members),
                    "delta": (cb - ca) / len(members),
                }
            )

    # The four conditions are one family: one question asked four times.
    adjusted = holm_bonferroni([float(cast(float, row["p_value"])) for row in condition_rows])
    for row, p_adjusted in zip(condition_rows, adjusted, strict=True):
        row["p_value_holm"] = p_adjusted
        row["significant_at_05"] = p_adjusted < 0.05

    saturation_rows: list[dict[str, object]] = []
    for policy, side in ((RULE_POLICY, rule), (LLM_POLICY, llm)):
        frame = saturation[policy]
        if frame.empty:
            continue
        for _, record in frame.iterrows():
            saturation_rows.append(
                {
                    "policy": policy,
                    "extractor": side.extractor,
                    "variant": record["variant"],
                    "label": condition_label(str(record["variant"])),
                    "answerable_queries": int(record["answerable_queries"]),
                    "answerable_with_evidence": int(record["answerable_with_evidence"]),
                    "reader_declined_with_evidence": int(record["reader_declined_with_evidence"]),
                    "reader_saturation": float(record["reader_saturation"]),
                }
            )

    return WritePolicyAnalysis(
        conditions=pd.DataFrame(condition_rows),
        categories=pd.DataFrame(category_rows),
        saturation=pd.DataFrame(saturation_rows),
        problems=problems,
        run_ids=[rule.run_dir.name, llm.run_dir.name],
        configuration_hashes=[rule.configuration_hash, llm.configuration_hash],
        extractors={RULE_POLICY: rule.extractor, LLM_POLICY: llm.extractor},
    )


# ----------------------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------------------

TABLE_NAME = "table9_write_policy"
TABLE_LABEL = "tab:write-policy"
TABLE_CAPTION = (
    "LongMemEval under \\texttt{gpt-5-mini}, at the full store, with the guarded backend's write "
    "policy swapped from the rule-based extractor to the LLM extractor and everything else held "
    "constant. The three baselines have no write policy; their movement is run-to-run drift."
)
FIGURE_NAME = "fig_write_policy"
WORKBOOK_NAME = "figure_data_write_policy.xlsx"


def _p_text(value: float) -> str:
    if math.isnan(value):
        return "--"
    return "$<$0.0001" if value < 0.0001 else f"{value:.4f}"


def _arrow(before: float, after: float, digits: int) -> str:
    if math.isnan(before) and math.isnan(after):
        return "--"
    left = "--" if math.isnan(before) else f"{before:,.{digits}f}"
    right = "--" if math.isnan(after) else f"{after:,.{digits}f}"
    return f"{left}$\\rightarrow${right}"


def write_latex(
    analysis: WritePolicyAnalysis, output_dir: Path, provenance: Provenance
) -> TableResult:
    """Emit the write-policy table, or a pending note that says why there is none."""
    path = output_dir / f"{TABLE_NAME}.tex"
    if not analysis.paired:
        return _pending(path, TABLE_CAPTION, TABLE_LABEL, analysis.reason)
    rows: list[dict[str, object]] = []
    for _, record in analysis.conditions.iterrows():
        rows.append(
            {
                "label": record["label"],
                "acc_rule": (
                    f"{record['accuracy_rule']:.3f} [{record['accuracy_rule_ci_low']:.2f}, "
                    f"{record['accuracy_rule_ci_high']:.2f}]"
                ),
                "acc_llm": (
                    f"{record['accuracy_llm']:.3f} [{record['accuracy_llm_ci_low']:.2f}, "
                    f"{record['accuracy_llm_ci_high']:.2f}]"
                ),
                "discordant": (
                    f"{int(record['discordant_pairs'])} ({int(record['gained'])}/"
                    f"{int(record['lost'])})"
                ),
                "p_holm": _p_text(float(record["p_value_holm"])),
                "tokens": _arrow(
                    float(record["injected_tokens_rule"]), float(record["injected_tokens_llm"]), 0
                ),
                "usd": _arrow(
                    float(record["total_cost_usd_rule"]), float(record["total_cost_usd_llm"]), 3
                ),
                "ingest": _arrow(
                    float(record.get("latency_p95_ms_ingestion_rule", float("nan"))),
                    float(record.get("latency_p95_ms_ingestion_llm", float("nan"))),
                    0,
                ),
                "storage": _arrow(
                    float(record.get("database_bytes_rule", float("nan"))) / 1048576,
                    float(record.get("database_bytes_llm", float("nan"))) / 1048576,
                    1,
                ),
            }
        )
    columns = [
        ("label", "condition"),
        ("acc_rule", "acc.\\ rule [95\\% CI]"),
        ("acc_llm", "acc.\\ LLM [95\\% CI]"),
        ("discordant", "disc.\\ (gained/lost)"),
        ("p_holm", "$p_{\\mathrm{Holm}}$"),
        ("tokens", "inj.\\ tokens"),
        ("usd", "USD"),
        ("ingest", "ingest p95 ms"),
        ("storage", "DB MiB"),
    ]
    note = (
        f"Paired on the same {int(analysis.conditions['n'].max())} queries; the response cache "
        "serves identical requests identically, so the three seeds are replicas and are counted "
        "once. "
        "``disc.'' is the number of outcomes the two policies disagreed on, the comparison's "
        "real sample size, split into queries the LLM policy gained and lost; $p$ is McNemar, "
        "Holm-adjusted within the family of conditions. Tokens are mean injected memory tokens per "
        "answer. USD is the provider cost actually billed for answering and, for the LLM "
        "policy, extraction; grading calls and cached replica seeds are excluded. Ingest p95 is "
        "the per-event write latency over the final checkpoint window; DB is the on-disk store at "
        "the full store. Arrows read rule$\\rightarrow$LLM. Extractors: "
        f"\\texttt{{{analysis.extractors.get(RULE_POLICY, '')}}} and "
        f"\\texttt{{{analysis.extractors.get(LLM_POLICY, '')}}}."
    ).replace("_", "\\_")
    options: dict[str, Any] = {"digits": 3, "note": note}
    # Seven columns do not fit a single column of a two-column page. ``_tabular`` may grow a
    # ``wide`` flag; until it does, the float environment is widened here.
    if "wide" in inspect.signature(_tabular).parameters:
        options["wide"] = True
        content = _tabular(
            pd.DataFrame(rows), columns, TABLE_CAPTION, TABLE_LABEL, provenance, **options
        )
    else:
        content = (
            _tabular(pd.DataFrame(rows), columns, TABLE_CAPTION, TABLE_LABEL, provenance, **options)
            .replace(r"\begin{table}[t]", r"\begin{table*}[t]")
            .replace(r"\end{table}", r"\end{table*}")
        )
    return TableResult(path.stem, _write(path, content), False)


#: Figure geometry, in inches: one IEEE column wide, and as tall as its rows need.
FIGURE_WIDTH = 3.4
ROW_HEIGHT = 0.16
FIGURE_MARGIN = 1.05
GROUP_GAP = 0.6


def _finish_wrapped(figure: Figure, provenance: Provenance, path: Path) -> Path:
    """The provenance footer of ``figures._finish``, wrapped to the figure's width.

    A single-column figure is narrower than the footer's run ids; left unwrapped, the footer
    would widen the saved bounding box and the page would scale the data down to fit it.
    """
    # The PDF that goes into the paper carries no footer (DECISIONS D-023); the PNG and the
    # provenance sidecar keep it.
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    height = figure.get_size_inches()[1]
    line = (4.5 * 1.35 / 72) / height
    for index, text in enumerate(textwrap.wrap(provenance.figure_footer(), width=96)):
        figure.text(
            0.0,
            -0.015 - line * index,
            text,
            fontsize=4.5,
            color="#555555",
            transform=figure.transFigure,
            va="top",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    provenance.write_sidecar(path)
    return path


def _tint(colour: str) -> str:
    """The same hue at a third of the strength, for the bar that is the "before" of a pair."""
    red, green, blue = (int(colour[index : index + 2], 16) for index in (1, 3, 5))
    mix = tuple(int(255 - (255 - channel) * 0.35) for channel in (red, green, blue))
    return "#{:02x}{:02x}{:02x}".format(*mix)


def write_figure(
    analysis: WritePolicyAnalysis, output_dir: Path, provenance: Provenance
) -> FigureResult:
    """Paired horizontal bars: per condition and category, accuracy under each write policy."""
    output = output_dir / f"{FIGURE_NAME}.png"
    if not analysis.paired:
        return FigureResult(output.stem, None, True, analysis.reason)
    categories = analysis.categories[analysis.categories["n"] > 0]
    if categories.empty:
        return FigureResult(output.stem, None, True, "no category had a paired outcome")

    _style()
    plt.rcParams.update({"font.size": 8, "legend.fontsize": 7})

    # Lay the rows out top to bottom: a bold header per condition, then one row per category.
    headers: list[tuple[float, str, str]] = []
    rows: list[tuple[float, str, str, float, float]] = []
    position = 0.0
    drawn = [
        variant
        for variant in _ordered(sorted(set(categories["variant"])), CONDITION_ORDER)
        if variant in FIGURE_CONDITIONS
    ]
    for variant in drawn:
        group = categories[categories["variant"] == variant]
        by_category: dict[str, tuple[int, float, float]] = {
            str(category): (int(n), float(before), float(after))
            for category, n, before, after in zip(
                group["category"],
                group["n"],
                group["accuracy_rule"],
                group["accuracy_llm"],
                strict=True,
            )
        }
        headers.append((position, condition_label(variant), _color(variant)))
        position += 1.0
        for category in _ordered(list(by_category), CATEGORY_ORDER):
            n, before, after = by_category[category]
            rows.append(
                (position, f"{category_label(category)} (n={n})", _color(variant), before, after)
            )
            position += 1.0
        position += GROUP_GAP

    height = ROW_HEIGHT * position + FIGURE_MARGIN
    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH, height))
    # Two bars per category: the rule-based policy in a light tint of the condition's colour
    # above, the model-based policy in the full colour below, each with its value at its end.
    bar = 0.36
    first_rows = {min(y for y, *_ in rows if y > header) for header, *_ in headers}
    for y, _, colour, before, after in rows:
        tint = _tint(colour)
        axes.barh(y - bar / 2, before, height=bar, color=tint, zorder=2)
        axes.barh(y + bar / 2, after, height=bar, color=colour, zorder=2)
        # The policies are named on the first pair of each group, after the value, in place of
        # a legend: a legend swatch in a neutral colour would match no bar in the figure.
        rule = f"{before:.2f}" + ("  rule-based policy" if y in first_rows else "")
        model = f"{after:.2f}" + ("  model-based policy" if y in first_rows else "")
        axes.text(before + 0.015, y - bar / 2, rule, va="center", fontsize=6, color="#333333")
        axes.text(after + 0.015, y + bar / 2, model, va="center", fontsize=6, color="#333333")

    ticks = [y for y, *_ in headers] + [y for y, *_ in rows]
    labels = [label for _, label, _ in headers] + [label for _, label, *_ in rows]
    order = sorted(range(len(ticks)), key=lambda index: ticks[index])
    axes.set_yticks([ticks[index] for index in order])
    axes.set_yticklabels([labels[index] for index in order], fontsize=7)
    header_positions = {y for y, *_ in headers}
    header_colours = {y: colour for y, _, colour in headers}
    for tick, label in zip(axes.get_yticks(), axes.get_yticklabels(), strict=True):
        if float(tick) in header_positions:
            label.set_fontweight("bold")
            label.set_color(header_colours[float(tick)])
            label.set_fontsize(7.5)
    axes.set_ylim(position - GROUP_GAP - 0.4, -0.6)
    axes.set_xlim(0.0, 1.45)
    axes.set_xticks([0.0, 0.5, 1.0])
    axes.set_xlabel("accuracy at the full store")
    axes.grid(axis="y", visible=False)
    axes.tick_params(axis="y", length=0)
    for y in list(header_positions)[1:]:
        axes.axhline(y - 0.5 - GROUP_GAP / 2, color="#DDDDDD", linewidth=0.5, zorder=0)

    figure.tight_layout()
    return FigureResult(output.stem, _finish_wrapped(figure, provenance, output), False)


def write_workbook(
    analysis: WritePolicyAnalysis, output: Path, provenance: Provenance
) -> WorkbookResult:
    """The figure's and the table's data, as a spreadsheet a reader can re-plot from."""
    candidates = {
        "conditions": analysis.conditions,
        "categories": analysis.categories,
        "saturation": analysis.saturation,
    }
    results: list[SheetResult] = []
    writable: dict[str, pd.DataFrame] = {}
    for name, frame in candidates.items():
        if frame.empty:
            results.append(SheetResult(name, 0, True, analysis.reason or "no data"))
            continue
        writable[name] = frame
        results.append(SheetResult(name, len(frame), False))
    if not writable:
        return WorkbookResult(None, results, True, analysis.reason or "no sheet had data")
    output.parent.mkdir(parents=True, exist_ok=True)
    group = "write policy: " + " vs ".join(analysis.run_ids)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _readme(provenance, group, results).to_excel(writer, sheet_name="README", index=False)
        for name, frame in writable.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        _autofit(writer)
    provenance.write_sidecar(output)
    return WorkbookResult(output, results)


def write_processed(analysis: WritePolicyAnalysis, processed_dir: Path) -> list[Path]:
    """The frames behind the table and the figure, plus which runs were paired and why not."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in (
        ("conditions", analysis.conditions),
        ("categories", analysis.categories),
        ("saturation", analysis.saturation),
    ):
        if frame.empty:
            continue
        path = processed_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path)
    selection = processed_dir / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "generated_at": utcnow().isoformat(),
                "paired_run_ids": analysis.run_ids,
                "extractors": analysis.extractors,
                "problems": analysis.problems,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    written.append(selection)
    return written


@dataclass
class WritePolicyResult:
    analysis: WritePolicyAnalysis
    processed_files: list[Path]
    table: TableResult
    figure: FigureResult
    workbook: WorkbookResult

    def summary(self) -> str:
        table_state = "pending" if self.table.pending else "written"
        figure_state = (
            f"skipped ({self.figure.reason})" if self.figure.skipped else str(self.figure.path)
        )
        lines = [
            f"write policy pair    : {' vs '.join(self.analysis.run_ids) or 'none found'}",
            f"write policy table   : {self.table.path} ({table_state})",
            f"write policy figure  : {figure_state}",
        ]
        if self.analysis.problems:
            lines.append("write policy problems:")
            lines.extend(f"  - {problem}" for problem in self.analysis.problems)
        return "\n".join(lines)


def _record_in_index(index_path: Path, result: WritePolicyResult) -> None:
    """Add this analysis to the report index the per-group report wrote, replacing by name."""
    payload: dict[str, Any] = (
        json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    )
    payload.setdefault("generated_at", utcnow().isoformat())
    figures = [item for item in payload.get("figures", []) if item.get("name") != FIGURE_NAME]
    figures.append(
        {"name": FIGURE_NAME, "skipped": result.figure.skipped, "reason": result.figure.reason}
    )
    tables = [item for item in payload.get("tables", []) if item.get("name") != TABLE_NAME]
    tables.append(
        {"name": TABLE_NAME, "pending": result.table.pending, "reason": result.table.reason}
    )
    payload["figures"] = figures
    payload["tables"] = tables
    payload["write_policy"] = {
        "run_ids": result.analysis.run_ids,
        "problems": result.analysis.problems,
        "workbook": str(result.workbook.path) if result.workbook.path else None,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build(
    results_root: Path,
    paper_dir: Path,
    *,
    benchmark: str = DEFAULT_BENCHMARK,
    processed_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> WritePolicyResult:
    """Run the analysis and place its outputs exactly where the per-group report places its own."""
    processed_dir = processed_dir or Path("results/processed/write_policy")
    figures_dir = figures_dir or Path("results/figures/write_policy")
    analysis = analyse(results_root, benchmark=benchmark)
    # The pair as first graded, by the harness's own judge, is kept beside it so the effect of
    # the judge on every number can be read off; it feeds no table or figure.
    secondary = analyse(results_root, benchmark=benchmark, judge=SECONDARY_JUDGE)
    if secondary.paired:
        write_processed(secondary, processed_dir.parent / f"{processed_dir.name}_{SECONDARY_JUDGE}")
    provenance = Provenance.create(
        artifact="memcheck report (write policy)",
        script="src/memcheck/reporting/writepolicy.py",
        run_ids=analysis.run_ids,
        configuration_hashes=analysis.configuration_hashes,
    )
    processed = write_processed(analysis, processed_dir)
    generated = paper_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    table = write_latex(analysis, generated, provenance)
    figure = write_figure(analysis, figures_dir, provenance)
    workbook = write_workbook(analysis, figures_dir / WORKBOOK_NAME, provenance)

    # Copy the paper-facing artifacts beside the per-group ones; sidecars stay with the originals.
    target = generated / "figures"
    target.mkdir(parents=True, exist_ok=True)
    if figure.path is not None:
        for suffix in (".png", ".pdf"):
            source = figure.path.with_suffix(suffix)
            if source.exists():
                (target / source.name).write_bytes(source.read_bytes())
    if workbook.path is not None:
        (target / workbook.path.name).write_bytes(workbook.path.read_bytes())

    result = WritePolicyResult(analysis, processed, table, figure, workbook)
    _record_in_index(generated / "report_index.json", result)
    return result


__all__ = [
    "PALETTE",
    "WritePolicyAnalysis",
    "WritePolicyResult",
    "analyse",
    "build",
    "pair_signature",
    "policy_of",
    "write_figure",
    "write_latex",
    "write_processed",
    "write_workbook",
]
