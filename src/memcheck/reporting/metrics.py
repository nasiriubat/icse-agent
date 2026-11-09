"""Metric computation (Section 16).

Every metric here is a pure function of the raw streams. Nothing is estimated, imputed, or
smoothed, and failed conditions are counted rather than dropped.
"""

from __future__ import annotations

from itertools import combinations
from typing import cast

import pandas as pd

from memcheck.utilities.statistics import (
    holm_bonferroni,
    mcnemar,
    percentile,
    summarize,
    wilson_interval,
)

GROUP_KEYS = ["variant", "fault", "checkpoint"]

#: Labels that count as the agent having answered correctly, including a correct refusal.
CORRECT_LABELS = frozenset({"correct", "abstained_correctly"})


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def task_utility(grades: pd.DataFrame) -> pd.DataFrame:
    """16.1 — accuracy and the breakdown of *how* answers fail."""
    if grades.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in grades.groupby(GROUP_KEYS, dropna=False):
        total = len(group)
        labels = group["label"].value_counts().to_dict()
        correct = labels.get("correct", 0) + labels.get("abstained_correctly", 0)
        low, high = wilson_interval(correct, total)
        rows.append(
            {
                **dict(zip(GROUP_KEYS, keys, strict=True)),
                "n": total,
                "accuracy": _safe_rate(correct, total),
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "exact_match": _safe_rate(labels.get("correct", 0), total),
                "stale_memory_error_rate": _safe_rate(labels.get("stale", 0), total),
                "deletion_leakage_rate": _safe_rate(labels.get("leaked", 0), total),
                "unsupported_answer_rate": _safe_rate(labels.get("unsupported", 0), total),
                "incorrect_rate": _safe_rate(labels.get("incorrect", 0), total),
                "wrong_abstention_rate": _safe_rate(labels.get("abstained_wrongly", 0), total),
            }
        )
    return pd.DataFrame(rows)


def unique_outcomes(group: pd.DataFrame, keys: list[str], variant: str) -> pd.Series:
    """Per-query correct/incorrect for one condition, refusing a duplicated query.

    A duplicate means two runs that are not two samples of the same experiment were pooled, and
    pairing them would silently compare a query against a different run's answer to it. That is
    the failure this raises on: pandas reports it as "non-unique multi-index", which says nothing
    about the cause.
    """
    outcomes = group.set_index(keys)["label"].isin(CORRECT_LABELS)
    if outcomes.index.has_duplicates:
        duplicated = outcomes.index[outcomes.index.duplicated()][:3].tolist()
        raise ValueError(
            f"condition {variant!r} answers the same query more than once (e.g. {duplicated}). "
            "Runs that are not samples of one experiment have been pooled; check that every "
            "field distinguishing them is in fairness.GROUPING_FIELDS."
        )
    return outcomes


def seeds_are_replicas(outcomes: dict[str, pd.Series]) -> bool:
    """True when, for every condition, every query had the same outcome under every seed.

    A single disagreement anywhere keeps the (query, seed) pairing, because then the seeds did
    carry information and collapsing them would discard it. A condition run under one seed says
    nothing either way.
    """
    seen_multi_seed = False
    for series in outcomes.values():
        if "seed" not in (series.index.names or []):
            return False
        frame = series.rename("correct").reset_index()
        if frame["seed"].nunique() < 2:
            continue
        seen_multi_seed = True
        if not bool(frame.groupby("query_id")["correct"].nunique().eq(1).all()):
            return False
    return seen_multi_seed


def collapse_replicas(outcomes: dict[str, pd.Series]) -> tuple[dict[str, pd.Series], bool]:
    """Collapse (query, seed) outcomes to one per query when the seeds are replicas.

    Seeds that reproduce each other exactly are copies, not samples; pairing on (query, seed)
    would count each query once per seed and overstate the evidence by that factor (D-022,
    V-015, V-018). Returns the outcomes to test on and whether a collapse happened, so a report
    can say so.
    """
    if not seeds_are_replicas(outcomes):
        return outcomes, False
    return {
        variant: series.groupby(level="query_id").first() for variant, series in outcomes.items()
    }, True


def paired_comparisons(grades: pd.DataFrame) -> pd.DataFrame:
    """Paired significance tests between every pair of conditions, at every checkpoint.

    Two conditions answer the *same* query set, so their per-query outcomes are paired and only
    the queries they disagree on carry information. McNemar is therefore the correct test, and
    treating the two accuracy rates as independent samples would overstate the evidence.

    Every comparison within a checkpoint belongs to one family, so p-values are Holm-adjusted
    within that family. Both the raw and the adjusted value are reported: the adjustment is a
    decision about multiplicity that a reader is entitled to check.
    """
    if grades.empty or "query_id" not in grades.columns:
        return pd.DataFrame()
    # Faults change the query outcomes, so pairing is only meaningful within one fault setting.
    unfaulted = grades[grades["fault"] == "none"] if "fault" in grades.columns else grades
    if unfaulted.empty:
        unfaulted = grades

    rows: list[dict[str, object]] = []
    for checkpoint, checkpoint_grades in unfaulted.groupby("checkpoint", dropna=False):
        # A query answered under several seeds appears once per seed; the seed is part of the
        # pairing key so a query is matched against its own counterpart, not another draw.
        keys = ["query_id", "seed"] if "seed" in checkpoint_grades.columns else ["query_id"]
        outcomes, collapsed = collapse_replicas(
            {
                str(variant): unique_outcomes(group, keys, str(variant))
                for variant, group in checkpoint_grades.groupby("variant", dropna=False)
            }
        )
        family: list[dict[str, object]] = []
        for left, right in combinations(sorted(outcomes), 2):
            # Align on the shared queries only. A condition that was not asked a question cannot
            # be said to have got it wrong.
            shared = outcomes[left].index.intersection(outcomes[right].index)
            if shared.empty:
                continue
            a = outcomes[left].reindex(shared)
            b = outcomes[right].reindex(shared)
            result = mcnemar(list(a), list(b))
            family.append(
                {
                    "checkpoint": checkpoint,
                    "variant_a": left,
                    "variant_b": right,
                    "n_paired": result.n,
                    "accuracy_a": float(a.mean()),
                    "accuracy_b": float(b.mean()),
                    "a_correct_b_wrong": result.statistic,
                    "p_value": result.p_value,
                    "cohens_g": result.effect_size,
                    "effect_label": result.effect_label,
                    "method": result.note,
                    "seeds_collapsed": collapsed,
                    "warnings": "; ".join(result.warnings),
                }
            )
        adjusted = holm_bonferroni([float(cast(float, row["p_value"])) for row in family])
        for row, p_adjusted in zip(family, adjusted, strict=True):
            row["p_value_holm"] = p_adjusted
            row["significant_at_05"] = p_adjusted < 0.05
        rows.extend(family)
    return pd.DataFrame(rows)


def memory_quality(grades: pd.DataFrame) -> pd.DataFrame:
    """16.2 — evidence quality, update success, abstention precision and recall."""
    if grades.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in grades.groupby(GROUP_KEYS, dropna=False):
        categories = group.get("metadata")
        category_series = (
            categories.map(lambda value: (value or {}).get("category"))
            if categories is not None
            else pd.Series([None] * len(group), index=group.index)
        )
        updates = group[category_series == "knowledge_update"]
        historical = group[category_series == "temporal_reasoning"]

        expected_abstain = group[
            group["label"].isin(["abstained_correctly", "unsupported", "leaked"])
        ]
        did_abstain = group[group["label"].isin(["abstained_correctly", "abstained_wrongly"])]
        true_abstain = int((group["label"] == "abstained_correctly").sum())

        rows.append(
            {
                **dict(zip(GROUP_KEYS, keys, strict=True)),
                "n": len(group),
                "evidence_recall": float(group["evidence_recall"].dropna().mean())
                if "evidence_recall" in group
                else float("nan"),
                "evidence_precision": float(group["evidence_precision"].dropna().mean())
                if "evidence_precision" in group
                else float("nan"),
                "update_success_rate": _safe_rate(int(updates["correct"].sum()), len(updates)),
                "historical_query_correctness": _safe_rate(
                    int(historical["correct"].sum()), len(historical)
                ),
                "abstention_precision": _safe_rate(true_abstain, len(did_abstain)),
                "abstention_recall": _safe_rate(true_abstain, len(expected_abstain)),
            }
        )
    return pd.DataFrame(rows)


def contract_violations(violations: pd.DataFrame) -> pd.DataFrame:
    """Violation counts per condition, so a count in the paper binds to an artifact.

    ``reliability`` reports the pass *rate* over contract evaluations, which is a different
    quantity: one failing contract can raise many violations, and it is the violation count that
    a reader compares against an accuracy gap.
    """
    if violations.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["variant", "fault"] if "fault" in violations.columns else ["variant"]
    for group_keys, group in violations.groupby(keys, dropna=False):
        values = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        severities = group["severity"].value_counts().to_dict()
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "violations": len(group),
                "critical_violations": int(severities.get("critical", 0)),
                "contracts_violated": int(group["contract_name"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def reader_saturation(grades: pd.DataFrame) -> pd.DataFrame:
    """How often the reader declined *wrongly* although the backend did supply evidence.

    This separates "the memory system found nothing" from "the reader could not use what it was
    given". Correct abstentions — on a deleted fact or an unresolved contradiction — are
    successes and are excluded; only ``abstained_wrongly`` counts, and only where evidence was
    actually retrieved.

    On a templated workload the value is small. On a free-text benchmark the deterministic
    oracle reader saturates it and every backend then scores the same, which is a fact about the
    reader rather than about memory and must not be reported as a finding.
    """
    if grades.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in grades.groupby(GROUP_KEYS, dropna=False):
        # A query that expected an answer: anything not graded as a correct abstention.
        answerable = group[group["label"] != "abstained_correctly"]
        retrieved = answerable["retrieved_memory_ids"].map(
            lambda value: bool(value) and value != "[]"
        )
        with_evidence = answerable[retrieved]
        declined = with_evidence[with_evidence["label"] == "abstained_wrongly"]
        rows.append(
            {
                **dict(zip(GROUP_KEYS, keys, strict=True)),
                "answerable_queries": len(answerable),
                "answerable_with_evidence": len(with_evidence),
                "reader_declined_with_evidence": len(declined),
                "reader_saturation": _safe_rate(len(declined), len(with_evidence)),
            }
        )
    return pd.DataFrame(rows)


def reliability(
    contracts: pd.DataFrame,
    violations: pd.DataFrame,
    faults: pd.DataFrame,
    writes: pd.DataFrame,
) -> pd.DataFrame:
    """16.3 — contract violations, recovery, duplicate amplification, leakage."""
    rows = []
    if not contracts.empty:
        for keys, group in contracts.groupby(["variant", "fault"], dropna=False):
            # A vacuous result never exercised its invariant: it is neither a pass nor a failure.
            vacuous = (
                group["vacuous"].fillna(False).astype(bool)
                if "vacuous" in group.columns
                else pd.Series(False, index=group.index)
            )
            evaluated = group[~group["skipped"] & ~vacuous]
            rows.append(
                {
                    "variant": keys[0],
                    "fault": keys[1],
                    "contracts_vacuous": int(vacuous.sum()),
                    "contracts_evaluated": len(evaluated),
                    "contracts_failed": int((~evaluated["passed"]).sum()),
                    "contract_pass_rate": _safe_rate(
                        int(evaluated["passed"].sum()), len(evaluated)
                    ),
                }
            )
    frame = pd.DataFrame(rows)

    if not violations.empty:
        severity = (
            violations.groupby(["backend_name", "fault_name", "severity"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        frame.attrs["severity_breakdown"] = severity
        contract_counts = (
            violations.groupby(["backend_name", "contract_name"], dropna=False)
            .size()
            .reset_index(name="violations")
        )
        frame.attrs["by_contract"] = contract_counts

    if not faults.empty:
        recovery = (
            faults.groupby(["variant", "fault_name"], dropna=False)["recovered"]
            .agg(["sum", "count"])
            .reset_index()
        )
        recovery["recovery_success_rate"] = recovery["sum"] / recovery["count"]
        frame = frame.merge(
            recovery.rename(columns={"fault_name": "fault"})[
                ["variant", "fault", "recovery_success_rate"]
            ],
            on=["variant", "fault"],
            how="outer",
        )

    if not writes.empty and "outcome" in writes.columns:
        duplicates = (
            writes.assign(is_duplicate=writes["outcome"] == "duplicate_ignored")
            .groupby(["variant", "fault"], dropna=False)["is_duplicate"]
            .agg(["sum", "count"])
            .reset_index()
        )
        duplicates["duplicate_ignored_rate"] = duplicates["sum"] / duplicates["count"]
        frame = frame.merge(
            duplicates[["variant", "fault", "duplicate_ignored_rate"]],
            on=["variant", "fault"],
            how="outer",
        )
    return frame


def efficiency(answers: pd.DataFrame, latency: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """16.4 — tokens, money, and p50/p95 latency by phase."""
    rows = []
    if not answers.empty:
        for keys, group in answers.groupby(GROUP_KEYS, dropna=False):
            injected = group["injected_tokens"].astype(float)
            summary = summarize(injected.tolist())
            rows.append(
                {
                    **dict(zip(GROUP_KEYS, keys, strict=True)),
                    "n_answers": len(group),
                    "injected_tokens_mean": summary.mean,
                    "injected_tokens_median": summary.median,
                    "injected_tokens_stdev": summary.stdev,
                    "injected_tokens_ci_low": summary.ci_low,
                    "injected_tokens_ci_high": summary.ci_high,
                    "injected_tokens_total": float(injected.sum()),
                }
            )
    frame = pd.DataFrame(rows)

    if not latency.empty:
        latency_rows = []
        latency_groups = latency.groupby(["variant", "fault", "checkpoint", "phase"], dropna=False)
        for group_keys, group in latency_groups:
            # A four-key groupby always yields a 4-tuple; naming the parts keeps mypy honest.
            variant, fault_name, checkpoint, phase = tuple(group_keys)
            values = group["latency_ms"].astype(float).tolist()
            latency_rows.append(
                {
                    "variant": variant,
                    "fault": fault_name,
                    "checkpoint": checkpoint,
                    "phase": phase,
                    "latency_p50_ms": percentile(values, 0.50),
                    "latency_p95_ms": percentile(values, 0.95),
                    "latency_mean_ms": sum(values) / len(values) if values else float("nan"),
                    "n_measurements": len(values),
                }
            )
        latency_frame = pd.DataFrame(latency_rows)
        wide = latency_frame.pivot_table(
            index=["variant", "fault", "checkpoint"],
            columns="phase",
            values=["latency_p50_ms", "latency_p95_ms"],
        )
        # pivot_table over two value columns yields a two-level column index; flatten it to
        # names like "latency_p95_ms_retrieval" that the tables and figures ask for by name.
        # pandas-stubs declares .columns as Index[str], which does not describe the MultiIndex
        # this call actually produces, so the cast documents the real shape.
        levels = cast(list[tuple[str, str]], list(wide.columns))
        wide.columns = pd.Index([f"{metric}_{phase}" for metric, phase in levels])
        frame = frame.merge(wide.reset_index(), on=GROUP_KEYS, how="outer")
        frame.attrs["latency_long"] = latency_frame

    if not costs.empty:
        # A cached response carries the original call's cost, so summing estimated_cost_usd
        # counts a replicated request once per replica. ``actual_cost_usd`` is what the provider
        # billed: uncached calls only.
        # Grading calls are the evaluation's cost, not the memory system's, so they are split
        # out rather than billed to the condition that was graded.
        uncached = (
            costs["estimated_cost_usd"].where(~costs["cached"].fillna(False).astype(bool), 0.0)
            if "cached" in costs.columns
            else costs["estimated_cost_usd"]
        )
        grading = (
            costs["purpose"].eq("judge")
            if "purpose" in costs.columns
            else pd.Series(False, index=costs.index)
        )
        billed = costs.assign(
            actual_cost_usd=uncached.where(~grading, 0.0),
            judge_cost_usd=uncached.where(grading, 0.0),
        )
        cost_rows = (
            billed.groupby(GROUP_KEYS, dropna=False)
            .agg(
                input_tokens=("input_tokens", "sum"),
                output_tokens=("output_tokens", "sum"),
                estimated_cost_usd=("estimated_cost_usd", "sum"),
                actual_cost_usd=("actual_cost_usd", "sum"),
                judge_cost_usd=("judge_cost_usd", "sum"),
                provider_calls=("request_identifier", "count"),
            )
            .reset_index()
        )
        cost_rows["total_tokens"] = cost_rows["input_tokens"] + cost_rows["output_tokens"]
        frame = frame.merge(cost_rows, on=GROUP_KEYS, how="outer")
    return frame


def storage_growth(statistics: pd.DataFrame) -> pd.DataFrame:
    """16.5 — database and index size, record counts, duplicate ratio, growth per session."""
    if statistics.empty:
        return pd.DataFrame()
    overall = statistics[statistics["tenant_id"].isna()]
    if overall.empty:
        overall = statistics
    return (
        overall.groupby(GROUP_KEYS, dropna=False)
        .agg(
            database_size_bytes=("database_size_bytes", "mean"),
            index_size_bytes=("index_size_bytes", "mean"),
            event_count=("event_count", "mean"),
            memory_record_count=("memory_record_count", "mean"),
            active_memory_count=("active_memory_count", "mean"),
            superseded_record_count=("superseded_record_count", "mean"),
            deleted_record_count=("deleted_record_count", "mean"),
            conflicted_record_count=("conflicted_record_count", "mean"),
            duplicate_ratio=("duplicate_ratio", "mean"),
            sessions_ingested=("sessions_ingested", "mean"),
            storage_growth_per_session=("storage_growth_per_session", "mean"),
            observations=("database_size_bytes", "count"),
        )
        .reset_index()
    )


def coding_outcomes(coding: pd.DataFrame) -> pd.DataFrame:
    """RepoEvolutionBench: unit-test pass rate and why a patch was wrong."""
    if coding.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in coding.groupby(["variant", "fault", "checkpoint"], dropna=False):
        total = len(group)
        passed = int(group["tests_passed"].sum())
        low, high = wilson_interval(passed, total)
        rows.append(
            {
                "variant": keys[0],
                "fault": keys[1],
                "checkpoint": keys[2],
                "n_tasks": total,
                "coding_task_success": _safe_rate(passed, total),
                "success_ci_low": low,
                "success_ci_high": high,
                "unit_test_pass_rate": _safe_rate(passed, int(group["tests_run"].sum())),
                "patch_applied_rate": _safe_rate(int(group["patch_applied"].sum()), total),
                "abstained_rate": _safe_rate(int(group["abstained"].sum()), total),
                "stale_value_rate": _safe_rate(int(group["used_stale_value"].sum()), total),
                "forbidden_value_rate": _safe_rate(int(group["used_forbidden_value"].sum()), total),
            }
        )
    return pd.DataFrame(rows)


def compute_all(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute every metric family from the loaded raw streams."""
    empty = pd.DataFrame()
    return {
        "task_utility": task_utility(frames.get("grades", empty)),
        "memory_quality": memory_quality(frames.get("grades", empty)),
        "reliability": reliability(
            frames.get("contracts", empty),
            frames.get("violations", empty),
            frames.get("faults", empty),
            frames.get("writes", empty),
        ),
        "efficiency": efficiency(
            frames.get("answers", empty),
            frames.get("latency", empty),
            frames.get("costs", empty),
        ),
        "storage_growth": storage_growth(frames.get("store_statistics", empty)),
        "reader_saturation": reader_saturation(frames.get("grades", empty)),
        "paired_comparisons": paired_comparisons(frames.get("grades", empty)),
        "contract_violations": contract_violations(frames.get("violations", empty)),
        "coding_outcomes": coding_outcomes(frames.get("coding_tasks", empty)),
    }
