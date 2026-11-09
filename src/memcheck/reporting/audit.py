"""Human audit pipeline (Section 18).

Even though MemCheck's default grading is deterministic, an audit path is required so that any
grade a human might dispute can be checked, and so that an LLM-judged configuration has a
measurable agreement number rather than an assumed one.

The sample is deterministic and stratified over correct, incorrect, abstained, and disputed
cases, so two people auditing the same results review the same rows.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from memcheck.models import content_hash
from memcheck.providers.prompts import prompt_hash

STRATA = ("correct", "incorrect", "abstained", "disputed")

AUDIT_COLUMNS = [
    "audit_id",
    "stratum",
    "run_id",
    "condition_id",
    "variant",
    "checkpoint",
    "query_id",
    "query",
    "gold_answer",
    "model_answer",
    "judge_label",
    "judge_rationale",
    "retrieved_evidence",
    "human_label",
    "human_confidence",
    "human_notes",
]


@dataclass
class AuditExport:
    path: Path
    rows: int
    strata_counts: dict[str, int]
    judge_prompt_hash: str


def _stratum_of(row: pd.Series) -> str:
    label = str(row.get("label", ""))
    if label in ("correct", "abstained_correctly"):
        return "correct" if label == "correct" else "abstained"
    if label in ("abstained_wrongly",):
        return "abstained"
    if label in ("stale", "leaked", "unsupported"):
        # Cases where the grader and a human are most likely to disagree about intent.
        return "disputed"
    return "incorrect"


def build_sample(frames: dict[str, pd.DataFrame], *, sample_size: int = 60) -> pd.DataFrame:
    """Deterministic stratified sample, balanced across the four strata."""
    grades = frames.get("grades", pd.DataFrame())
    if grades.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    working = grades.copy()
    working["stratum"] = working.apply(_stratum_of, axis=1)
    # A content hash gives a stable, machine-independent ordering — no RNG involved.
    working["audit_id"] = working.apply(
        lambda row: content_hash(
            [row.get("run_id"), row.get("condition_id"), row.get("query_id"), row.get("checkpoint")]
        )[:16],
        axis=1,
    )
    working = working.sort_values("audit_id")

    per_stratum = max(1, sample_size // len(STRATA))
    parts = []
    for stratum in STRATA:
        subset = working[working["stratum"] == stratum]
        parts.append(subset.head(per_stratum))
    sample = pd.concat(parts).drop_duplicates(subset=["audit_id"])

    retrievals = frames.get("retrievals", pd.DataFrame())
    evidence_by_key: dict[tuple[str, str, str], str] = {}
    if not retrievals.empty:
        for _, row in retrievals.iterrows():
            key = (
                str(row.get("condition_id")),
                str(row.get("query_id")),
                str(row.get("checkpoint")),
            )
            evidence = row.get("evidence") or []
            if isinstance(evidence, str):
                try:
                    evidence = json.loads(evidence)
                except json.JSONDecodeError:
                    evidence = []
            evidence_by_key[key] = " || ".join(
                str(item.get("content", "")) for item in evidence if isinstance(item, dict)
            )

    answers = frames.get("answers", pd.DataFrame())
    query_text: dict[tuple[str, str], str] = {}
    if not answers.empty:
        for _, row in answers.iterrows():
            query_text[(str(row.get("condition_id")), str(row.get("query_id")))] = str(
                row.get("query", "")
            )

    rows = []
    for _, row in sample.iterrows():
        key = (str(row.get("condition_id")), str(row.get("query_id")), str(row.get("checkpoint")))
        rows.append(
            {
                "audit_id": row["audit_id"],
                "stratum": row["stratum"],
                "run_id": row.get("run_id", ""),
                "condition_id": row.get("condition_id", ""),
                "variant": row.get("variant", row.get("backend_name", "")),
                "checkpoint": row.get("checkpoint", ""),
                "query_id": row.get("query_id", ""),
                "query": query_text.get(
                    (str(row.get("condition_id")), str(row.get("query_id"))), ""
                ),
                "gold_answer": row.get("gold_answer", ""),
                "model_answer": row.get("model_answer", ""),
                "judge_label": row.get("label", ""),
                "judge_rationale": row.get("judge_rationale") or f"grader={row.get('grader', '')}",
                "retrieved_evidence": evidence_by_key.get(key, ""),
                "human_label": "",
                "human_confidence": "",
                "human_notes": "",
            }
        )
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def export(
    frames: dict[str, pd.DataFrame], output_dir: Path, *, sample_size: int = 60
) -> AuditExport:
    """Write the editable audit CSV plus a README explaining how to fill it in."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = build_sample(frames, sample_size=sample_size)
    path = output_dir / "audit_sample.csv"
    sample.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)

    (output_dir / "README.md").write_text(
        "# Human audit\n\n"
        "1. Open `audit_sample.csv`.\n"
        "2. For each row, read `query`, `gold_answer`, and `model_answer`. The question you are "
        "answering is: **does the model answer say the same thing as the gold answer?** Judge "
        "the answer against the gold answer, not against `retrieved_evidence`; the evidence is "
        "there only to help you understand the case. An answer that faithfully repeats evidence "
        "which the gold answer contradicts is still `incorrect`.\n"
        "3. Fill `human_label` with one of: `correct` (same meaning as the gold answer), "
        "`incorrect` (a different or contradicting value), `abstained` (the model declined to "
        "answer, whether or not that was the right thing to do), `unsupported` (the gold answer "
        "is empty because the question rests on a false premise, and the model answered anyway), "
        "`leaked`, `stale`.\n"
        "4. Fill `human_confidence` with high, medium, or low, and add `human_notes` where the "
        "automatic label seems wrong.\n"
        "5. Do **not** edit any other column: agreement is computed against the recorded "
        "`judge_label`.\n"
        "6. Run `uv run memcheck audit --agreement results/audit/audit_sample.csv`. The output "
        "reports agreement over every labelled row and, separately, over the rows whose "
        "correct/incorrect decision was made by the LLM judge (`judge_*` keys); abstentions "
        "and false-premise questions are graded deterministically and do not test the judge.\n\n"
        "The sample is deterministic: the same results always produce the same rows, so two "
        "reviewers audit the same cases.\n",
        encoding="utf-8",
    )
    return AuditExport(
        path=path,
        rows=len(sample),
        strata_counts=(
            {str(key): int(value) for key, value in sample["stratum"].value_counts().items()}
            if not sample.empty
            else {}
        ),
        judge_prompt_hash=prompt_hash("judge_v1"),
    )


def agreement(csv_path: Path) -> dict[str, float | int | str]:
    """Compute human/judge agreement, including Cohen's kappa, from a filled-in audit CSV."""
    frame = pd.read_csv(csv_path)
    labelled = frame[frame["human_label"].notna() & (frame["human_label"].astype(str) != "")]
    if labelled.empty:
        return {
            "rows": len(frame),
            "labelled": 0,
            "note": "no human labels present yet; fill in the human_label column first",
        }

    def normalize(label: str) -> str:
        label = str(label).strip().lower()
        if label.startswith("abstain"):
            return "abstained"
        if label in ("correct", "abstained_correctly"):
            return "correct" if label == "correct" else "abstained"
        return label

    def score(subset: pd.DataFrame) -> dict[str, float | int]:
        human = subset["human_label"].map(normalize)
        judge = subset["judge_label"].map(normalize)
        agreed = int((human.to_numpy() == judge.to_numpy()).sum())
        total = len(subset)
        observed = agreed / total
        categories = sorted(set(human) | set(judge))
        expected = sum(
            (human == category).mean() * (judge == category).mean() for category in categories
        )
        kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
        return {
            "labelled": total,
            "agreements": agreed,
            "observed_agreement": observed,
            "expected_agreement": expected,
            "cohens_kappa": kappa,
        }

    result: dict[str, float | int | str] = {"rows": len(frame), **score(labelled)}
    # Only the correct/incorrect decision on an answered, non-abstention query is the judge's
    # (``grade_answer``). Abstentions, false-premise questions, leaks and stale values are graded
    # deterministically, so agreement on those rows says nothing about the judge.
    judged = labelled[labelled.apply(_decided_by_judge, axis=1)]
    if not judged.empty:
        result.update({f"judge_{key}": value for key, value in score(judged).items()})
    else:
        result["judge_labelled"] = 0
    result["judge_prompt_hash"] = prompt_hash("judge_v1")
    return result


def _decided_by_judge(row: pd.Series) -> bool:
    """True when the row's label was the LLM judge's correct/incorrect verdict.

    The export records the judge's rationale on exactly those rows; deterministic grades carry
    either no rationale or a ``grader=`` marker.
    """
    if str(row.get("judge_label", "")).strip().lower() not in ("correct", "incorrect"):
        return False
    rationale = row.get("judge_rationale")
    if rationale is None or (isinstance(rationale, float) and pd.isna(rationale)):
        return False
    text = str(rationale).strip()
    return bool(text) and not text.startswith("grader=")
