"""Blinded human audit of the LongMemEval judge (DECISIONS D-027).

The first audit (DEVIATIONS V-017) had three design faults: it showed the retrieved evidence,
which led the rater to grade against it rather than the gold answer; it balanced the sample on
the judge's own label, which fixes the judge's marginals at one half and depresses kappa; and it
counted seed replicas as separate rows. This sample fixes all three:

* the unit is a distinct (question, answer) pair, so a replica is shown once;
* strata are LongMemEval question types, never the judge's verdict;
* the rater sees the question, its date, the gold answer, the grading criterion for that
  question type in plain words, and the answer. Evidence, condition, and every judge label are
  in a separate key file the rater does not open.

Answers that abstain on an answerable question are left out: every grader marks them incorrect,
so they cannot reveal a disagreement.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from memcheck.models import content_hash

#: The official LongMemEval criterion for each question type, in plain words (D-025).
CRITERIA: dict[str, str] = {
    "default": (
        "Correct if the answer contains the gold answer, or all the steps needed to reach it. "
        "An answer with only part of the required information is incorrect."
    ),
    "temporal-reasoning": (
        "Correct if the answer contains the gold answer, or all the steps needed to reach it. "
        "An off-by-one error in a number of days, weeks or months is still correct."
    ),
    "knowledge-update": (
        "Correct if the answer gives the updated (latest) value in the gold answer. Mentioning "
        "older information as well is fine."
    ),
    "single-session-preference": (
        "The gold field is a rubric. Correct if the answer recalls and uses the user's personal "
        "information correctly. It does not need to cover every point of the rubric."
    ),
    "abstention": (
        "The question cannot be answered from the conversation history; the gold field explains "
        "why. Correct if the answer says the information is missing or not mentioned. Incorrect "
        "if it gives an answer."
    ),
}

BLIND_COLUMNS = [
    "audit_id",
    "question_type",
    "question_date",
    "question",
    "gold_answer",
    "criterion",
    "model_answer",
    "human_label",
    "human_notes",
]

KEY_COLUMNS = [
    "audit_id",
    "query_id",
    "question_type",
    "official_label",
    "judge_v1_label",
    "conditions",
    "runs",
]


@dataclass
class LmeAuditExport:
    blind_path: Path
    key_path: Path
    rows: int
    per_type: dict[str, int]
    population: int


def _criterion_key(question_id: str, question_type: str) -> str:
    if question_id.endswith("_abs"):
        return "abstention"
    return question_type if question_type in CRITERIA else "default"


def _pairs(run_dirs: list[Path], references: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per distinct (question, answer) across the given re-judged runs."""
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        for line in (run_dir / "grades.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            grade = json.loads(line)
            metadata = grade.get("metadata") or {}
            rows.append(
                {
                    "run_id": run_dir.name,
                    "variant": "+".join(
                        [str(grade.get("backend_name", "")), *grade.get("ablations", [])]
                    ),
                    "query_id": str(grade["query_id"]),
                    "model_answer": str(grade.get("model_answer") or ""),
                    "official_label": grade["label"],
                    "judge_v1_label": metadata.get("source_label", ""),
                }
            )
    frame = pd.DataFrame(rows)
    frame = frame[frame["query_id"].isin(references)]
    abstention_question = frame["query_id"].str.endswith("_abs")
    trivial = frame["model_answer"].str.startswith("ABSTAIN") & ~abstention_question
    frame = frame[~trivial]
    grouped = (
        frame.groupby(["query_id", "model_answer"], sort=True)
        .agg(
            official_label=("official_label", "first"),
            judge_v1_label=("judge_v1_label", "first"),
            conditions=("variant", lambda values: ";".join(sorted(set(values)))),
            runs=("run_id", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
    )
    grouped["question_type"] = grouped["query_id"].map(
        lambda qid: _criterion_key(qid, references[qid]["question_type"])
    )
    grouped["audit_id"] = grouped.apply(
        lambda row: content_hash([row["query_id"], row["model_answer"]])[:12], axis=1
    )
    return grouped


def export(
    run_dirs: list[Path],
    references_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 60,
    seed: int = 20261103,
) -> LmeAuditExport:
    """Write ``audit_blind.csv`` for the rater and ``audit_key.csv`` for scoring."""
    references = {
        entry["question_id"]: entry
        for entry in json.loads(references_path.read_text(encoding="utf-8"))
    }
    pairs = _pairs(run_dirs, references)
    types = sorted(pairs["question_type"].unique())
    rng = random.Random(seed)
    per_type = max(1, sample_size // len(types))
    chosen: list[pd.DataFrame] = []
    for question_type in types:
        pool = pairs[pairs["question_type"] == question_type].sort_values("audit_id")
        indices = list(pool.index)
        rng.shuffle(indices)
        chosen.append(pool.loc[indices[:per_type]])
    sample = pd.concat(chosen)
    # Shuffle the presentation order so neighbouring rows do not share a type.
    order = list(sample.index)
    rng.shuffle(order)
    sample = sample.loc[order]

    output_dir.mkdir(parents=True, exist_ok=True)
    blind = pd.DataFrame(
        {
            "audit_id": sample["audit_id"],
            "question_type": sample["question_type"],
            "question_date": sample["query_id"].map(
                lambda qid: references[qid].get("question_date", "")
            ),
            "question": sample["query_id"].map(lambda qid: references[qid]["question"]),
            "gold_answer": sample["query_id"].map(lambda qid: str(references[qid]["answer"])),
            "criterion": sample["question_type"].map(CRITERIA),
            "model_answer": sample["model_answer"],
            "human_label": "",
            "human_notes": "",
        },
        columns=BLIND_COLUMNS,
    )
    blind_path = output_dir / "audit_blind.csv"
    blind.to_csv(blind_path, index=False, quoting=csv.QUOTE_ALL)
    key_path = output_dir / "audit_key.csv"
    sample[KEY_COLUMNS].to_csv(key_path, index=False)
    (output_dir / "README.md").write_text(
        "# Blinded audit of the LongMemEval judge\n\n"
        "Open `audit_blind.csv` only. Do not open `audit_key.csv` until you have finished.\n\n"
        "For each row:\n\n"
        "1. Read `question`, `gold_answer` and `criterion`. The criterion is the benchmark's own "
        "rule for that question type.\n"
        "2. Read `model_answer` and decide whether it meets the criterion **against the gold "
        "answer**. You are not shown the memory evidence on purpose: grade only the answer.\n"
        "3. Write `correct` or `incorrect` in `human_label`. If you truly cannot decide, write "
        "`unsure` and say why in `human_notes`; such rows are reported, not scored.\n"
        "4. Change no other column.\n\n"
        "Then run `uv run memcheck audit --lme-agreement "
        f"{blind_path.as_posix()} --lme-key {key_path.as_posix()}`.\n\n"
        f"The sample has {len(blind)} distinct (question, answer) pairs, {per_type} per question "
        f"type, drawn with seed {seed} from {len(pairs)} distinct pairs in: "
        + ", ".join(run_dir.name for run_dir in run_dirs)
        + ". Answers that abstain on an answerable question are excluded, because every grader "
        "marks them incorrect.\n",
        encoding="utf-8",
    )
    return LmeAuditExport(
        blind_path=blind_path,
        key_path=key_path,
        rows=len(blind),
        per_type={str(k): int(v) for k, v in blind["question_type"].value_counts().items()},
        population=len(pairs),
    )


def _binary(label: object) -> str | None:
    text = str(label).strip().lower()
    if text in ("correct", "abstained_correctly"):
        return "correct"
    if text in ("incorrect", "unsupported", "abstained_wrongly", "stale", "leaked"):
        return "incorrect"
    return None


def _kappa(left: list[str], right: list[str]) -> tuple[float, float]:
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / total
    expected = sum(
        (left.count(category) / total) * (right.count(category) / total)
        for category in ("correct", "incorrect")
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return observed, kappa


def _score(
    left: list[str], right: list[str], *, seed: int, resamples: int = 2000
) -> dict[str, Any]:
    observed, kappa = _kappa(left, right)
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(resamples):
        picks = [rng.randrange(len(left)) for _ in left]
        _, value = _kappa([left[i] for i in picks], [right[i] for i in picks])
        if value == value:  # a resample where both raters used one label has no kappa
            boots.append(value)
    boots.sort()
    low = boots[int(0.025 * len(boots))] if boots else float("nan")
    high = boots[int(0.975 * len(boots)) - 1] if boots else float("nan")
    confusion = {
        f"{a}/{b}": sum(1 for x, y in zip(left, right, strict=True) if (x, y) == (a, b))
        for a in ("correct", "incorrect")
        for b in ("correct", "incorrect")
    }
    return {
        "n": len(left),
        "agreements": sum(a == b for a, b in zip(left, right, strict=True)),
        "observed_agreement": observed,
        "cohens_kappa": kappa,
        "kappa_ci95_bootstrap": [low, high],
        "confusion_left_right": confusion,
    }


def agreement(blind_path: Path, key_path: Path, *, seed: int = 20261103) -> dict[str, Any]:
    """Human against each judge, and the two judges against each other, on the labelled rows."""
    blind = pd.read_csv(blind_path, dtype=str, keep_default_na=False)
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
    merged = blind.merge(key, on="audit_id", how="inner", suffixes=("", "_key"))
    merged["human"] = merged["human_label"].map(_binary)
    merged["official"] = merged["official_label"].map(_binary)
    merged["judge_v1"] = merged["judge_v1_label"].map(_binary)
    unsure = merged[merged["human_label"].str.strip().str.lower() == "unsure"]
    labelled = merged[merged["human"].notna()]
    result: dict[str, Any] = {
        "rows": len(merged),
        "labelled": len(labelled),
        "unsure": len(unsure),
        "unlabelled": len(merged) - len(labelled) - len(unsure),
    }
    if labelled.empty:
        result["note"] = "no human labels yet"
        return result
    result["human_vs_official_judge"] = _score(
        list(labelled["human"]), list(labelled["official"]), seed=seed
    )
    with_v1 = labelled[labelled["judge_v1"].notna()]
    if not with_v1.empty:
        result["human_vs_judge_v1"] = _score(
            list(with_v1["human"]), list(with_v1["judge_v1"]), seed=seed
        )
        result["official_vs_judge_v1"] = _score(
            list(with_v1["official"]), list(with_v1["judge_v1"]), seed=seed
        )
    result["by_question_type"] = {
        str(question_type): {
            "n": len(group),
            "agree_with_official": int((group["human"] == group["official"]).sum()),
        }
        for question_type, group in labelled.groupby("question_type")
    }
    result["disagreements_with_official"] = labelled[labelled["human"] != labelled["official"]][
        ["audit_id", "question_type", "human_label", "official_label", "human_notes"]
    ].to_dict(orient="records")
    return result


def judge_agreement(
    run_dirs: list[Path], references_path: Path, *, seed: int = 20261103
) -> dict[str, Any]:
    """The two graders against each other on every distinct (question, answer) pair.

    The population is the same one the human sample is drawn from: trivial abstentions on
    answerable questions are excluded, because every grader marks them incorrect.
    """
    references = {
        entry["question_id"]: entry
        for entry in json.loads(references_path.read_text(encoding="utf-8"))
    }
    pairs = _pairs(run_dirs, references)
    official = list(pairs["official_label"].map(_binary))
    original = list(pairs["judge_v1_label"].map(_binary))
    result: dict[str, Any] = {
        "runs": [run_dir.name for run_dir in run_dirs],
        "official_vs_judge_v1": _score(official, original, seed=seed),
        "by_question_type": {},
    }
    for question_type, group in pairs.groupby("question_type"):
        left = list(group["official_label"].map(_binary))
        right = list(group["judge_v1_label"].map(_binary))
        result["by_question_type"][str(question_type)] = {
            "n": len(group),
            "agreements": sum(a == b for a, b in zip(left, right, strict=True)),
            "official_correct": left.count("correct"),
            "judge_v1_correct": right.count("correct"),
        }
    return result
