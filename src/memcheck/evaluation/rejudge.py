"""Re-grade a stored LongMemEval run with the benchmark's official judge (DECISIONS D-025).

The source run is never modified. Its answers are read from ``grades.jsonl``, each distinct
(question, answer) pair is judged once with the per-question-type prompt from LongMemEval's own
``src/evaluation/evaluate_qa.py``, and a new run directory is written beside it. That run has
the source's artifacts unchanged, ``grades.jsonl`` re-labelled, the judge's own calls in
``judge_calls.jsonl`` (kept apart from ``costs.jsonl``, so grading is not billed to a memory
system), and a manifest whose judge prompt hash differs from the source's, so the fairness
validator never pools the two.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memcheck.models import ProviderCall, content_hash
from memcheck.utilities.hashing import hash_directory
from memcheck.utilities.jsonl import JsonlWriter

#: Upstream source of the templates below, pinned to the commit they were copied from.
OFFICIAL_SOURCE = (
    "https://github.com/xiaowu0162/LongMemEval/blob/"
    "d6dc8b50a2d9ac0c99485ea28fa5755c62414c34/src/evaluation/evaluate_qa.py"
)
JUDGE_ID = "judge_lme_official_v1"
#: The metric model the official script names for its reported numbers.
OFFICIAL_MODEL = "gpt-4o-2024-08-06"
#: The official script asks for a one-word verdict.
OFFICIAL_MAX_TOKENS = 10

_PREFIX = (
    "I will give you a question, a correct answer, and a response from a model. Please answer "
    "yes if the response contains the correct answer. Otherwise, answer no. "
)
_STEPS = (
    "If the response is equivalent to the correct answer or contains all the intermediate steps "
    "to get the correct answer, you should also answer yes. If the response only contains a "
    "subset of the information required by the answer, answer no. "
)
_TAIL = (
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

#: Verbatim from the pinned upstream file, including its spacing.
TEMPLATES: dict[str, str] = {
    "default": _PREFIX + _STEPS + _TAIL,
    "temporal-reasoning": _PREFIX
    + _STEPS
    + "In addition, do not penalize off-by-one errors for the number of days. If the question "
    "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
    "(e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
    + _TAIL,
    "knowledge-update": _PREFIX
    + "If the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the required "
    "answer." + _TAIL,
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response "
        "from a model. Please answer yes if the response satisfies the desired response. "
        "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
        "The response is correct as long as it recalls and utilizes the user's personal "
        "information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. The "
        "model could say that the information is incomplete, or some other information is given "
        "but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel "
        "Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer "
        "yes or no only."
    ),
}

#: Hash of every template, recorded in the derived manifest's prompt hashes.
TEMPLATES_HASH = content_hash(json.dumps(TEMPLATES, sort_keys=True))

#: Question types the official script grades with its default template.
_DEFAULT_TYPES = ("single-session-user", "single-session-assistant", "multi-session")


def official_prompt(
    question_type: str, question_id: str, question: str, gold: str, answer: str
) -> str:
    """The prompt the official script would build for this question."""
    if "_abs" in question_id:
        template = TEMPLATES["abstention"]
    elif question_type in _DEFAULT_TYPES:
        template = TEMPLATES["default"]
    elif question_type in TEMPLATES:
        template = TEMPLATES[question_type]
    else:
        raise ValueError(f"no official template for question type {question_type!r}")
    return template.format(question, gold, answer)


def official_verdict(text: str) -> bool:
    """The official parser: the verdict is yes if 'yes' appears anywhere in the reply."""
    return "yes" in text.lower()


def relabel(original: dict[str, Any], verdict: bool) -> str:
    """Map a yes/no verdict onto the harness's failure taxonomy."""
    abstention_question = str(original["query_id"]).endswith("_abs")
    answered_with_abstention = str(original.get("model_answer") or "").startswith("ABSTAIN")
    if abstention_question:
        return "abstained_correctly" if verdict else "unsupported"
    if verdict:
        return "correct"
    return "abstained_wrongly" if answered_with_abstention else "incorrect"


@dataclass
class RejudgeResult:
    run_dir: Path
    judged_pairs: int
    rows: int
    changed_rows: int
    cost_usd: float


#: Takes a prompt, returns the judge's reply and the provider call that produced it.
Judge = Callable[[str], tuple[str, ProviderCall]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def unique_pairs(source: Path) -> int:
    """How many judge calls a re-grade of ``source`` will make, for the cost gate."""
    rows = _read_jsonl(source / "grades.jsonl")
    return len({(str(row["query_id"]), str(row.get("model_answer") or "")) for row in rows})


def rejudge(
    source: Path,
    references: Path,
    output_root: Path,
    judge: Judge,
    *,
    judge_model: str = OFFICIAL_MODEL,
) -> RejudgeResult:
    """Write ``<source>_rejudged_lme`` with every answer graded by the official judge."""
    references_by_id = {
        entry["question_id"]: entry for entry in json.loads(references.read_text(encoding="utf-8"))
    }
    grades = _read_jsonl(source / "grades.jsonl")
    target = output_root / f"{source.name}_rejudged_lme"
    if target.exists():
        raise FileExistsError(f"{target} exists; raw results are written once")

    verdicts: dict[tuple[str, str], tuple[bool, str]] = {}
    calls: list[dict[str, Any]] = []
    for row in grades:
        key = (str(row["query_id"]), str(row.get("model_answer") or ""))
        if key in verdicts:
            continue
        reference = references_by_id[key[0]]
        prompt = official_prompt(
            reference["question_type"],
            key[0],
            reference["question"],
            str(reference["answer"]),
            key[1],
        )
        reply, call = judge(prompt)
        verdicts[key] = (official_verdict(reply), reply)
        calls.append(
            {
                **call.model_dump(mode="json"),
                "query_id": key[0],
                "prompt_sha256": content_hash(prompt),
            }
        )

    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("checksums.json", "grades.jsonl", "manifest.json"),
    )
    changed = 0
    with JsonlWriter(target / "grades.jsonl") as writer:
        for row in grades:
            verdict, reply = verdicts[(str(row["query_id"]), str(row.get("model_answer") or ""))]
            label = relabel(row, verdict)
            changed += label != row["label"]
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    "question_type": references_by_id[str(row["query_id"])]["question_type"],
                    "source_label": row["label"],
                    "source_grader": row.get("grader"),
                }
            )
            writer.write(
                {
                    **row,
                    "label": label,
                    "correct": label in ("correct", "abstained_correctly"),
                    "grader": "llm_judge",
                    "judge_rationale": reply[:300],
                    "metadata": metadata,
                }
            )
    with JsonlWriter(target / "judge_calls.jsonl") as writer:
        writer.write_many(calls)

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    hashes = {
        name: value
        for name, value in manifest.get("prompt_hashes", {}).items()
        if name != "judge_v1"
    }
    hashes[JUDGE_ID] = TEMPLATES_HASH
    manifest["prompt_hashes"] = hashes
    manifest["run_id"] = target.name
    manifest["notes"] = (
        f"derived_from={source.name}; judge={judge_model} via {JUDGE_ID} ({OFFICIAL_SOURCE}); "
        + str(manifest.get("notes", ""))
    )
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    checksums = hash_directory(target, exclude=("checksums.json",))
    (target / "checksums.json").write_text(
        json.dumps(
            {"run_id": target.name, "algorithm": "sha256", "files": checksums},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return RejudgeResult(
        run_dir=target,
        judged_pairs=len(verdicts),
        rows=len(grades),
        changed_rows=changed,
        cost_usd=sum(float(call.get("estimated_cost_usd") or 0.0) for call in calls),
    )
