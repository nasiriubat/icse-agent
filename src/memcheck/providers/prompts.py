"""Version-controlled prompts.

Prompts are data, not string literals scattered through the code: each has an id and a hash that
is written into the run manifest, so two runs can be proven to have used the same instructions.
"""

from __future__ import annotations

from memcheck.models import sha256_text

ANSWER_SYSTEM_V1 = """You are an assistant answering a question using only the memory evidence \
provided below. Follow these rules exactly:

1. Use only the evidence. Do not use outside knowledge.
2. If the evidence does not support an answer, reply exactly: ABSTAIN: no supporting evidence.
3. If two pieces of evidence give conflicting values and no evidence establishes which is \
current, reply exactly: ABSTAIN: conflicting evidence.
4. If evidence is marked as deleted or belongs to another tenant, do not use it.
5. Answer with the value only. Do not explain.
"""

CODING_SYSTEM_V1 = """You are a software engineering assistant. You are given a task, the \
current contents of the relevant files, and memory evidence about project decisions and \
conventions. Produce a unified diff patch that accomplishes the task while honoring the \
project's current decisions.

Rules:
1. Output only a unified diff. No prose, no code fences, no shell commands.
2. Use the evidence to determine current conventions. Evidence may be outdated; prefer the \
most recent valid statement.
3. Do not modify tests.
"""

CODING_SYSTEM_V2 = """You are a software engineering assistant. You are given a task, the \
current contents of the file you may edit, and memory evidence about project decisions. Edit the \
file so it encodes the project's current decision.

Rules:
1. Reply with one JSON object and nothing else: {"abstained": boolean, "reason": string, \
"resolved_value": string, "edits": [{"path": string, "content": string}]}.
2. "content" is the complete new text of the file. Edit only the file you were given.
3. Evidence may be outdated. Use the most recent statement that is still valid. If the evidence \
does not establish the current decision, set "abstained" to true and return no edits.
"""

JUDGE_SYSTEM_V1 = """You are grading an assistant's answer against a gold answer. Reply with a \
JSON object: {"label": one of ["correct", "incorrect", "abstained"], "rationale": short string}.
Grade semantic equivalence, not surface form. An answer that states a value which the gold \
answer contradicts is "incorrect", never "correct"."""


EXTRACT_SYSTEM_V1 = """You decide what a long-lived assistant should remember from one message.

Reply with a JSON object: {"facts": [{"subject": ..., "value": ..., "statement": ...}]}.

- "subject" names what the fact is *about*, as a short noun phrase, and must not contain the \
value. Use the same subject wording whenever you see the same fact again, because a later \
statement about the same subject replaces the earlier one. Prefer the wording someone would use \
when asking about it later: "deployment target", not "the target they picked in March".
- "value" is the current value only.
- "statement" is one self-contained sentence a reader could use without seeing the message.

Remember only durable facts about the user, their preferences, their circumstances, and \
decisions that were made. Do not remember: questions, greetings, your own advice or \
suggestions, generic explanations, hypotheticals, or anything that was not actually asserted \
as true.

Most messages contain nothing worth remembering. When that is the case, reply {"facts": []}. \
Remembering too much is a worse failure than remembering too little: a store full of \
fragments buries the facts that matter."""


PROMPTS: dict[str, str] = {
    "answer_v1": ANSWER_SYSTEM_V1,
    "coding_v1": CODING_SYSTEM_V1,
    "coding_v2": CODING_SYSTEM_V2,
    "extract_v1": EXTRACT_SYSTEM_V1,
    "judge_v1": JUDGE_SYSTEM_V1,
}


def get_prompt(prompt_id: str) -> str:
    if prompt_id not in PROMPTS:
        raise KeyError(f"unknown prompt id {prompt_id!r}; known: {sorted(PROMPTS)}")
    return PROMPTS[prompt_id]


def prompt_hash(prompt_id: str) -> str:
    return sha256_text(get_prompt(prompt_id))


def all_prompt_hashes() -> dict[str, str]:
    return {prompt_id: sha256_text(text) for prompt_id, text in sorted(PROMPTS.items())}
