"""Token counting.

The default counter is a deterministic word/punctuation approximation so that budgets are
identical across backends, machines, and providers with no tokenizer download. Real providers
override :func:`count_tokens` through their own ``count_tokens`` method; the *counter used for
budget enforcement* is recorded in the run manifest so budgets are never compared across
different tokenizers.
"""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

#: Function words that carry no evidential content. Shared by the retrieval index and the
#: reader so that "matching" means the same thing on both sides: a document that matches a query
#: only on words like "the" or "is" is not evidence, and must not be returned as though it were.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "do",
        "does",
        "did",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "current",
        "currently",
        "now",
        "please",
        "tell",
        "me",
        "our",
        "we",
        "you",
        "your",
        "it",
        "its",
        "this",
        "that",
        "and",
        "or",
        "as",
        "from",
        "about",
        "there",
        "their",
    }
)


def count_tokens(text: str) -> int:
    """Approximate token count: words plus standalone punctuation."""
    if not text:
        return 0
    return len(_TOKEN_PATTERN.findall(text))


def truncate_to_tokens(text: str, limit: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``limit`` tokens. Returns the text and whether it was truncated."""
    if limit <= 0:
        return "", bool(text)
    tokens = _TOKEN_PATTERN.findall(text)
    if len(tokens) <= limit:
        return text, False
    kept = tokens[:limit]
    # Rebuild by slicing the original at the end offset of the last kept token so the
    # truncated text stays a genuine prefix of the input.
    matches = list(_TOKEN_PATTERN.finditer(text))
    end = matches[len(kept) - 1].end()
    return text[:end], True
