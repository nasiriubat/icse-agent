"""Stable provider interfaces: chat, structured extraction, embedding, tokens, cost.

Provider-specific SDKs are optional extras. Nothing in this module imports them.
"""

from __future__ import annotations

import abc
from typing import Any

from memcheck.models import ChatResponse, EmbeddingResponse, ProviderCall, StrictModel
from memcheck.utilities.tokens import count_tokens as approx_count_tokens


class ChatMessage(StrictModel):
    role: str
    content: str


class ChatRequest(StrictModel):
    """Everything that identifies a request. The cache key is the hash of this object."""

    model_identifier: str
    system_prompt: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_output_tokens: int = 512
    tools: list[str] = []
    response_schema: dict[str, Any] | None = None
    purpose: str = "answer"
    # Structured payload passed alongside the rendered prompt. Real providers ignore it and read
    # the rendered messages; the deterministic mock reader consumes it (DECISIONS D-006).
    structured_input: dict[str, Any] | None = None
    # Independent repetition of an otherwise identical request. Non-zero values enter the cache
    # key so a repeat run makes fresh calls; zero leaves the key as it was before the field.
    attempt: int = 0


class ProviderError(RuntimeError):
    """Raised for provider misconfiguration, refusal, or exhausted retries."""


class ProviderLengthError(ProviderError):
    """The model stopped on its output budget before emitting any content.

    Kept distinct from other provider failures because it is the one a caller can sensibly
    retry with a larger budget, whereas a refusal or an exhausted backoff cannot be.
    """


class Provider(abc.ABC):
    """Interface every provider implements."""

    name: str = "abstract"
    is_paid: bool = True

    @abc.abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a response. Must populate a :class:`ProviderCall` with full telemetry."""

    def extract(self, request: ChatRequest) -> ChatResponse:
        """Structured extraction. Default: a chat call carrying a response schema."""
        if request.response_schema is None:
            raise ProviderError("extract() requires request.response_schema")
        return self.chat(request)

    @abc.abstractmethod
    def embed(self, texts: list[str], *, model_identifier: str | None = None) -> EmbeddingResponse:
        """Embed texts. Providers without embeddings raise :class:`ProviderError`."""

    def count_tokens(self, text: str) -> int:
        """Token count used for budgets. The counter identity is recorded in the manifest."""
        return approx_count_tokens(text)

    @abc.abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int, model_identifier: str) -> float:
        """Monetary estimate in USD from the dated pricing table."""

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


def empty_call(provider: str, model_identifier: str, purpose: str = "") -> ProviderCall:
    return ProviderCall(
        provider=provider,
        model_identifier=model_identifier,
        request_identifier="none",
        purpose=purpose,
    )
