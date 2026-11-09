"""Anthropic provider.

The ``anthropic`` SDK is an optional extra; this module imports it lazily so the package,
the tests, and CI all work without it installed.

Paid calls are gated twice: once by :func:`memcheck.config.paid_run_authorization` before a run
starts, and once here per request against the run's remaining budget.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

from memcheck.config import PricingTable
from memcheck.models import ChatResponse, EmbeddingResponse, ProviderCall, content_hash
from memcheck.providers.base import ChatRequest, Provider, ProviderError
from memcheck.providers.cache import ResponseCache, request_key, stamp_attempt

# Models that reject sampling parameters (see the Anthropic migration guide: `temperature`,
# `top_p`, and `top_k` return 400 on these). Sending them anyway would fail the run.
_NO_SAMPLING_PARAMS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-fable-5",
    "claude-sonnet-5",
)

_RETRYABLE_STATUS = (408, 409, 429, 500, 502, 503, 529)


class AnthropicProvider(Provider):
    """Chat and structured extraction through the Anthropic Messages API."""

    name = "anthropic"
    is_paid = True

    def __init__(
        self,
        *,
        pricing: PricingTable,
        cache: ResponseCache | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        seed: int = 0,
        attempt: int = 0,
    ) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. MemCheck reads credentials from the "
                "environment only; it never accepts a key as a config value."
            )
        try:
            import anthropic
        except ImportError as error:  # pragma: no cover - optional extra
            raise ProviderError(
                "the anthropic extra is not installed; run: uv pip install -e '.[anthropic]'"
            ) from error
        self._client = anthropic.Anthropic()
        self._errors = anthropic
        self.pricing = pricing
        self.cache = cache
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._rng = random.Random(seed)
        self.attempt = attempt

    def chat(self, request: ChatRequest) -> ChatResponse:
        request = stamp_attempt(request, self.attempt)
        key = request_key(self.name, request)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        payload: dict[str, Any] = {
            "model": request.model_identifier,
            "max_tokens": request.max_output_tokens,
            "system": request.system_prompt,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
        }
        # Sampling parameters are rejected outright by current Opus/Sonnet models. Behaviour is
        # steered by the prompt instead, which is what the fairness validator compares anyway.
        if not request.model_identifier.startswith(_NO_SAMPLING_PARAMS):
            payload["temperature"] = request.temperature
        if request.response_schema is not None:
            payload["output_config"] = {
                "format": {"type": "json_schema", "schema": request.response_schema}
            }

        started = time.perf_counter()
        response, retries = self._call_with_backoff(payload)
        latency_ms = (time.perf_counter() - started) * 1000

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        input_tokens = int(response.usage.input_tokens)
        output_tokens = int(response.usage.output_tokens)
        call = ProviderCall(
            provider=self.name,
            model_identifier=response.model,
            # The SDK's request id, never the key: nothing derived from a secret is stored.
            request_identifier=str(
                getattr(response, "_request_id", None) or content_hash(key)[:16]
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            retry_count=retries,
            estimated_cost_usd=self.estimate_cost(
                input_tokens, output_tokens, request.model_identifier
            ),
            purpose=request.purpose,
        )
        result = ChatResponse(
            text=text, call=call, finish_reason=str(response.stop_reason or "stop")
        )
        if self.cache is not None:
            self.cache.put(key, self.name, request.model_identifier, result)
        return result

    def _call_with_backoff(self, payload: dict[str, Any]) -> tuple[Any, int]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._client.messages.create(**payload), attempt
            except self._errors.APIStatusError as error:
                if error.status_code not in _RETRYABLE_STATUS:
                    raise ProviderError(
                        f"anthropic request failed with status {error.status_code}"
                    ) from error
                last_error = error
            except self._errors.APIConnectionError as error:
                last_error = error
            delay = min(self.base_delay * 2**attempt + self._rng.random(), self.max_delay)
            time.sleep(delay)
        raise ProviderError(
            f"anthropic request failed after {self.max_retries} attempts"
        ) from last_error

    def embed(self, texts: list[str], *, model_identifier: str | None = None) -> EmbeddingResponse:
        raise ProviderError(
            "Anthropic does not expose an embedding endpoint; use the local or mock "
            "embedding provider and record the identifier in the manifest"
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int, model_identifier: str) -> float:
        return self.pricing.cost_for(f"anthropic:{model_identifier}", input_tokens, output_tokens)
