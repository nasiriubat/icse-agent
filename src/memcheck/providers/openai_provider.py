"""OpenAI provider (optional extra).

Implemented for parity so a cross-provider comparison is possible; not exercised in the shipped
results, which use the deterministic mock (see DEVIATIONS V-007).
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from memcheck.config import PricingTable
from memcheck.models import ChatResponse, EmbeddingResponse, ProviderCall, content_hash
from memcheck.providers.base import ChatRequest, Provider, ProviderError, ProviderLengthError
from memcheck.providers.cache import ResponseCache, request_key, stamp_attempt

#: Reasoning models reject any sampling temperature but their default of 1 (HTTP 400,
#: ``unsupported_value``), so none is sent to them. Every other model receives the configured
#: value, including 0, which a truthiness check would silently drop.
_FIXED_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def accepts_temperature(model_identifier: str) -> bool:
    """Whether the API accepts a ``temperature`` other than the default for this model."""
    name = model_identifier.lower()
    if name.startswith("gpt-5-chat"):
        return True
    return not name.startswith(_FIXED_TEMPERATURE_PREFIXES)


class OpenAIProvider(Provider):
    name = "openai"
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
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError("OPENAI_API_KEY is not set")
        try:
            import openai
        except ImportError as error:  # pragma: no cover - optional extra
            raise ProviderError(
                "the openai extra is not installed; run: uv pip install -e '.[openai]'"
            ) from error
        self._client = openai.OpenAI()
        self._openai = openai
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
            "max_completion_tokens": request.max_output_tokens,
            # An empty system prompt is omitted rather than sent blank, so a protocol that
            # specifies a single user message (the LongMemEval judge) is reproduced exactly.
            "messages": (
                [{"role": "system", "content": request.system_prompt}]
                if request.system_prompt
                else []
            )
            + [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if accepts_temperature(request.model_identifier):
            payload["temperature"] = request.temperature
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "memcheck",
                    "schema": request.response_schema,
                    "strict": True,
                },
            }

        started = time.perf_counter()
        response, retries = self._call_with_backoff(payload)
        latency_ms = (time.perf_counter() - started) * 1000

        text = response.choices[0].message.content or ""
        finish = str(response.choices[0].finish_reason or "stop")
        if not text.strip() and finish == "length":
            # Reasoning models spend max_completion_tokens on reasoning before emitting content,
            # so a budget that looks generous for a short answer can yield no content at all.
            # Failing loudly matters: a silently empty answer is indistinguishable from a real
            # one downstream, and 47% of a run was scored on empty completions before this check.
            raise ProviderLengthError(
                f"{request.model_identifier} returned no content and stopped on 'length': the "
                f"max_output_tokens budget of {request.max_output_tokens} was consumed by "
                "reasoning. Raise model.max_output_tokens (reasoning models need several "
                "thousand) or choose a non-reasoning model."
            )
        input_tokens = int(response.usage.prompt_tokens)
        output_tokens = int(response.usage.completion_tokens)
        call = ProviderCall(
            provider=self.name,
            model_identifier=response.model,
            request_identifier=str(getattr(response, "id", None) or content_hash(key)[:16]),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            retry_count=retries,
            estimated_cost_usd=self.estimate_cost(
                input_tokens, output_tokens, request.model_identifier
            ),
            purpose=request.purpose,
        )
        structured: dict[str, Any] | None = None
        if request.response_schema is not None:
            # A malformed reply stays unstructured; the caller then treats it as no edits.
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            structured = parsed if isinstance(parsed, dict) else None
        result = ChatResponse(text=text, call=call, finish_reason=finish, structured=structured)
        if self.cache is not None:
            self.cache.put(key, self.name, request.model_identifier, result)
        return result

    def _call_with_backoff(self, payload: dict[str, Any]) -> tuple[Any, int]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._client.chat.completions.create(**payload), attempt
            except (self._openai.RateLimitError, self._openai.APIConnectionError) as error:
                # An exhausted credit balance is reported as a 429 like a transient rate limit,
                # but no amount of waiting fixes it; retrying only delays the failure.
                if "insufficient_quota" in str(error) or "credit_balance_exhausted" in str(error):
                    raise ProviderError(
                        "openai refused the request: the account has no credits remaining"
                    ) from error
                last_error = error
            except self._openai.APIStatusError as error:
                if error.status_code < 500:
                    raise ProviderError(
                        f"openai request failed with status {error.status_code}"
                    ) from error
                last_error = error
            time.sleep(min(self.base_delay * 2**attempt + self._rng.random(), self.max_delay))
        raise ProviderError(
            f"openai request failed after {self.max_retries} attempts"
        ) from last_error

    def embed(self, texts: list[str], *, model_identifier: str | None = None) -> EmbeddingResponse:
        model = model_identifier or "text-embedding-3-small"
        response = self._client.embeddings.create(model=model, input=texts)
        call = ProviderCall(
            provider=self.name,
            model_identifier=model,
            request_identifier=str(getattr(response, "id", "embed")),
            input_tokens=int(response.usage.prompt_tokens),
            purpose="embed",
        )
        return EmbeddingResponse(vectors=[item.embedding for item in response.data], call=call)

    def estimate_cost(self, input_tokens: int, output_tokens: int, model_identifier: str) -> float:
        return self.pricing.cost_for(f"openai:{model_identifier}", input_tokens, output_tokens)
