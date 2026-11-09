"""Provider registry and factory."""

from __future__ import annotations

from pathlib import Path

from memcheck.config import ExperimentConfig, PricingTable, load_pricing, paid_run_authorization
from memcheck.providers.base import ChatMessage, ChatRequest, Provider, ProviderError
from memcheck.providers.cache import ResponseCache
from memcheck.providers.mock import MockProvider
from memcheck.providers.prompts import all_prompt_hashes, get_prompt, prompt_hash

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "MockProvider",
    "Provider",
    "ProviderError",
    "ResponseCache",
    "all_prompt_hashes",
    "build_provider",
    "get_prompt",
    "prompt_hash",
]


def build_provider(
    config: ExperimentConfig,
    *,
    cache_path: str | Path | None = None,
    seed: int = 0,
) -> Provider:
    """Construct the provider named by the configuration.

    Paid providers are refused unless both environment controls authorize the run. The check is
    here, at construction, so no code path can reach a paid endpoint without passing it.
    """
    if config.model.provider == "mock":
        return MockProvider(identifier=config.model.identifier)

    authorization = paid_run_authorization()
    if not authorization.allowed:
        raise ProviderError(
            f"refusing to build the paid provider {config.model.qualified_name!r}: "
            f"{authorization.reason}. Set ALLOW_PAID_RUNS=true and MAX_EXPERIMENT_COST_USD=<limit>."
        )

    pricing: PricingTable = load_pricing(config.cost.pricing_file)
    cache = ResponseCache(cache_path) if cache_path is not None else None

    if config.model.provider == "anthropic":
        from memcheck.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(pricing=pricing, cache=cache, seed=seed, attempt=config.attempt)
    if config.model.provider == "openai":
        from memcheck.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(pricing=pricing, cache=cache, seed=seed, attempt=config.attempt)
    raise ProviderError(f"unknown provider {config.model.provider!r}")
